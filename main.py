import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from sklearn.model_selection import train_test_split
import flax.linen as nn
import optax
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
from flax.struct import dataclass
import wandb
import time
from pprint import pprint
from jax.experimental import io_callback
from typing import Tuple
import numpy as np
from hydra.utils import instantiate

from picard_epo.problems.problem import Problem, BatchData


@dataclass
class TrainingState:
    params: dict
    opt_state: dict
    data_index: int = 0


def create_spo_loss(batch_optimize):
    """Create SPO+ loss function, for maximization, with custom gradients."""

    def spo_fun(pred_cost, true_cost, true_sol, true_obj):
        sol, obj = batch_optimize(2 * pred_cost - true_cost)
        loss = -obj + 2 * jnp.sum(pred_cost * true_sol, axis=1) - true_obj
        loss = jnp.mean(loss)
        return loss, sol

    def spo_fwd(pred_cost, true_cost, true_sol, true_obj):
        loss, sol = spo_fun(pred_cost, true_cost, true_sol, true_obj)
        return loss, (sol, true_sol)

    def spo_bwd(res, g):
        sol, true_sol = res
        grad = 2 * (true_sol - sol)
        # No gradients needed for true_cost, true_sol, or true_obj
        return grad * g, None, None, None

    @jax.custom_vjp
    def spo_loss(pred_cost, true_cost, true_sol, true_obj):
        return spo_fun(pred_cost, true_cost, true_sol, true_obj)[0]

    spo_loss.defvjp(spo_fwd, spo_bwd)
    return spo_loss


def create_loss_fn(model, spo_loss):
    """Create loss function for SPO+ training."""

    def loss_fn(params, batch_data: BatchData):
        """Compute SPO+ loss for a batch."""
        pred_c = model.apply(params, batch_data.x)
        loss = spo_loss(
            pred_c, batch_data.c, batch_data.true_sols, batch_data.true_objs
        )
        return loss

    return loss_fn


def create_train_epoch(optimizer, loss_fn, length, window_size, mesh):
    """Create function to train on batch data using scan."""

    def picard_mapper(training_state: TrainingState, batch_data: BatchData):
        loss, grads = jax.value_and_grad(loss_fn)(
            training_state.params, batch_data
        )
        return grads, loss

    def picard_reducer(
        training_state: TrainingState, grad
    ) -> Tuple[TrainingState, TrainingState]:
        updates, opt_state = optimizer.update(grad, training_state.opt_state)
        params = optax.apply_updates(training_state.params, updates)
        new_state = TrainingState(
            params=params,
            opt_state=opt_state,
            data_index=training_state.data_index + 1,
        )
        return new_state, new_state

    def shard_picard_mapper(
        training_state: TrainingState, batch_data: BatchData
    ):
        grads, losses = jax.vmap(picard_mapper, in_axes=(0, 0))(
            training_state, batch_data
        )
        grads = jax.lax.psum(grads, axis_name="batch")
        losses = jax.lax.pmean(losses, axis_name="batch")
        return grads, losses[:, None]

    def picard_iteration(state_and_data: Tuple[TrainingState, BatchData], _):
        training_state, batch_data = state_and_data
        window_batch = jax.tree.map(lambda x: x[:window_size], batch_data)
        if mesh is not None:
            grads, losses = shard_map(
                shard_picard_mapper,
                mesh=mesh,
                in_specs=(P("picard"), P("picard", "batch")),
                out_specs=(P("picard"), P("picard")),
                check_rep=False,
            )(training_state, window_batch)
        else:
            grads, losses = jax.vmap(picard_mapper, in_axes=(0, 0))(
                training_state, window_batch
            )

        init_state = jax.tree.map(
            lambda x: x[0],
            training_state,
        )
        new_training_state = jax.lax.scan(picard_reducer, init_state, grads)[1]
        new_batch_data = jax.tree.map(
            lambda x: jnp.roll(x, 1, axis=0), batch_data
        )
        return (new_training_state, new_batch_data), losses[0]

    def train_epoch(training_state: TrainingState, batch_data: BatchData):
        """Train on batch data using scan."""
        (final_state, final_batch), losses = jax.lax.scan(
            picard_iteration, (training_state, batch_data), length=length
        )
        return final_state, losses

    return train_epoch


def create_val_epoch(model, batch_optimize, loss_fn):
    """Create function to evaluate model on train and test datasets."""

    def val_step(params, batch_data: BatchData):
        """Evaluate model on entire dataset."""
        pred_c = model.apply(params, batch_data.x)
        loss = loss_fn(
            pred_c, batch_data.c, batch_data.true_sols, batch_data.true_objs
        )
        pred_sol, pred_obj = batch_optimize(pred_c)
        true_objs_pred = jnp.mean(
            jnp.sum(pred_sol * batch_data.c, axis=1), axis=0
        )
        subopt = 1 - true_objs_pred / jnp.mean(batch_data.true_objs)
        return {
            "loss": loss,
            "subopt": subopt,
            "pred_obj": pred_obj.mean(),
            "true_obj": true_objs_pred,
            "l2_err_c": jnp.mean(jnp.sum((pred_c - batch_data.c) ** 2, axis=1)),
            "l2_err_soln": jnp.mean(
                jnp.sum((pred_sol - batch_data.true_sols) ** 2, axis=1)
            ),
        }

    def val_epoch(params, train_data: BatchData, test_data: BatchData):
        """Evaluate model on full train and test datasets."""
        train_loss = val_step(params, train_data)
        val_loss = val_step(params, test_data)

        return {
            **{"train/" + k: v for k, v in train_loss.items()},
            **{"val/" + k: v for k, v in val_loss.items()},
        }

    return val_epoch


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # Convert DictConfig to our dataclass
    print(omegaconf.OmegaConf.to_yaml(config))

    # Initialize wandb
    wandb.init(
        project="picard-epo",
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Data Generation
    problem: Problem = instantiate(config.problem)
    dataset: BatchData = problem.dataset
    n = len(dataset)
    train_idx, test_idx = train_test_split(
        jnp.arange(n),
        test_size=config.training.test_size,
        random_state=config.training.random_seed,
    )

    # Ensure train / test sizes is divisible by mesh
    mesh_batch = (
        1 if config.training.mesh is None else config.training.mesh.batch
    )
    train_idx = train_idx[: (len(train_idx) // mesh_batch) * mesh_batch]
    test_idx = test_idx[: (len(test_idx) // mesh_batch) * mesh_batch]
    n_train = len(train_idx)
    n_test = len(test_idx)

    # TODO Warn here

    train_data = jax.tree.map(lambda x: x[train_idx], dataset)
    test_data = jax.tree.map(lambda x: x[test_idx], dataset)
    spo_loss = create_spo_loss(problem.batch_optimize)

    # Initialize model and optimizer
    model = instantiate(config.model)
    key = jax.random.PRNGKey(config.training.random_seed)
    params = model.init(key, dataset.x[0:1])

    optimizer = optax.sgd(learning_rate=config.training.learning_rate)
    opt_state = optimizer.init(params)

    # Initialize training state
    training_state = TrainingState(
        params=params, opt_state=opt_state, data_index=jnp.zeros(1)
    )

    # Create mesh from config
    if config.training.mesh is not None:
        devices = jax.devices()
        requested_devices = (
            config.training.mesh.batch * config.training.mesh.picard
        )
        devices = np.array(jax.devices()[:requested_devices])
        mesh = Mesh(
            devices.reshape(
                (config.training.mesh.picard, config.training.mesh.batch)
            ),
            axis_names=("picard", "batch"),
        )
    else:
        mesh = None

    loss_fn = create_loss_fn(model, spo_loss)

    train_epoch = create_train_epoch(
        optimizer,
        loss_fn,
        config.training.batches_per_eval,
        config.training.picard.window_size,
        mesh,
    )
    val_epoch = create_val_epoch(model, problem.batch_optimize, spo_loss)

    # Training loop
    print(f"Starting training for {config.training.n_epochs} epochs...")

    batch_size = config.training.batch_size

    # How many batches to split the training data. May require multiple
    # passes through training data if data is small.
    num_batches = max(
        # If training size is large enough, just use it
        n_train // batch_size,
        # Have to be able to create entire window, +batches_per_eval
        config.training.picard.window_size + config.training.batches_per_eval,
    )

    # Number of passes required through training data
    eff_n_train = num_batches * batch_size
    num_perms = int(np.ceil(eff_n_train / n_train))
    num_evals_per_epoch = max(
        (num_batches - config.training.picard.window_size)
        // config.training.batches_per_eval,
        1,
    )

    print(f"""
    Splitting {n_train} samples into {num_batches} batches of size {batch_size}
    (requires {num_perms} passes through training data).
    """)
    

    assert (
        num_batches >= config.training.picard.window_size
    ), "Number of batches per eval must be at least as large as Picard window size."

    start_time = time.time()

    # Create logging callbacks
    def log_metrics_callback(metrics, step):
        """Host callback for logging metrics to wandb."""
        print(f"** {time.time() - start_time} seconds **")
        wandb.log(metrics, step=int(step))
        pprint(metrics)
        return None

    def train_eval_step(carry, epoch_idx):
        """Single epoch with evaluation step for scan."""
        training_state, global_step = carry

        # Shuffle training data
        rng = jax.random.fold_in(
            jax.random.PRNGKey(config.training.random_seed), epoch_idx
        )
        perm = jax.vmap(jax.random.permutation, in_axes=(0, None))(
            jax.random.split(rng, num_perms), n_train
        ).reshape(-1)
        shuffled_train_data = jax.tree.map(lambda x: x[perm], train_data)
        batches = jax.tree.map(
            lambda x: x[:eff_n_train].reshape((num_batches, batch_size, -1)),
            shuffled_train_data,
        )

        def shard_val_epoch(params, train_data, test_data):
            results = jax.vmap(val_epoch, in_axes=(0, None, None))(
                params, train_data, test_data
            )
            # results = jax.lax.pmean(results, axis_name="batch")
            results = jax.tree.map(lambda x: x.reshape(-1, 1), results)
            return results

        def eval_step(carry, eval_idx):
            """Single evaluation step within an epoch."""
            ts, gs = carry

            # Training
            ts, losses = train_epoch(ts, batches)
            train_metrics = {
                "train/loss": losses[-1],
                "train/avg_loss": jnp.mean(losses[-1]),
            }

            # Validation

            if mesh is not None:
                val_metrics = shard_map(
                    shard_val_epoch,
                    mesh=mesh,
                    in_specs=(P("picard"), P("batch"), P("batch")),
                    out_specs=P("picard", "batch"),
                    check_rep=False,
                )(ts.params, train_data, test_data)
                val_metrics = jax.tree.map(
                    lambda x: jnp.mean(x, axis=1), val_metrics
                )
            else:
                val_metrics = jax.vmap(val_epoch, in_axes=(0, None, None))(
                    ts.params, train_data, test_data
                )

            best_t = jnp.argmin(
                val_metrics[config.training.picard.restart_metric]
            )

            # Log all metrics
            all_metrics = {
                **{"train/" + k: v for k, v in train_metrics.items()},
                **{"val/first/" + k: v[0] for k, v in val_metrics.items()},
                **{"val/best/" + k: v[best_t] for k, v in val_metrics.items()},
                **{"val/last/" + k: v[-1] for k, v in val_metrics.items()},
            }

            if config.training.picard.restart_rule == "best":
                restart_t = best_t
            elif config.training.picard.restart_rule == "last":
                restart_t = -1
            elif config.training.picard.restart_rule == "first":
                restart_t = 0
            else:
                raise ValueError(
                    f"Unknown restart rule {config.training.picard.restart_rule} (best, last, first are supported)."
                )

            if config.training.picard.fill_rule == "best":
                fill_t = best_t
            elif config.training.picard.fill_rule == "last":
                fill_t = -1
            else:
                raise ValueError(
                    f"Unknown fill rule {config.training.picard.fill_rule} (best, last are supported)."
                )

            window_size = config.training.picard.window_size
            window_idx = jnp.arange(window_size)
            new_window_idx = jnp.where(
                window_idx >= window_size - restart_t,
                jnp.roll(window_idx, -restart_t, axis=0),
                fill_t,
            )
            ts = jax.tree.map(lambda x: x[new_window_idx], ts)
            new_gs = gs + config.training.batches_per_eval
            io_callback(log_metrics_callback, None, all_metrics, new_gs)

            return (ts, new_gs), all_metrics

        # Run all evaluations for this epoch
        (final_ts, final_gs), metrics = jax.lax.scan(
            eval_step,
            (training_state, global_step),
            jnp.arange(num_evals_per_epoch),
        )

        return (final_ts, final_gs), metrics

    # Run all epochs using scan
    initial_carry = (
        jax.tree.map(
            lambda x: jnp.repeat(
                x[None, ...], config.training.picard.window_size, axis=0
            ),
            training_state,
        ),
        0,
    )

    results = jax.lax.scan(
        train_eval_step, initial_carry, jnp.arange(config.training.n_epochs)
    )
    jax.block_until_ready(results)

    print("Training completed!")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
