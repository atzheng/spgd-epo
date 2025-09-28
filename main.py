import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from mpax import create_lp, r2HPDHG
from sklearn.model_selection import train_test_split
import pyepo
import flax.linen as nn
import optax
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
from flax.struct import dataclass
import wandb
import time
from pprint import pprint
from jax_tqdm import scan_tqdm
from jax.experimental import io_callback
import hashlib
import pickle
import os
import tempfile
from typing import Tuple
import numpy as np


@dataclass
class TrainingState:
    params: dict
    opt_state: dict
    data_index: int = 0


@dataclass
class BatchData:
    x: jnp.ndarray
    c: jnp.ndarray
    true_sols: jnp.ndarray
    true_objs: jnp.ndarray


def create_data_hash(data_arrays):
    """Create a hash from multiple data arrays for cache key generation."""
    hash_obj = hashlib.sha256()
    for arr in data_arrays:
        # Convert JAX array to numpy for consistent hashing
        np_arr = jnp.asarray(arr)
        hash_obj.update(np_arr.tobytes())
    return hash_obj.hexdigest()


def get_cache_path(data_hash, cache_type="optimal_solutions"):
    """Get cache file path in temp directory."""
    temp_dir = tempfile.gettempdir()
    cache_filename = f"{cache_type}_{data_hash}.pkl"
    return os.path.join(temp_dir, cache_filename)


def load_cached_solutions(data_hash):
    """Load cached optimal solutions if they exist."""
    cache_path = get_cache_path(data_hash)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                print(f"Loaded cached solutions from {cache_path}")
                return (
                    cached_data["train_sols"],
                    cached_data["train_objs"],
                    cached_data["test_sols"],
                    cached_data["test_objs"],
                )
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    return None


def save_cached_solutions(
    data_hash, train_sols, train_objs, test_sols, test_objs
):
    """Save optimal solutions to cache."""
    cache_path = get_cache_path(data_hash)
    try:
        cached_data = {
            "train_sols": train_sols,
            "train_objs": train_objs,
            "test_sols": test_sols,
            "test_objs": test_objs,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cached_data, f)
        print(f"Saved solutions to cache: {cache_path}")
    except Exception as e:
        print(f"Error saving cache: {e}")


def create_batch_optimizer(A, b, G, h, optimizer_config):
    """Create a batched optimization function."""

    def single_optimize(c_vector):
        """
        Returns the solution and optimal objective value for a single LP,
        MAXIMIZING c^T x
        """
        lp = create_lp(c_vector, A, b, G, h, 0, 1)
        solver = r2HPDHG(
            eps_abs=optimizer_config.solver_eps_abs,
            eps_rel=optimizer_config.solver_eps_rel,
            verbose=optimizer_config.solver_verbose,
        )
        result = solver.optimize(lp)
        obj = jnp.dot(c_vector, result.primal_solution)
        return result.primal_solution, obj

    return jax.vmap(single_optimize)


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
        return grads

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
        grads = jax.vmap(picard_mapper, in_axes=(0, 0))(
            training_state, batch_data
        )
        return grads

    def picard_iteration(_, state_and_data: Tuple[TrainingState, BatchData]):
        training_state, batch_data = state_and_data
        grads = shard_map(
            shard_picard_mapper,
            mesh=mesh,
            in_specs=(P("picard"), P("picard", "batch")),
            out_specs=P("picard", "batch"),
            check_rep=False,
        )(training_state, jax.tree.map(lambda x: x[:window_size], batch_data))
        grads = jax.tree.map(lambda x: jnp.mean(x, axis=1), grads)
        init_state = jax.tree.map(
            lambda x: x[0],
            training_state,
        )
        new_training_state = jax.lax.scan(picard_reducer, init_state, grads)[1]
        new_batch_data = jax.tree.map(
            lambda x: jnp.roll(x, 1, axis=0), batch_data
        )
        return new_training_state, new_batch_data

    def train_epoch(training_state: TrainingState, batch_data: BatchData):
        """Train on batch data using scan."""
        final_state, final_batch = jax.lax.fori_loop(
            0, length, picard_iteration, (training_state, batch_data)
        )
        return final_state

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


class MLP(nn.Module):
    """Multi-layer perceptron for predicting c from x."""

    hidden_sizes: list
    output_size: int

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)  # Output layer predicts c values
        return x


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert DictConfig to our dataclass
    print(omegaconf.OmegaConf.to_yaml(cfg))
    config = cfg

    # Initialize wandb
    wandb.init(
        project="picard-epo", config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Generate data for 2D knapsack
    weights, x, c = pyepo.data.knapsack.genData(
        config.data.n + config.data.test_size,
        config.data.p,
        config.data.m,
        deg=config.data.deg,
        dim=config.data.dim,
        noise_width=config.data.noise_width,
    )

    c = -c  # pyepo generates costs for minimization; we want maximization

    # Data split
    x_train, x_test, c_train, c_test = list(
        map(
            jnp.asarray,
            train_test_split(
                x,
                c,
                test_size=config.data.test_size,
                random_state=config.data.random_state,
            ),
        )
    )

    # Setup optimization constraints
    caps = [config.data.caps_per_dim] * config.data.dim
    A = jax.experimental.sparse.empty((config.data.dim, config.data.m))
    b = jnp.zeros((config.data.dim,))
    G = -jnp.asarray(weights)
    h = -jnp.array(caps)

    batch_optimize = create_batch_optimizer(A, b, G, h, config.optimizer)

    spo_loss = create_spo_loss(batch_optimize)

    # Initialize model and optimizer
    model = MLP(
        hidden_sizes=config.model.hidden_sizes, output_size=config.data.m
    )
    key = jax.random.PRNGKey(config.training.random_seed)
    dummy_x = jnp.ones((1, config.data.p))  # dummy input for initialization
    params = model.init(key, dummy_x)

    optimizer = optax.adam(learning_rate=config.training.learning_rate)
    opt_state = optimizer.init(params)

    # Initialize training state
    training_state = TrainingState(
        params=params, opt_state=opt_state, data_index=jnp.zeros(1)
    )

    # Create mesh from config
    devices = jax.devices()
    requested_devices = cfg.training.mesh["batch"] * cfg.training.mesh["picard"]
    devices = np.array(jax.devices()[:requested_devices])
    mesh = Mesh(
        devices.reshape(
            (cfg.training.mesh["picard"], cfg.training.mesh["batch"])
        ),
        axis_names=("picard", "batch"),
    )

    # Precompute true solutions and objectives for training and validation data with caching
    data_hash = create_data_hash([c_train, c_test, G, h])
    print(f"Data hash: {data_hash}")

    # Try to load cached solutions
    cached_solutions = load_cached_solutions(data_hash)
    if cached_solutions is not None:
        (
            true_sols_train,
            true_objs_train,
            true_sols_test,
            true_objs_test,
        ) = cached_solutions
    else:
        print("Computing true solutions for training data...")
        true_sols_train, true_objs_train = batch_optimize(c_train)
        print("Computing true solutions for validation data...")
        true_sols_test, true_objs_test = batch_optimize(c_test)

        # Save to cache
        save_cached_solutions(
            data_hash,
            true_sols_train,
            true_objs_train,
            true_sols_test,
            true_objs_test,
        )
    loss_fn = create_loss_fn(model, spo_loss)

    train_epoch = create_train_epoch(
        optimizer,
        loss_fn,
        config.training.batches_per_eval,
        config.training.picard.window_size,
        mesh,
    )
    val_epoch = create_val_epoch(model, batch_optimize, spo_loss)

    # Training loop
    n_train = len(x_train)
    print(f"Starting training for {config.training.n_epochs} epochs...")

    batch_size = config.training.batch_size
    num_batches = n_train // batch_size
    eff_n_train = num_batches * batch_size
    batches_per_epoch = cfg.training.batches_per_epoch or num_batches
    num_evals_per_epoch = num_batches // cfg.training.batches_per_eval

    assert (
        num_batches >= cfg.training.picard.window_size
    ), "Number of batches per eval must be at least as large as Picard window size."

    train_batch_data = BatchData(
        x=x_train,
        c=c_train,
        true_sols=true_sols_train,
        true_objs=true_objs_train,
    )

    test_batch_data = BatchData(
        x=x_test,
        c=c_test,
        true_sols=true_sols_test,
        true_objs=true_objs_test,
    )

    start_time = time.time()

    # Create logging callbacks
    def log_metrics_callback(metrics, step):
        print(f"** {time.time() - start_time} seconds **")
        """Host callback for logging metrics to wandb."""
        wandb.log(metrics, step=int(step))
        pprint(metrics)
        return None

    def train_eval_step(carry, epoch_idx):
        """Single epoch with evaluation step for scan."""
        training_state, global_step = carry

        # Shuffle training data
        perm = jax.random.permutation(jax.random.PRNGKey(epoch_idx), n_train)
        shuffled_train_data = BatchData(
            x=train_batch_data.x[perm],
            c=train_batch_data.c[perm],
            true_sols=train_batch_data.true_sols[perm],
            true_objs=train_batch_data.true_objs[perm],
        )

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
            ts = train_epoch(ts, batches)
            train_metrics = {}  # FIXME

            # Validation

            val_metrics = shard_map(
                shard_val_epoch,
                mesh=mesh,
                in_specs=(P("picard"), P("batch"), P("batch")),
                out_specs=P("picard", "batch"),
                check_rep=False,
            )(ts.params, train_batch_data, test_batch_data)
            val_metrics = jax.tree.map(
                lambda x: jnp.mean(x, axis=1), val_metrics
            )

            best_t = jnp.argmin(val_metrics["val/subopt"])

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
            new_gs = gs + cfg.training.batches_per_eval
            io_callback(log_metrics_callback, None, all_metrics, new_gs)

            return (ts, new_gs), None

        # Run all evaluations for this epoch
        (final_ts, final_gs), _ = jax.lax.scan(
            eval_step,
            (training_state, global_step),
            jnp.arange(num_evals_per_epoch),
        )

        return (final_ts, final_gs), None

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
    (final_training_state, final_global_step), _ = jax.lax.scan(
        train_eval_step, initial_carry, jnp.arange(config.training.n_epochs)
    )
    jax.block_until_ready(final_training_state)
    print("Training completed!")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
