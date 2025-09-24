import jax
import jax.numpy as jnp
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


@dataclass
class TrainingState:
    params: dict
    opt_state: dict
    data_index: int = 0


def create_batch_optimizer(A, b, G, h, optimizer_config):
    """Create a batched optimization function."""

    def single_optimize(c_vector):
        """
        Returns the solution and optimal objective value for a single LP,
        MAXIMIZING c^T x
        """
        lp = create_lp(-c_vector, A, b, G, h, 0, 1)
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
    """Create SPO+ loss function with custom gradients."""

    def spo_fun(pred_cost, true_cost, true_sol, true_obj):
        sol, obj = batch_optimize(true_cost - 2 * pred_cost)
        loss = obj + 2 * jnp.sum(pred_cost * true_sol, axis=1) - true_obj
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

    def loss_fn(params, x_batch, c_batch, true_sols_batch, true_objs_batch):
        """Compute SPO+ loss for a batch."""
        pred_c = model.apply(params, x_batch)
        loss = spo_loss(pred_c, c_batch, true_sols_batch, true_objs_batch)
        return loss

    return loss_fn


def create_train_epoch(optimizer, loss_fn):
    """Create function to train on batch data using scan."""

    def train_step(
        params, opt_state, x_batch, c_batch, true_sols_batch, true_objs_batch
    ):
        """Single training step."""
        loss, grads = jax.value_and_grad(loss_fn)(
            params, x_batch, c_batch, true_sols_batch, true_objs_batch
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train_epoch(state: TrainingState, batch_data, length: int):
        """Train on batch data using scan."""

        def scan_fn(carry, _):
            training_state = carry
            data_idx = jnp.mod(state.data_index, batch_data[0].shape[0])
            x_batch, c_batch, true_sols_batch, true_objs_batch = jax.tree.map(
                lambda x: x[data_idx],
                batch_data,
            )

            # Training step
            params, opt_state, loss = train_step(
                training_state.params,
                training_state.opt_state,
                x_batch,
                c_batch,
                true_sols_batch,
                true_objs_batch,
            )

            new_state = TrainingState(
                params=params,
                opt_state=opt_state,
                data_index=training_state.data_index + 1,
            )

            return new_state, loss

        final_state, batch_losses = jax.lax.scan(scan_fn, state, length=length)
        metrics = {
            "train_loss": batch_losses[-1],
        }

        return final_state, metrics

    return train_epoch


def create_val_epoch(model, batch_optimize, loss_fn):
    """Create function to evaluate model on train and test datasets."""

    def val_step(params, x_data, c_data, true_sols_data, true_objs_data):
        """Evaluate model on entire dataset."""
        pred_c = model.apply(params, x_data)
        loss = loss_fn(pred_c, c_data, true_sols_data, true_objs_data)
        pred_sol, pred_obj = batch_optimize(pred_c)
        subopt = 1 - jnp.mean(jnp.sum(pred_sol * c_data, axis=1), axis=0) / jnp.sum(
            true_objs_data
        )
        return {
            "loss": loss,
            "subopt": subopt,
        }

    def val_epoch(
        params,
        x_train_all,
        c_train_all,
        true_sols_train_all,
        true_objs_train_all,
        x_test,
        c_test,
        true_sols_test,
        true_objs_test,
    ):
        """Evaluate model on full train and test datasets."""
        train_loss = val_step(
            params,
            x_train_all,
            c_train_all,
            true_sols_train_all,
            true_objs_train_all,
        )
        val_loss = val_step(
            params, x_test, c_test, true_sols_test, true_objs_test
        )

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
        params=params, opt_state=opt_state, data_index=0
    )

    # Precompute true solutions and objectives for training and validation data
    print("Computing true solutions for training data...")
    true_sols_train, true_objs_train = batch_optimize(c_train)
    print("Computing true solutions for validation data...")
    true_sols_test, true_objs_test = batch_optimize(c_test)
    loss_fn = create_loss_fn(model, spo_loss)

    train_epoch = create_train_epoch(optimizer, loss_fn)
    val_epoch = jax.jit(create_val_epoch(model, batch_optimize, spo_loss))

    # Training loop
    n_train = len(x_train)
    print(f"Starting training for {config.training.n_epochs} epochs...")

    batch_size = config.training.batch_size
    num_batches = n_train // batch_size
    batches_per_epoch = cfg.training.batches_per_epoch or num_batches
    num_evals_per_epoch = num_batches // cfg.training.batches_per_eval
    assert (
        n_train % batch_size == 0
    ), "For now, train_size must be divisible by batch_size."

    data = (
        x_train,
        c_train,
        true_sols_train,
        true_objs_train,
    )

    for epoch in range(config.training.n_epochs):
        # Shuffle training data
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), n_train)
        batches = jax.tree.map(
            lambda data: data[perm].reshape(
                (
                    num_batches,
                    batch_size,
                    -1,
                )
            ),
            data,
        )
        for batch_idx in range(num_evals_per_epoch):
            start_time = time.time()
            training_state, train_metrics = train_epoch(
                training_state, batches, cfg.training.batches_per_eval
            )
            epoch_time = time.time() - start_time
            train_metrics["train/epoch_time"] = epoch_time
            print(f"Training took {epoch_time:.2f}s")

            # Evaluate after training on this chunk
            start_time = time.time()
            val_metrics = val_epoch(
                training_state.params,
                x_train,
                c_train,
                true_sols_train,
                true_objs_train,
                x_test,
                c_test,
                true_sols_test,
                true_objs_test,
            )
            val_time = time.time() - start_time
            val_metrics["epoch_time"] = val_time
            print(f"Validation took {val_time:.2f}s")

            global_step = epoch * batches_per_epoch + batch_idx
            wandb.log(
                {"train/" + k: v for k, v in train_metrics.items()},
                step=global_step,
            )
            wandb.log(
                {"val/" + k: v for k, v in val_metrics.items()},
                step=global_step,
            )
            pprint(val_metrics)

    print("Training completed!")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
