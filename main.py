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
from dataclasses import dataclass, field
import wandb


@dataclass
class DataConfig:
    m: int = 16  # number of items
    n: int = 100  # number of data
    p: int = 5  # size of feature
    deg: int = 6  # polynomial degree
    dim: int = 2  # dimension of knapsack
    noise_width: float = 0.5  # noise half-width
    caps_per_dim: int = 20  # capacity per dimension
    test_size: int = 1000
    random_state: int = 246


@dataclass
class ModelConfig:
    hidden_sizes: list = None  # [32, 128, 512, 2048]

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [32, 128, 512, 2048]


@dataclass
class OptimizerConfig:
    solver_eps_abs: float = 1e-4
    solver_eps_rel: float = 1e-4
    solver_verbose: bool = False


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 100
    n_epochs: int = 10
    random_seed: int = 42
    eval_every_n_batches: int = 10  # Evaluate every N minibatches


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


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
    x_train, x_test, c_train, c_test = train_test_split(
        x,
        c,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )

    # Setup optimization constraints
    caps = [config.data.caps_per_dim] * config.data.dim
    A = jax.experimental.sparse.empty((config.data.dim, config.data.m))
    b = jnp.zeros((config.data.dim,))
    G = -weights
    h = -jnp.array(caps)

    def single_optimize(c_vector):
        lp = create_lp(c_vector, A, b, G, h, 0, 1)
        solver = r2HPDHG(
            eps_abs=config.optimizer.solver_eps_abs,
            eps_rel=config.optimizer.solver_eps_rel,
            verbose=config.optimizer.solver_verbose,
        )
        result = solver.optimize(lp)
        obj = jnp.dot(c_vector, result.primal_solution)
        return result.primal_solution, obj

    batch_optimize = jax.vmap(single_optimize)

    @jax.custom_vjp
    def spo_loss(pred_cost, true_cost, true_sol, true_obj):
        return spo_fun(pred_cost, true_cost, true_sol, true_obj)[0]

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

    spo_loss.defvjp(spo_fwd, spo_bwd)

    # Initialize model and optimizer
    model = MLP(
        hidden_sizes=config.model.hidden_sizes, output_size=config.data.m
    )
    key = jax.random.PRNGKey(config.training.random_seed)
    dummy_x = jnp.ones((1, config.data.p))  # dummy input for initialization
    params = model.init(key, dummy_x)

    optimizer = optax.adam(learning_rate=config.training.learning_rate)
    opt_state = optimizer.init(params)

    # Precompute true solutions and objectives for training and validation data
    print("Computing true solutions for training data...")
    true_sols_train, true_objs_train = batch_optimize(c_train)
    print("Computing true solutions for validation data...")
    true_sols_test, true_objs_test = batch_optimize(c_test)

    def loss_fn(params, x_batch, c_batch, true_sols_batch, true_objs_batch):
        """Compute SPO+ loss for a batch."""
        pred_c = model.apply(params, x_batch)
        loss = spo_loss(pred_c, c_batch, true_sols_batch, true_objs_batch)
        return loss

    def train_step(
        params, opt_state, x_batch, c_batch, true_sols_batch, true_objs_batch
    ):
        """Single training step."""
        loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, c_batch, true_sols_batch, true_objs_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def val_step(params, x_data, c_data, true_sols_data, true_objs_data):
        """Evaluate model on entire dataset."""
        pred_c = model.apply(params, x_data)
        loss, _ = spo_fwd(pred_c, c_data, true_sols_data, true_objs_data)
        return loss

    # Training loop
    n_train = len(x_train)
    total_batches = 0

    print(f"Starting training for {config.training.n_epochs} epochs...")
    print(f"Evaluating every {config.training.eval_every_n_batches} minibatches...")
    
    for epoch in range(config.training.n_epochs):
        # Shuffle training data
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), n_train)
        x_train_shuffled = x_train[perm]
        c_train_shuffled = c_train[perm]
        true_sols_shuffled = true_sols_train[perm]
        true_objs_shuffled = true_objs_train[perm]

        # Mini-batch training
        for i in range(0, n_train, config.training.batch_size):
            end_idx = min(i + config.training.batch_size, n_train)
            x_batch = x_train_shuffled[i:end_idx]
            c_batch = c_train_shuffled[i:end_idx]
            true_sols_batch = true_sols_shuffled[i:end_idx]
            true_objs_batch = true_objs_shuffled[i:end_idx]

            params, opt_state, loss = train_step(
                params,
                opt_state,
                x_batch,
                c_batch,
                true_sols_batch,
                true_objs_batch,
            )
            total_batches += 1

            # Evaluate every N batches
            if total_batches % config.training.eval_every_n_batches == 0:
                # Evaluate on full training and validation sets
                train_loss = val_step(
                    params, x_train, c_train, true_sols_train, true_objs_train
                )
                val_loss = val_step(
                    params, x_test, c_test, true_sols_test, true_objs_test
                )

                # Log metrics to wandb
                wandb.log(
                    {
                        "batch": total_batches,
                        "epoch": epoch + 1,
                        "batch_loss": float(loss),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                    }
                )

                print(
                    f"Batch {total_batches} (Epoch {epoch+1}/{config.training.n_epochs}), "
                    f"Batch Loss: {loss:.6f}, "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

    print("Training completed!")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
