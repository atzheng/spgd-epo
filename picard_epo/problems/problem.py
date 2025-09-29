from flax.struct import dataclass
import jax.numpy as jnp
import jax

from .utils import load_cache, save_cache


@dataclass
class BatchData:
    x: jnp.ndarray
    c: jnp.ndarray
    true_sols: jnp.ndarray
    true_objs: jnp.ndarray

    def __len__(self):
        return self.c.shape[0]


class Problem(object):
    def __init__(self, x, c, data_hash=None, use_cached=True):
        self.data_hash = data_hash
        self.x = x
        self.c = c
        self.use_cached = use_cached

    def single_optimize(self, cost):
        raise NotImplementedError()

    def batch_optimize(self, c_vectors):
        return jax.vmap(self.single_optimize)(c_vectors)

    @property
    def dataset(self):
        return BatchData(
            x=jnp.array(self.x),
            c=jnp.array(self.c),
            true_sols=jnp.array(self.true_solutions[0]),
            true_objs=jnp.array(self.true_solutions[1]),
        )

    @property
    def true_solutions(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        cached_solutions = (
            load_cache(self.data_hash, "optimal_solutions")
            if self.use_cached
            else None
        )
        if cached_solutions is not None:
            return cached_solutions
        else:
            print("Computing true solutions for training data...")
            true_sols, true_objs = self.batch_optimize(self.c)

            # Save to cache
            save_cache(
                self.data_hash,
                (true_sols, true_objs),
                "optimal_solutions",
            )
            return true_sols, true_objs


    
