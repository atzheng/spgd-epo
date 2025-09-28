import jax
import jax.numpy as jnp
from .problem import Problem
import pyepo
from mpax import create_lp, r2HPDHG

from picard_epo.problems.utils import (
    create_problem_data_hash,
    load_cache,
    save_cache,
)


class KnapsackProblem(Problem):
    def __init__(self, x, c, weights, capacities, solver=None, data_hash=None):
        super().__init__(x=x, c=-c, data_hash=data_hash)
        self.A = jax.experimental.sparse.empty(weights.shape)
        self.b = jnp.zeros(weights.shape[0])
        self.G = -jnp.asarray(weights)
        self.h = -jnp.array(capacities)
        self.solver = solver or (
            r2HPDHG(verbose=False, eps_abs=1e-4, eps_rel=1e-4)
        )

    def single_optimize(self, c_vector):
        """
        Returns the solution and optimal objective value for a single LP,
        MAXIMIZING c^T x
        """
        lp = create_lp(c_vector, self.A, self.b, self.G, self.h, 0, 1)
        result = self.solver.optimize(lp)
        obj = jnp.dot(c_vector, result.primal_solution)
        return result.primal_solution, obj

    def batch_optimize(self, c_vectors):
        return jax.vmap(self.single_optimize)(c_vectors)

    @classmethod
    def from_pyepo(
        cls,
        m=16,
        n=1000,
        p=5,
        deg=6,
        dim=2,
        noise_width=0.5,
        caps_per_dim=20,
        test_size=100,
        random_state=246,
    ):
        """
        Generate a knapsack problem instance using pyepo's knapsack data generator.
        """
        # Generate data for 2D knapsack with caching
        problem_hash = create_problem_data_hash(
            {
                "n": n,
                "p": p,
                "m": m,
                "deg": deg,
                "dim": dim,
                "noise_width": noise_width,
                "caps_per_dim": caps_per_dim,
                "test_size": test_size,
                "random_state": random_state,
            }
        )
        print(f"Problem data hash: {problem_hash}")

        # Try to load cached problem data
        cached_problem = load_cache(problem_hash, "problem_data")
        if cached_problem is not None:
            weights, x, c = cached_problem
        else:
            print("Generating data...")
            weights, x, c = pyepo.data.knapsack.genData(
                n + test_size,
                p,
                m,
                deg=deg,
                dim=dim,
                noise_width=noise_width,
            )
            print("Data generation complete.")
            save_cache(
                problem_hash,
                {"weights": weights, "x": x, "c": c},
                "problem_data",
            )

        return cls(
            x=cached_problem["x"],
            c=cached_problem["c"],
            weights=cached_problem["weights"],
            capacities=caps_per_dim * jnp.ones(dim),
            data_hash=problem_hash,
        )
