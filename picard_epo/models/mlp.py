import flax.linen as nn


class Linear(nn.Module):
    """
    Linear model for predicting c from x.
    Accepts inputs B x n_items x dim
    and outputs B x n_items
    """

    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x.reshape(x.shape[0], -1, self.dim)).squeeze()
        return x


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
