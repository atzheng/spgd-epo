import flax.linen as nn


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
