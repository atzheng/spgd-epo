import flax.linen as nn


class BatchMLP(nn.Module):
    """
    Batch version of MLP for predicting c from x.
    Accepts inputs B x (n_items x dim)
    and outputs B x n_items
    """

    hidden_sizes: list
    dim: int
    output_size: int = 1
    squeeze: bool = True

    @nn.compact
    def __call__(self, x):
        # x is of shape (B, n_items * dim)
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1, self.dim))  
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)  # Output layer predicts c values
        x = x.reshape(batch_size, -1, self.output_size)  # Reshape back to (B, n_items, output_size)
        if self.squeeze and self.output_size == 1:
            x = x.squeeze(-1)
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
