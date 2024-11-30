from typeguard import typechecked
from torchtyping import TensorType
import torch


@typechecked
def dirichlet_energy(representations: TensorType["b", "n", "d"], normalized=False) -> TensorType["b"]:
    """Calculate the Dirichlet energy between all pairs of representations.

    The Dirichlet energy is defined as the sum of squared L2 distances between
    all pairs of connected representations. Here we assume all representations
    are connected to each other.

    Args:
      representations: Tensor of shape (B, N, D) containing B batches of N D-dimensional representations

    Returns:
      Scalar tensor containing the total Dirichlet energy
    """
    outer_product_difference: TensorType["b", "n", "n", "d"] = representations.unsqueeze(
        2
    ) - representations.unsqueeze(1)
    squared_distances: TensorType["b", "n", "n"] = torch.sum(
        outer_product_difference**2, dim=-1
    )
    n_pairs = representations.shape[1] * (representations.shape[1] - 1) / 2 if normalized else 1
    return torch.sum(squared_distances, dim=(1, 2)) / (2 * n_pairs)
