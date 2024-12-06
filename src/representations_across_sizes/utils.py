from tqdm import tqdm
from typeguard import typechecked
from torchtyping import TensorType
import torch

from torch import Tensor
from jaxtyping import Float

from nnsight.models.LanguageModel import LanguageModelProxy
from nnsight import LanguageModel

from typing import List

# @typechecked
def dirichlet_energy(representations: Float[Tensor | LanguageModelProxy, "b n d"]) -> Float[Tensor | LanguageModelProxy, "b"]:  # noqa
    """Calculate the Dirichlet energy between all pairs of representations.

    The Dirichlet energy is defined as the sum of squared L2 distances between
    all pairs of connected representations. Here we assume all representations
    are connected to each other.

    Squared distances are computed using the identity ||x-y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>

    Args:
      representations: Tensor of shape (B, N, D) containing B batches of N D-dimensional representations

    Returns:
      Scalar tensor containing the total Dirichlet energy
    """
    # Compute pairwise squared distances
    norm_sq = torch.sum(representations**2, dim=-1, keepdim=True)  # (B, N, 1)
    pairwise_sq = norm_sq + norm_sq.transpose(-2, -1)  # (B, N, N)
    dot_products = torch.matmul(
        representations, representations.transpose(-2, -1)
    )  # (B, N, N)
    squared_distances = pairwise_sq - 2 * dot_products

    # Return the Dirichlet energy
    return torch.sum(squared_distances, dim=(1, 2)) / 2


def get_activation_cache(
        model: LanguageModel, 
        dataset: List[str],
        layer_idxs: List[int] = [12, 20],
        llm_batch_size: int = 32,
    ) -> dict[int, list[torch.Tensor]]:
    """
    Compute the activation cache for a specific entity across all samples.
    
    Args:
        model: The language model
        dataset: List of dataset samples, untokenized string contexts
        layer: Model layer to extract activations from
        llm_batch_size: Batch size for processing

    Dimension annotations:
    - B: Batch size
    - L: Sequence length
    - D: Hidden dimension
    """

    cache = {layer_idx: [] for layer_idx in layer_idxs} # could be done with defaultdict
    
    # Create progress bar
    for batch_idx in tqdm(range(0, len(dataset), llm_batch_size)):
        torch.cuda.empty_cache()
        batch_str = dataset[batch_idx:batch_idx + llm_batch_size]

        # Get activations
        tracer_kwargs = {'scan': False, 'validate': False}
        with torch.no_grad(), model.trace(batch_str, **tracer_kwargs):
            for layer in layer_idxs:
                resid_post_module = model.model.layers[layer]
                resid_post_BLD = resid_post_module.output[0] # residual stream is a weird tuple so we have to zero index it
                resid_post_BLD = resid_post_BLD.to('cpu').save() # indicate we wanna use this tensor outside tracing context
                cache[layer].append(resid_post_BLD)
            

    # for layer in layer_idxs:
    #     cache[layer] = torch.cat(cache[layer], dim=0)
            
    return cache


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert isinstance(size, torch.Size)
    return " × ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc

    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s:%s%s %s"
                        % (
                            type(obj).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.is_pinned else "",
                            pretty_size(obj.size()),
                        )
                    )
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s → %s:%s%s%s%s %s"
                        % (
                            type(obj).__name__,
                            type(obj.data).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.data.is_pinned else "",
                            " grad" if obj.requires_grad else "",
                            " volatile" if obj.volatile else "",
                            pretty_size(obj.data.size()),
                        )
                    )
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)