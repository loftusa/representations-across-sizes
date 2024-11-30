# %%  setup
from representations_across_sizes.utils import dirichlet_energy
import torch
import numpy as np
import os
from nnsight import LanguageModel, CONFIG
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked
import lovely_tensors as lt
from dotenv import load_dotenv
import nnsight
from tqdm import tqdm
from beartype import beartype
lt.monkey_patch()

np.random.seed(0)
load_dotenv()
API_KEY = os.getenv("NNSIGHT_API_KEY")
CONFIG.set_default_api_key(API_KEY)
print(API_KEY)

# This experiment measures the dirichlet energy of representations as a function of the layer. We first reconstruct figure 4 from https://openreview.net/pdf?id=pXlmOmlHJZ

model = "meta-llama/Meta-Llama-3.1-8B"
lm = LanguageModel(model, device_map='auto')
context_length = 16
words = np.array(["apple", "bird", "car", "egg", "house", "milk", "plane", "opera", "box", "sand", "sun", "mango", "rock", "math", "code", "phone"])
prompts = [" ".join(list(np.random.choice(words, size=context_length))) for _ in range(1000)]


@jaxtyped(typechecker=beartype)
def dirichlet_energy(representations: Float[Tensor, "b n d"]) -> Float[Tensor, "b"]:
    """Calculate the Dirichlet energy between all pairs of representations.

    The Dirichlet energy is defined as the sum of squared L2 distances between
    all pairs of connected representations. Here we assume all representations
    are connected to each other.

    Args:
      representations: Tensor of shape (B, N, D) containing B batches of N D-dimensional representations

    Returns:
      Scalar tensor containing the total Dirichlet energy
    """
    outer_product_difference: Float[Tensor, "b n n d"] = (
        representations.unsqueeze(2) - representations.unsqueeze(1)
    )
    squared_distances: Float[Tensor, "b n n"] = torch.sum(
        outer_product_difference**2, dim=-1
    )
    return torch.sum(squared_distances, dim=(1, 2)) / 2
#%%

prompt = prompts[0]
with lm.trace() as tracer:
  all_layers = nnsight.list().save()
  for i, layer in tqdm(enumerate(lm.model.layers)):
    per_layer_prompt = []
    for prompt in prompts:
      with tracer.invoke(prompt) as invoker:
        per_layer_prompt.append(layer.output[0])
    all_layers.append(per_layer_prompt)

#%%
all_layers = torch.stack([torch.stack(l) for l in all_layers.value])
#%%


# energies = [dirichlet_energy(all_layers[i].squeeze()).mean().item() for i in range(all_layers.shape[0])]
# %%
import seaborn as sns
sns.lineplot(x=list(range(len(energies))), y=energies)

#%%