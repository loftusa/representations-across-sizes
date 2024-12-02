# %%  setup
from representations_across_sizes.utils import dirichlet_energy
import torch
import numpy as np
import os
from nnsight import LanguageModel, CONFIG
from jaxtyping import Float
from torch import Tensor
import lovely_tensors as lt
from dotenv import load_dotenv
from pathlib import Path
# lt.monkey_patch()
import gc
np.random.seed(0)
load_dotenv()
API_KEY = os.getenv("NNSIGHT_API_KEY")
CONFIG.set_default_api_key(API_KEY)
print(API_KEY)

from safetensors.torch import save_file

# This experiment measures the dirichlet energy of representations as a function of the layer. We first reconstruct figure 4 from https://openreview.net/pdf?id=pXlmOmlHJZ




def get_energies(lm, remote=True):
    n_prompts = len(prompts)
    with lm.session(remote=remote) as session:
        n_layers = len(lm.model.layers)
        energies: Float[Tensor, "n"] = torch.zeros(n_layers).save()

        # Let invoke handle the batching internally
        # two batches of n_prompts//2
        with lm.trace() as tracer:
            with tracer.invoke(prompts[n_prompts // 2 :]) as invoker:
                for layer_idx, layer in enumerate(lm.model.layers):
                    energies[layer_idx] += dirichlet_energy(layer.output[0]).mean().item()

        with lm.trace() as tracer:
            with tracer.invoke(prompts[: n_prompts // 2]) as invoker:
                for layer_idx, layer in enumerate(lm.model.layers):
                    energies[layer_idx] += dirichlet_energy(layer.output[0]).mean().item()


        # Scale the final result
        energies /= len(prompts)
        energies = energies.save()
    return energies.value.to(float).numpy(force=True)

if __name__ == "__main__":
  # get 1b, 3b, and 8b models
  context_length = 16
  words = np.array(["apple", "bird", "car", "egg", "house", "milk", "plane", "opera", "box", "sand", "sun", "mango", "rock", "math", "code", "phone"])
  prompts = [" ".join(list(np.random.choice(words, size=context_length))) for _ in range(10)]
  models = {
      "1b": "meta-llama/Llama-3.2-1B",
      "3b": "meta-llama/Llama-3.2-3B",
      "8b": "meta-llama/Meta-Llama-3.1-8B",
  }

  save_loc = Path("results")
  save_loc.mkdir(exist_ok=True)
  for model_name, model_size in models.items():
    print(f"Getting energies for {model_name}")
    lm = LanguageModel(model_size, device_map="auto")
    remote = True if model_name == "8b" else False
    energies = get_energies(lm, remote=remote)
    print(energies)
    save_file({"energies": torch.tensor(energies)}, save_loc / f"{model_name}.safetensors")

