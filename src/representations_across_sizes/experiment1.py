#%%
import sys
import os

# I hate that I have to do this to get #%% working with relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nnsight import LanguageModel
import torch
from representations_across_sizes.causal_tracing import causal_trace_llama, CausalTracingInput

def get_llama_model(model="meta-llama/Llama-3.2-1B"):
    return LanguageModel(model)


if __name__ == "__main__":
  model = get_llama_model()
  print(model)
# %%
