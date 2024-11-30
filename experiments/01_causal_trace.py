# %%
# %load_ext autoreload
# %autoreload 

import sys
import os
import torch

from nnsight import LanguageModel
from representations_across_sizes.causal_tracing import (
    causal_trace_llama,
    CausalTracingInput,
)


def get_llama_model(model="meta-llama/Llama-3.2-1B"):
    return LanguageModel(model)


if __name__ == "__main__":
    model = get_llama_model()
    print(model)