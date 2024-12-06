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

