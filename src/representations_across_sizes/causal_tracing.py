from functools import partial
from typing import NamedTuple

import torch
from torchtyping import TensorType




STDEV = 0.094


class CausalTracingInput(NamedTuple):
    prompt: TensorType["seq"]
    """Prompt tokens"""

    subject_idxs: TensorType["seq"]
    """Subject tokens"""

    target_id: TensorType["seq"]
    """Target tokens"""


def causal_trace_llama(model, cfg: CausalTracingInput, n_iters: int = 5):
    """
    Causal tracing for Llama models. Updated from https://github.com/cadentj/fact_localization/tree/master/localization/tracing
    TODO: still untested
    """

    # Arange prompts for token-wise interventions
    n_toks = len(cfg.prompt)
    n_layers = len(model.model.layers)
    batch = cfg.prompt.repeat(n_toks, 1)

    # Declare envoys
    mlps = [layer.mlp for layer in model.model.layers]
    model_in = model.model.embed_tokens

    def _window(layer, n_layers, window_size):
        return max(0, layer - window_size), min(n_layers, layer + window_size + 1)

    window = partial(_window, n_layers=n_layers, window_size=4)

    # Create noise
    noise = torch.randn(1, len(cfg.subject_idxs), 2048) * STDEV

    # Initialize results
    results = torch.zeros((len(model.model.layers), n_toks), device=model.device)

    for _ in range(n_iters):
        with torch.no_grad():
            with model.trace(cfg.prompt):
                clean_states = [
                    mlps[layer_idx].down_proj.output.cpu().save() for layer_idx in range(n_layers)
                ]

            clean_states = torch.cat(clean_states, dim=0)

            with model.trace(cfg.prompt):
                model_in.output[:, cfg.subject_idxs] += noise

                corr_logits = model.lm_head.output.softmax(-1)[
                    :, -1, cfg.target_id
                ].save()

            for layer_idx in range(n_layers):
                s, e = window(layer_idx)
                with model.trace(batch):
                    model_in.output[:, cfg.subject_idxs] += noise

                    for token_idx in range(n_toks):
                        s, e = window(layer_idx)
                        for l in range(s, e):
                            mlps[l].down_proj.output[token_idx, token_idx, :] = clean_states[
                                l, token_idx, :
                            ]

                    restored_logits = model.lm_head.output.softmax(-1)[
                        :, -1, cfg.target_id
                    ]

                    diff = restored_logits - corr_logits

                    diff.save()

                results[layer_idx, :] += diff.value

    results = results.detach().cpu()

    return results

