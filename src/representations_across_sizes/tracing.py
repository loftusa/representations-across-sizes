from functools import partial
from typing import NamedTuple
from copy import deepcopy
from typing import List, Literal

from transformers import AutoTokenizer
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
                    mlps[layer_idx].down_proj.output.cpu().save()
                    for layer_idx in range(n_layers)
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
                            mlps[l].down_proj.output[token_idx, token_idx, :] = (
                                clean_states[l, token_idx, :]
                            )

                    restored_logits = model.lm_head.output.softmax(-1)[
                        :, -1, cfg.target_id
                    ]

                    diff = restored_logits - corr_logits

                    diff.save()

                results[layer_idx, :] += diff.value

    results = results.detach().cpu()

    return results

def format_template(
    tok: AutoTokenizer,
    context_templates: List[str],
    words: str,
    subtoken: Literal["last", "all"] = "last",
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "Multiple fill-ins not supported."

    # assert subtoken == "last", "Only last token retrieval supported."

    # Compute prefixes and suffixes of the tokenized context
    prefixes, suffixes = _split_templates(context_templates)
    _words = deepcopy(words)

    # Compute lengths of prefixes, words, and suffixes
    prefixes_len, words_len, suffixes_len = _get_split_lengths(
        tok, prefixes, _words, suffixes
    )

    # Format the prompts bc why not
    input_tok = tok(
        [template.format(word) for template, word in zip(context_templates, words)],
        return_tensors="pt",
        padding=True,
    )

    size = input_tok["input_ids"].size(1)
    padding_side = tok.padding_side

    if subtoken == "all":
        word_idxs = [
            [prefixes_len[i] + _word_len for _word_len in range(words_len[i])]
            for i in range(len(prefixes))
        ]

        return input_tok, word_idxs

    # Compute indices of last tokens
    elif padding_side == "right":
        word_idxs = [prefixes_len[i] + words_len[i] - 1 for i in range(len(prefixes))]

        return input_tok, word_idxs

    elif padding_side == "left":
        word_idxs = [size - suffixes_len[i] - 1 for i in range(len(prefixes))]

        return input_tok, word_idxs

def prepend_bos(input, bos_token):
    batch_size = input.size(0)
    bos_tensor = torch.full((batch_size, 1), bos_token, dtype=input.dtype)
    output = torch.cat((bos_tensor, input), dim=1)
    return output

def load_fact(tokenizer, req):
    raw_prompt = req["prompt"]
    subject = req["subject"]
    target = req["target_true"]["str"]

    print(f"RAW: |{raw_prompt}|")
    print(f"SUBJECT: |{subject}|")
    print(f"TARGET: |{target}|")

    input_tok, subject_idxs = format_template(
        tokenizer, [raw_prompt], [subject], subtoken="all"
    )

    prompt = prepend_bos(input_tok["input_ids"], tokenizer.bos_token_id)[0]
    target_token = tokenizer.encode(" " + target, return_tensors="pt")[0].item()

    return CausalTracingInput(
        prompt=prompt,
        subject_idxs=[i + 1 for i in subject_idxs[0]],  # Adj for prepended BOS
        target_id=target_token,
    )


def _get_split_lengths(tok, prefixes, words, suffixes):
    # Pre-process tokens to account for different 
    # tokenization strategies
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]

            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    batch_tok = tok([*prefixes, *words, *suffixes])

    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)
    ]
    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]

    return prefixes_len, words_len, suffixes_len


def _split_templates(context_templates):
    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes = [tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)]
    suffixes = [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]

    return prefixes, suffixes


def get_activation_cache(
        model: LanguageModel, 
        dataset: List[str],
        layer_idxs: int = [12, 20],
        llm_batch_size: int = 32,
    ) -> dict:
    """
    Compute the activation cache for a specific entity across all samples.
    
    Args:
        model: The language model
        dataset: List of dataset samples, untokenized string contexts
        layer_idxs: Model layers to extract activations from
        llm_batch_size: Batch size for processing

    Dimension annotations:
    - B: Batch size
    - L: Sequence length
    - D: Hidden dimension
    """

    cache = {layer_idx: [] for layer_idx in layer_idxs} # could be done with defaultdict
    
    # Create progress bar
    for batch_idx in range(0, len(dataset), llm_batch_size):
        torch.cuda.empty_cache()
        batch_str = dataset[batch_idx:batch_idx + llm_batch_size]

        # Get activations
        tracer_kwargs = {'scan': False, 'validate': False}
        with torch.no_grad(), model.trace(batch_str, **tracer_kwargs):
            for layer in layer_idxs:
                resid_post_module = model.model.layers[layer]
                resid_post_BLD = resid_post_module.output[0] # residual stream is a weird tuple so we have to zero index it
                resid_post_BLD.save() # indicate we wanna use this tensor outside tracing context
                cache[layer].append(resid_post_BLD)
            
    return cache