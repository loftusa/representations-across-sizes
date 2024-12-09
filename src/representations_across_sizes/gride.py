from datasets import load_dataset
import torch
import numpy as np
from scipy.optimize import minimize_scalar
from jaxtyping import Float
from torch import Tensor

import numpy as np
from scipy.optimize import minimize_scalar

# correct implementation (easy)

from dadapy import data

from pathlib import Path
import json
import os
import dotenv
from typing import List
from nnsight import LanguageModel

from representations_across_sizes.utils import get_activation_cache

dotenv.load_dotenv()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR"))


def get_id_results(dataset: List[str], model_name: str = "meta-llama/Llama-3.2-1B"):
    lm = LanguageModel(model_name, device_map="auto")

    activations = get_activation_cache(
        lm,
        layer_idxs=list(range(len(lm.model.layers))),
        dataset=dataset,
        llm_batch_size=64,
    )
    results = {}
    for layer, acts in activations.items():
        acts: List[Tensor] = [act[:, -1, :] for act in acts]
        activations[layer] = torch.cat(acts, dim=0)
        ids_scaling, ids_scaling_err, rs_scaling = calculate_gride_id(acts.to("cpu"))
        results[layer] = ids_scaling.mean()

    # Save results if path provided
    save_path = RESULTS_DIR / f"{model_name}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results with model name and timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_to_save = {
        "model": model_name,
        "timestamp": timestamp,
        "intrinsic_dimensions": results,
    }

    with open(save_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    return results


def calculate_gride_id(activations: Float[Tensor, "n d"], range_max=2**13):
    """
    Calculate the intrinsic dimension using the GRIDE estimator from dadapy.

    Parameters
    ----------
    activations : Float[Tensor, "n d"]
        A 2D array of shape (n_samples, d) containing the representations
        of your data points.
    range_max : int
        Maximum scale parameter for the gride estimation. Defaults to 2**13
        as in the provided snippet.

    Returns
    -------
    ids_scaling : np.ndarray
        Array of ID estimates over various scales.
    ids_scaling_err : np.ndarray
        Array of ID estimate errors over various scales.
    rs_scaling : np.ndarray
        Array of scales at which IDs are estimated.

    Notes
    -----
    - Initializes a dadapy Data object with the activations.
    - Removes identical points.
    - Computes ID using the `return_id_scaling_gride` method.
    """

    # Convert torch tensor to numpy array with float64 precision
    if torch.is_tensor(activations):
        activations = activations.detach().cpu().numpy()
    activations = np.asarray(activations, dtype=np.float64)

    # Initialize dadapy Data object
    _data = data.Data(activations)
    _data.remove_identical_points()

    # Compute ID with the GRIDE estimator over various scales
    ids_scaling, ids_scaling_err, rs_scaling = _data.return_id_scaling_gride(
        range_max=range_max
    )

    return ids_scaling, ids_scaling_err, rs_scaling


# alex implementation
# outdated now because I just emailed them and got their code instead
# which I should have just done in the first place...


def gride_log_likelihood(d, n1, n2, mus_n1_n2):
    mu = np.asarray(mus_n1_n2, dtype=np.float64)
    if n2 < n1:
        return -np.inf
    if np.any(mu <= 1.0):
        return -np.inf

    k_ = n2 - n1
    mud = np.exp(d * np.log(mu)) - 1.0
    if np.any(mud <= 0):
        return -np.inf

    N = len(mu)
    term1 = N * np.log(d)
    term2 = (k_ - 1) * np.sum(np.log(mud))
    term3 = ((n2 - 1) * d + 1) * np.sum(np.log(mu))
    return term1 + term2 - term3


def gride_mle_point(mus_n1_n2, n1=1, n2=2, upper_D=None):
    if n2 < n1:
        raise ValueError("n2 should be greater than n1")
    if upper_D is None:
        raise ValueError("Please provide upper_D")

    def neg_log_lik(d):
        val = gride_log_likelihood(d, n1, n2, mus_n1_n2)
        return -val

    res = minimize_scalar(neg_log_lik, bounds=(1.01, float(upper_D)), method="bounded")
    return res.x


def rgera(nsim, n1, n2, d):
    # Placeholder sampler. For proper bootstrap CIs, implement correct sampling.
    samples = 1.0 + np.random.gamma(shape=max(d, 1.0), scale=1.0, size=nsim)
    return samples


def gride_bootstrap(mus_n1_n2, n1=1, n2=2, nsim=2000, upper_D=None):
    if n2 < n1:
        raise ValueError("n2 should be greater than n1")
    if upper_D is None:
        raise ValueError("Please provide upper_D")

    n = len(mus_n1_n2)
    mle_est = gride_mle_point(mus_n1_n2, n1, n2, upper_D)

    boot_mus_samples = np.column_stack([rgera(n, n1, n2, mle_est) for _ in range(nsim)])
    bootstrap_sample = np.empty(nsim, dtype=float)
    for i in range(nsim):
        x = boot_mus_samples[:, i]

        def neg_log_lik(d):
            val = gride_log_likelihood(d, n1, n2, x)
            return -val

        res = minimize_scalar(
            neg_log_lik, bounds=(1.01, float(upper_D)), method="bounded"
        )
        bootstrap_sample[i] = res.x

    return {"mle": mle_est, "boot_sample": bootstrap_sample}


def compute_mus(activations, n1=1, n2=2, epsilon=1e-12):
    if activations.is_cuda:
        activations = activations.cpu()
    dists = torch.cdist(activations, activations)
    n1_dists, _ = torch.topk(dists, k=n1 + 1, dim=1, largest=False)
    n2_dists, _ = torch.topk(dists, k=n2 + 1, dim=1, largest=False)
    r_n1 = n1_dists[:, n1]
    r_n2 = n2_dists[:, n2]

    # Add epsilon to avoid division by zero
    ratios = (r_n2 / (r_n1 + epsilon)).numpy()
    return ratios


def filter_outliers(ratios, lower_quantile=0.2, upper_quantile=0.8):
    low_th = np.quantile(ratios, lower_quantile)
    high_th = np.quantile(ratios, upper_quantile)
    # Keep only ratios within these quantiles
    filtered = ratios[(ratios >= low_th) & (ratios <= high_th)]
    return filtered


def calculate_intrinsic_dimension(
    activations, n1=1, n2=2, upper_D=None, nsim=2000, alpha=0.95, do_bootstrap=False
):
    n_points, d = activations.shape
    if upper_D is None:
        upper_D = d + 1

    ratios = compute_mus(activations, n1, n2)
    # Filter outliers
    ratios = filter_outliers(ratios)

    mle_id = gride_mle_point(ratios, n1=n1, n2=n2, upper_D=upper_D)
    result = {"mle": mle_id}

    if do_bootstrap:
        res = gride_bootstrap(ratios, n1=n1, n2=n2, nsim=nsim, upper_D=upper_D)
        boot_sample = res["boot_sample"]
        one_m_alpha = 1 - alpha
        lb, ub = np.quantile(boot_sample, [one_m_alpha / 2, 1 - one_m_alpha / 2])
        result["ci"] = (lb, ub)

    return result


def get_sequences(dataset_name, num_sequences=10000, seq_length=20, num_partitions=5) -> List[List[str]]:
    """Get sequences from dataset following paper's methodology"""
    if dataset_name == "bookcorpus":
        dataset = load_dataset("bookcorpus/bookcorpus", split="train")
    elif dataset_name == "pile":
        dataset = load_dataset("NeelNanda/pile-10k", split="train")
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    partitions = []
    for _ in range(num_partitions):
        sequences = []
        used_indices = set()
        while len(sequences) < num_sequences:
            # Get random document
            doc_idx = np.random.randint(len(dataset))
            if doc_idx in used_indices:
                continue
            used_indices.add(doc_idx)
            text = dataset[doc_idx]["text"]
            
            # Get non-overlapping spans of seq_length tokens
            words = text.split()
            for start_idx in range(0, len(words) - seq_length + 1, seq_length):
                sequence = " ".join(words[start_idx:start_idx + seq_length])
                sequences.append(sequence)
                if len(sequences) >= num_sequences:
                    break
        
        partitions.append(sequences)
    
    return partitions