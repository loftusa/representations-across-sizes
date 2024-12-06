from datasets import load_dataset
import torch
import numpy as np
from scipy.optimize import minimize_scalar
from jaxtyping import Float
from torch import Tensor

def calculate_gride_id(activations: Float[Tensor, "n d"], n1: int = 1, n2: int = 2) -> float:
    """Calculate intrinsic dimensionality using GRIDE estimator
    
    The Generalized Ratios ID Estimator (GRIDE) estimates intrinsic dimension 
    by fitting a generalized Pareto distribution to ratios of k-nearest neighbor
    distances.
    
    Args:
        activations: Tensor of shape [n_points, hidden_dim] where:
                    - n_points is the number of samples
                    - hidden_dim is the dimensionality of each point
        n1: First nearest neighbor order (default 1) 
        n2: Second nearest neighbor order (default 2)
        
    Returns:
        float: Estimated intrinsic dimension
        
    References:
        Denti et al. 2022 (https://www.nature.com/articles/s41598-022-20991-1)
    """
    n_points, hidden_dim = activations.shape
    
    # Input validation
    if n2 <= n1:
        raise ValueError("n2 must be greater than n1")
    
    if n_points < 2:
        raise ValueError("Need at least 2 points to estimate ID")
        
    if n_points < n2+1:
        raise ValueError(f"Need at least {n2+1} points for n2={n2}")
    
    # 1. Calculate pairwise distances
    dists = torch.cdist(activations, activations)
    
    # 2. Get n1th and n2th nearest neighbor distances
    # Add 1 to k since we want to exclude self-distances
    n1_dists, _ = torch.topk(dists, k=n1+1, dim=1, largest=False)
    n2_dists, _ = torch.topk(dists, k=n2+1, dim=1, largest=False)
    
    # 3. Calculate ratios μ = r_{n2}/r_{n1}
    ratios = n2_dists[:,-1] / n1_dists[:,-1]
    
    # 4. Define log likelihood function
    def log_likelihood(d):
        """Log likelihood for generalized Pareto distribution"""
        d_n12 = n2 - n1
        
        if d_n12 == 1:
            # Special case when n2 = n1 + 1
            lognum = np.log(d) 
            logden = ((n2-1)*d + 1) * np.log(ratios)
            log_dens = lognum - logden
        else:
            # General case
            logB = sum(np.log(np.arange(1, d_n12))) + sum(np.log(np.arange(1, n1))) - \
                   sum(np.log(np.arange(1, n2)))
            lognum = np.log(d) + (d_n12-1)*np.log(ratios.pow(d) - 1)
            logden = ((n2-1)*d + 1)*np.log(ratios)
            log_dens = lognum - logden - logB
            
        return log_dens.sum()

    # 5. Maximize likelihood to estimate ID
    # Search in range (0, hidden_dim)
    result = minimize_scalar(
        lambda d: -log_likelihood(d),
        bounds=(0.01, hidden_dim),
        method='bounded'
    )
    
    return result.x

def get_sequences(dataset_name, num_sequences=10000, seq_length=20, num_partitions=5):
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