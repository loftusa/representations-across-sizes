from representations_across_sizes.gride import calculate_gride_id, get_sequences
import torch
import numpy as np
from nnsight import LanguageModel
from pathlib import Path
import gc
np.random.seed(0)

from representations_across_sizes.utils import get_activation_cache
from torch import Tensor
import dotenv
import os
from datetime import datetime
import json
dotenv.load_dotenv()

RESULTS_DIR = Path(os.getenv("RESULTS_DIR"))

def get_id_results(dataset: list[str], model_path: str, remote: bool = False, save=False):
    lm = LanguageModel(model_path, device_map="auto")

    activations = get_activation_cache(
        lm,
        layer_idxs=list(range(len(lm.model.layers))),
        dataset=dataset,
        llm_batch_size=64,
        remote=remote,
    )
    results = {}
    for layer, acts in activations.items():
        acts: list[Tensor] = [act[:, -1, :] for act in acts]
        activations[layer] = torch.cat(acts, dim=0)

    for layer, acts in activations.items():
        ids_scaling, ids_scaling_err, rs_scaling = calculate_gride_id(acts.to("cpu"))
        results[layer] = ids_scaling.mean()

    # Save results if path provided
    save_path = RESULTS_DIR / f"{model_path}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save:
        # Save results with model name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_to_save = {
        "model": model_path,
        "timestamp": timestamp,
            "intrinsic_dimensions": results,
        }

        with open(save_path, "w") as f:
            json.dump(results_to_save, f, indent=2)
    return results

def shuffle_sequence(sequence: list[str]) -> list[str]:
    """for every string in the sequence, shuffle all the words"""
    return [
        " ".join(np.random.permutation(string.split())) for string in sequence
    ]

if __name__ == "__main__":
    DEBUG = False
    # Following paper's setup
    datasets = ["bookcorpus", "pile", "wikitext"]
    models = {
        "1b": "meta-llama/Llama-3.2-1B",
        "3b": "meta-llama/Llama-3.2-3B", 
        "8b": "meta-llama/Meta-Llama-3.1-8B",
    }

    save_loc = Path("results")
    save_loc.mkdir(exist_ok=True)

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        # Get sequences following paper's methodology
        sequences = get_sequences(dataset_name)
        
        for model_name, model_path in models.items():
            print(f"Getting IDs for {model_name}")
            
            # Calculate IDs
            for shuffled in [True, False]:
                for i, sequence in enumerate(sequences):
                    save_path = RESULTS_DIR / f"{model_path}_{dataset_name}_{i}{'_shuffled' if shuffled else ''}.json"
                    if save_path.exists():
                        print(f"Skipping {save_path} because it already exists")
                        continue
                    if DEBUG:
                        sequence = sequence[:10]
                    if shuffled:
                        print("Shuffling sequence")
                        sequence = shuffle_sequence(sequence)
                    print(sequence[0])
                    ids = get_id_results(sequence, model_path, save=False)
                    # Save results with model name and timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_to_save = {
                    "model": model_path,
                    "timestamp": timestamp,
                        "intrinsic_dimensions": ids,
                    }

                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, "w") as f:
                        json.dump(results_to_save, f, indent=2)
                    gc.collect()
                    torch.cuda.empty_cache()

     