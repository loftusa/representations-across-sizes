import json
import os
from fastcore.all import Path
import dotenv
import torch
import numpy as np
from nnsight import LanguageModel
from typing import List, Dict, Optional, Any, Union
from collections import defaultdict
from tqdm import tqdm

from representations_across_sizes.gride import calculate_gride_id, get_sequences
from representations_across_sizes.utils import get_activation_cache


dotenv.load_dotenv()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR"))


def get_id_results(dataset: List[str], model_name: str = "meta-llama/Llama-3.2-1B", save_str: Optional[str] = None):
    save_path = RESULTS_DIR / f"{model_name}_{save_str}.json"
    if save_path.exists():
        return Path.read_json(save_path)

    lm = LanguageModel(model_name, device_map="auto")

    activations = get_activation_cache(
        lm,
        layer_idxs=list(range(len(lm.model.layers))),
        dataset=dataset,
        llm_batch_size=64,
    )
    for layer, acts in activations.items():
        acts: List[torch.Tensor] = [act[:, -1, :] for act in acts]
        activations[layer] = torch.cat(acts, dim=0)
    results = {}

    for layer, acts in tqdm(activations.items(), desc="Calculating intrinsic dimensions"):
        ids_scaling, *_ = calculate_gride_id(acts.to("cpu"))
        results[layer] = ids_scaling.mean()

    # Save results if path provided
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


if __name__ == "__main__":
    dataset_names = ["bookcorpus", "pile", "wikitext"]
    for dataset_name in dataset_names:
        partitions = get_sequences(dataset_name)
        for i, partition in tqdm(enumerate(partitions), desc=f"Processing {dataset_name} partitions"):
            get_id_results(partition, save_str=f"{dataset_name}_{i}")
            torch.cuda.empty_cache()
