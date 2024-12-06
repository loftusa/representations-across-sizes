#%%
from representations_across_sizes.utils import dirichlet_energy
from representations_across_sizes.gride import calculate_gride_id, get_sequences
import torch
import numpy as np
from nnsight import LanguageModel
from datasets import load_dataset
from pathlib import Path
import gc
np.random.seed(0)



def get_layer_ids(lm, prompts, remote=True):
    """Get intrinsic dimensionality for each layer's activations"""
    with lm.session(remote=remote) as session:
        n_layers = len(lm.model.layers)
        ids = torch.zeros(n_layers).save()

        # Process in batches to handle memory
        batch_size = 100
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            with lm.trace(scan=True, validate=True) as tracer:
                with tracer.invoke(batch_prompts) as invoker:
                    for layer_idx, layer in enumerate(lm.model.layers):
                        # Get last token activation for each sequence in batch
                        last_token_activation = layer.output[0][:,-1,:]
                        # Add to collection of points for this layer
                        if i == 0:
                            layer_points = last_token_activation
                        else:
                            layer_points = torch.cat([layer_points, last_token_activation])
                            
                    # Calculate ID using GRIDE (add batch dim of 1)
                    print(layer_points.shape)
                    ids[layer_idx] = calculate_gride_id(layer_points.unsqueeze(0))[0]

        ids = ids.save()
    return ids.value.to(float).cpu().numpy(force=True)

if __name__ == "__main__":
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
            lm = LanguageModel(model_path, device_map="auto")
            remote = True if model_name == "8b" else False
            
            # Calculate IDs
            ids = get_layer_ids(lm, sequences, remote=remote)
            
            # Save results
            save_file = save_loc / f"{model_name}_{dataset_name}_ids.safetensors"
            torch.save({"ids": torch.tensor(ids)}, save_file)
            
            del lm
            gc.collect()
            torch.cuda.empty_cache()
