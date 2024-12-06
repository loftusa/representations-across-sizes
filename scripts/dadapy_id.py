from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pickle
import numpy as np
from dadapy import data
import argparse
import json

PICKLE_FORMAT = '~/id/inf_imb/reps_pickles/hidden_{}_{}_{}_reps.pickle' # data, mode, model


parser = argparse.ArgumentParser(description='ID computation')

# Data selection
parser.add_argument('--model', type=str, default="llama", choices=['llama', 'olmo', 'opt', 'pythia', 'mistral'])
parser.add_argument('--dataset', type=str, default='wikitext', choices=['wikitext', 'bookcorpus', 'pile'])
parser.add_argument('--method', type=str, default='gride')
parser.add_argument('--mode', type=str, default='sane', choices=['sane', '5', 'shuffled', 'random', '128'])
parser.add_argument('--random_seed', type=int, default=32)
parser.add_argument('--step', type=int, default=None)
args = parser.parse_args()

np.random.seed(args.random_seed)

filepath = PICKLE_FORMAT.format(args.dataset, args.mode, args.model)

with open(filepath, 'rb') as f:
    reps = pickle.load(f) # dict {layer_idx: list of reps}

subset_idx = args.random_seed % 5

reps = {k: np.array(reps[k])[subset_idx * 10000: (subset_idx + 1) * 10000,:] for k in reps}

# initialise the Data class
if args.method == 'twonn':
    results = {'id': [None for _ in reps], 'r': [None for _ in reps], 'err': [None for _ in reps]}

    for layer, layer_reps in reps.items():
        _data = data.Data(layer_reps)
        _data.remove_identical_points()

        # estimate ID
        id_twoNN, _, r = _data.compute_id_2NN()
        print('Estimated twoNN with r=typical distance between point i and its neighbor')
        print(id_twoNN)
        print(r)
        results['id'][layer] = id_twoNN
        results['r'][layer] = r
elif args.method == 'gride':
    results = {layer: {'id': [],
                       'err': [],
                       'r': []
                       } for layer in reps}
    for layer, layer_reps in reps.items():
        _data = data.Data(layer_reps)
        _data.remove_identical_points()

        # estimate ID
        ids_scaling, ids_scaling_err, rs_scaling = _data.return_id_scaling_gride(range_max = 2**13)
        results[layer]['r'] = rs_scaling.tolist()
        results[layer]['err'] = ids_scaling_err.tolist()
        results[layer]['id'] = ids_scaling.tolist()

# Save dictionary as JSON
save_path = '~/id_transformers/id_comp/{}/hidden_{}_{}_{}_ids_rs{}{}.json'.format(
    args.method,
    args.dataset,
    args.mode,
    args.model,
    subset_idx,
    f"_step{args.step}" if args.step is not None else ""
)
with open(save_path, 'w') as json_file:
    json.dump(results, json_file)
