#!/usr/bin/python

# Usage example:
# information_imbalance.py --model1 ./source_dir/distances/corpus_model1 --model2 ./source_dir/distances/corpus_model2 --target_file ./target_dir/corpus_model1_model2_ii --k 1

import argparse
import numpy as np
from dadapy._utils.metric_comparisons import _return_imbalance
import glob
from joblib import delayed, Parallel
import pickle
rng = np.random.default_rng()


def func(path_to_ind1,path_to_ind2,n_layers2,rng,k):
    II_ab = []
    II_ba = []
    with open(path_to_ind1,'rb') as f:
        indices1 = pickle.load(f)
    for j in range(1,n_layers2):
        with open(f"{path_to_ind2}{j:d}_indices.pickle",'rb') as f:
            indices2 = pickle.load(f)
        assert len(indices1) == len(indices2), \
            "mismatch in number of samples, impossible to perform information imbalance"
        II_ab.append(_return_imbalance(
            indices1, indices2, rng, k=k))
        II_ba.append(_return_imbalance(
            indices2, indices1, rng, k=k))
    return [II_ab,II_ba]

def func1(indices1,path_to_ind2,rng,k):
    with open(path_to_ind2,'rb') as f:
        indices2 = pickle.load(f)
    assert len(indices1) == len(indices2), \
        "mismatch in number of samples, impossible to perform information imbalance"
    return [_return_imbalance(indices1, indices2, rng, k=k),
            _return_imbalance(indices2, indices1, rng, k=k)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model1", type=str)
    parser.add_argument("--model2", type=str)
    parser.add_argument("--target_file", type=str, default=None)
    parser.add_argument("--k", type=int, default=1)
    
    args = parser.parse_args()

    n_layers1 = len(glob.glob(f"{args.model1}*_indices.pickle"))
    n_layers2 = len(glob.glob(f"{args.model2}*_indices.pickle"))

    #II_ab = np.zeros((n_layers1-1,n_layers2-1))
    #II_ba = np.zeros((n_layers1-1,n_layers2-1))

    tmp = np.array(Parallel(n_jobs=-1)(delayed(func)
            (f"{args.model1}{i:d}_indices.pickle",args.model2,n_layers2,rng,args.k) for i in range(1,n_layers1)))
    
    II_ab = tmp[:,0]
    II_ba = tmp[:,1]

    # for i in range(1,n_layers1):
    #     print(f"computing II model 1, layer... {i:d}", end='\r')
    #     indices1 = np.load(f"{args.model1}{i:d}_indices.npy")
    #   #  tmp = np.array(Parallel(n_jobs=-1)(delayed(func1)
    #   #                           (indices1, f"{args.model2}{j:d}_indices.npy",rng,args.k) for j in range(1,n_layers2)))
    #   #  II_ab[i-1] = np.copy(tmp[:,0])
    #   #  II_ba[i-1] = np.copy(tmp[:,1])
        # for j in range(1,n_layers2):
        #     indices2 = np.load(f"{args.model2}{j:d}_indices.npy")
        #     assert len(indices1) == len(indices2), \
        #         "mismatch in number of samples, impossible to perform information imbalance"
        #     II_ab[i-1,j-1] = _return_imbalance(
        #         indices1, indices2, rng, k=args.k)
        #     II_ba[i-1,j-1] = _return_imbalance(
        #         indices2, indices1, rng, k=args.k)

    np.savetxt(f'{args.target_file}_ab.dat', II_ab, fmt="%.5f")
    np.savetxt(f'{args.target_file}_ba.dat', II_ba, fmt="%.5f")
