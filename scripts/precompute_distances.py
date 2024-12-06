# usage example:

# python compute_imbalance.py hidden_wikitext_opt_reps.pickle hidden_wikitext_opt

# output in this case will be a set of index files (one for each sub-corpus of 10k sequences and layer) each named:

# hidden_wikitext_opt_N_L_indices.pickle

# at the moment, we hardcode neighbourhood size (10000) and estimation size (also 10000)

import sys
import pickle
import numpy as np
import dadapy

out_prefix = sys.argv[2]

with open(sys.argv[1],'rb') as f:
    input_pickle=pickle.load(f)

layer_count = len(input_pickle)

segments_count = int(len(input_pickle[0]) / 10000)

for i in range(segments_count):
    # get representations of the 10k points between i*10000 and i+10000
    start = i*10000
    end = start + 10000
    # debug
    print("start " + str(start))
    print("end " + str(end))
    section = str(i+1)

    for layer in range(len(input_pickle)):
        dt = dadapy.Data(np.array(input_pickle[layer][start:end]))
        dt.compute_distances(maxk=9999)
        output_indices_pickle =  out_prefix + "_" + section + "_" + str(layer) + "_indices.pickle"
        
        with open(output_indices_pickle,'wb') as f:
            pickle.dump(dt.dist_indices,f)



    
