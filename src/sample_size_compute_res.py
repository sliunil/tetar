import os
from collections import Counter
from LTTL.Utils import get_expected_subsample_variety
from lexical_diversity import tokenize, sample_entropy, subsample_entropy, \
    MTLD,  draw_reduced_sample
import numpy as np
import multiprocessing as mp
from itertools import product
import pandas as pd

# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

# The corpora folder path
corpora_folder_path = "../data/real_corpora/cleaned"
# The results folder path
results_folder_path = "../results"
# File prefix 
results_file_prefix = "sample_size"
# Parameters
reduced_sample_sizes = np.linspace(2000, 29000, 10).astype(int)
num_reduce_sample = 10
subsample_lens = [50, 100, 500, 1000, 2000]
mtld_thresholds = [0.85, 0.80, 0.75, 0.70, 0.65]
num_subsamples = 10

# -------------------------------
# --- CODE
# -------------------------------

# The number of cpu available
n_cpu = mp.cpu_count()

# Create result file
results_file_path = \
    f"{results_folder_path}/{results_file_prefix}_" \
    f"{min(reduced_sample_sizes)}-{max(reduced_sample_sizes)}_" \
    f"{min(subsample_lens)}-{max(subsample_lens)}.csv"
with open(results_file_path, "w") as output_file:
    output_file.write("genre,file,reduced_sample_size,subsample_len,"
                      "mtld_thresholds,id_draw,sample_entropy,"
                      "subsample_entropy_rdm,subsample_entropy_mav,"
                      "exp_variety,mtld\n")
    
# Function for multiprocess computing
def compute_ld_indice(parameters):
    (reduced_sample, reduced_sample_size, id_draw), \
        (subsample_len, mtld_threshold, num_subsamples) = parameters
    smple_entropy = sample_entropy(reduced_sample)
    if subsample_len > len(reduced_sample):
        subsample_entropy_rdm = np.nan
        subsample_entropy_mav = np.nan
        exp_variety = np.nan
    else:
        subsample_entropy_rdm, _ = subsample_entropy(reduced_sample, 
                                                    subsample_len, 
                                                    num_subsamples)
        subsample_entropy_mav, _ = subsample_entropy(reduced_sample, 
                                                    subsample_len, 
                                                    num_subsamples,
                                                    mode="window")
        exp_variety = get_expected_subsample_variety(Counter(reduced_sample),
                                                    subsample_len)
    mtld = MTLD(reduced_sample, mtld_threshold)
    return smple_entropy, subsample_entropy_rdm, subsample_entropy_mav, \
        exp_variety, mtld
    

        
# List the subfolders
subfolder_names = os.listdir(corpora_folder_path)
for subfolder_name in subfolder_names:

    # Print status
    print("="*80)
    print(f"Folder: {subfolder_name}")

    # List the files
    file_names = os.listdir(f"{corpora_folder_path}/{subfolder_name}")
    for file_name in file_names:
        
        # Print status
        print(f"File: {file_name}")

        # Open file and tokenize it
        with open(f"{corpora_folder_path}/{subfolder_name}/{file_name}") \
            as input_file:
            content = input_file.read()
        sample = tokenize(content.lower())

        # Construct parameters list for multiprocessed results
        subsamples = [(draw_reduced_sample(sample, reduced_sample_size), 
                    reduced_sample_size, id_draw) 
                    for reduced_sample_size, id_draw in 
                    product(reduced_sample_sizes, range(num_reduce_sample))]
        method_parameters = [(subsample_lens[i], mtld_thresholds[i], 
                              num_subsamples)
                            for i in range(len(subsample_lens))]
        parameters_list = list(product(subsamples, method_parameters))

        # Compute the results
        with mp.Pool(n_cpu) as my_pool:
            pool_results = my_pool.map(compute_ld_indice, parameters_list)

        # Build a dataframe from results and write it
        meta_var = [(subfolder_name, file_name, p[0][1], p[1][0], p[1][1], 
                     p[0][2]) for p in parameters_list]
        results_df = pd.concat([pd.DataFrame(meta_var), 
                                pd.DataFrame(pool_results)], 
                               axis=1, ignore_index=True)

        results_df.to_csv(results_file_path, mode='a', index=False, 
                          header=False)