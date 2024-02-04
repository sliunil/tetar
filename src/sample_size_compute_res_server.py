import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
from LTTL.Utils import get_expected_subsample_variety
from lexical_diversity import tokenize, sample_entropy, subsample_entropy, \
    MTLD,  draw_reduced_sample
import multiprocessing as mp
from itertools import product

# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

# The corpora folder path
corpora_folder_path = "../data/real_corpora/cleaned"
# The results folder path
results_folder_path = "../results/results_small/all_files"
# File prefix 
results_file_prefix = "sample_size"
# Parameters
# reduced_sample_sizes = np.logspace(np.log10(2000), np.log10(29000), 
#                                    num=15).astype(int) + 1
# subsample_lens = [50, 100, 500, 2000, 5000]
reduced_sample_sizes = np.logspace(np.log10(50), np.log10(1000), 
                                   num=15).astype(int) + 1
subsample_lens = [10, 50, 100, 200, 500]
mtld_thresholds = [0.84, 0.78, 0.72, 0.66, 0.6]
num_reduce_sample = 10
num_subsamples = 10


# -------------------------------
# --- CODE
# -------------------------------

# The number of cpu available
n_cpu = mp.cpu_count()

# Get the job number
job_id = int(sys.argv[1])

# Construct lists of all files and corresponding folders
all_file_names = []
all_file_folder_names = []
subfolder_names = os.listdir(corpora_folder_path)
for subfolder_name in subfolder_names:
    file_names = os.listdir(f"{corpora_folder_path}/{subfolder_name}")
    all_file_names.extend(file_names)
    all_file_folder_names.extend([subfolder_name] * len(file_names))
    
# Select the file and folder 
file_name = all_file_names[job_id]
subfolder_name = all_file_folder_names[job_id]
    
# Function for multiprocess computing
def compute_ld_indice(parameters):
    (reduced_sample, reduced_sample_size, id_draw), \
        (subsample_len, mtld_threshold, num_subsamples) = parameters
    smple_entropy = sample_entropy(reduced_sample)
    reduced_sample_counter = Counter(reduced_sample)
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
        exp_variety = get_expected_subsample_variety(reduced_sample_counter,
                                                     subsample_len)
    if len(reduced_sample_counter) / len(reduced_sample) > mtld_threshold:
        mtld = np.nan
    else:
        mtld = MTLD(reduced_sample, mtld_threshold)
    return smple_entropy, subsample_entropy_rdm, subsample_entropy_mav, \
        exp_variety, mtld

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

# Build a dataframe from results and save it
meta_var = [(subfolder_name, file_name, p[0][1], p[1][0], p[1][1], 
                p[0][2]) for p in parameters_list]
results_df = pd.concat([pd.DataFrame(meta_var), 
                        pd.DataFrame(pool_results)], 
                        axis=1, ignore_index=True)
results_df.columns = ["genre", "file", "reduced_sample_size", "subsample_len",
                      "mtld_thresholds", "id_draw", "sample_entropy",
                      "subsample_entropy_rdm", "subsample_entropy_mav",
                      "exp_variety", "mtld"]
results_file_path = \
    f"{results_folder_path}/{results_file_prefix}_{file_name[:-4]}_" \
    f"{min(reduced_sample_sizes)}-{max(reduced_sample_sizes)}_" \
    f"{min(subsample_lens)}-{max(subsample_lens)}.csv"

if not os.path.exists(results_folder_path):   
    os.makedirs(results_folder_path) 
results_df.to_csv(results_file_path, index=False)