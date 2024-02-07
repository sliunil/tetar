import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
from LTTL.Utils import get_expected_subsample_variety
from lexical_diversity import generate_samples, sample_entropy, \
    subsample_entropy, MTLD
import multiprocessing as mp
from itertools import product



# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

# The file for the model
input_file_path = "../results/real_corpora/real_corpora_indices_4_zm.csv"
# The corpora folder path
corpora_folder_path = "../data/real_corpora/cleaned"
# The results folder path
results_folder_path = "../results/zm_res_1/all_files"
# File prefix 
results_file_prefix = "zm_res_1"
# Number of cuts
num_cut = 30
# Number of samples
num_samples = 10
# The length of subsample / mtld thresholds
subsample_len = 5000
mtld_threshold = 0.72


# -------------------------------
# --- CODE
# -------------------------------

# The number of cpu available
n_cpu = mp.cpu_count()

# Load dataset
ld_stat_df = pd.read_csv(input_file_path, index_col=0)

# Extract quantities of interest
slopes = ld_stat_df["zipf_slope"].to_numpy()
shifts = ld_stat_df["zipf_shift"].to_numpy()
length = ld_stat_df["length"].to_numpy()
variety = ld_stat_df["variety"].to_numpy()

# Compute values to test
mean_slope = np.mean(slopes)
mean_shift = np.mean(shifts)
mean_length = int(np.mean(length))
mean_variety = int(np.mean(variety))
mean_ranks = np.arange(1, mean_variety+1)
    
# The slope space
tested_slopes = np.linspace(np.min(slopes), np.max(slopes), num_cut)
tested_shifts = np.linspace(np.min(shifts), np.max(shifts), num_cut)
tested_varieties = np.linspace(np.min(variety), np.max(variety), num_cut)

# All tested parameters
tested_param = [('slope', sl, mean_shift, mean_variety) 
                for sl in tested_slopes] + \
               [('shift', mean_slope, sh, mean_variety) 
                for sh in tested_shifts] + \
               [('variety', mean_slope, mean_shift, va) 
                for va in tested_varieties] 

# Get the job number
job_id = int(sys.argv[1])

# Select the job to complete 
param = tested_param[job_id]

# Sample generation
samples = generate_samples(param[1], param[2], param[3], 
                           mean_length, num_samples)

# Function for multiprocess computing
def compute_ld_indice(sample):
    smple_entropy = sample_entropy(sample)
    sample_counter = Counter(sample)
    if subsample_len > len(sample):
        subsample_entropy_rdm = np.nan
        subsample_entropy_mav = np.nan
        exp_variety = np.nan
    else:
        subsample_entropy_rdm, _ = subsample_entropy(sample, 
                                                     subsample_len)
        subsample_entropy_mav, _ = subsample_entropy(sample, 
                                                     subsample_len,
                                                     mode="window")
        exp_variety = get_expected_subsample_variety(sample_counter,
                                                     subsample_len)
        mtld = MTLD(sample, mtld_threshold)
    return smple_entropy, subsample_entropy_rdm, subsample_entropy_mav, \
        exp_variety, mtld

# Safeguard for mac version
# if __name__ == '__main__':

# Compute the results
with mp.Pool(n_cpu) as my_pool:
    pool_results = my_pool.map(compute_ld_indice, samples)

# Build a dataframe from results and write it
meta_var = [(*param, id_sample) for id_sample in range(len(samples))]
results_df = pd.concat([pd.DataFrame(meta_var), 
                        pd.DataFrame(pool_results)], 
                        axis=1, ignore_index=True)
results_df.columns = ["type", "slope", "shift", "variety", "id_draw", 
                      "sample_entropy", "subsample_entropy_rdm", 
                      "subsample_entropy_mav", "exp_variety", "mtld"]
results_file_path = f"{results_folder_path}/{results_file_prefix}_" \
                    f"{param[1]}_{param[2]}_{param[3]}-" \
                    f"{subsample_len}.csv"

# Create folder if it do not exist
if not os.path.exists(results_folder_path):   
    os.makedirs(results_folder_path) 
results_df.to_csv(results_file_path, index=False)

