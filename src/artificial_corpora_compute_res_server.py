import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
from LTTL.Utils import get_expected_subsample_variety
from lexical_diversity import TextGenerator, sample_entropy, \
    subsample_entropy, MTLD
import multiprocessing as mp
from itertools import product

# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

# The file for the model
input_file_path = "../results/real_corpora/real_corpora_indices_200_10.csv"
# The corpora folder path
corpora_folder_path = "../data/real_corpora/cleaned"
# The results folder path
results_folder_path = "../results/ac_results_2/all_files"
# File prefix 
results_file_prefix = "sample_size"
# Tested slopes
tested_slopes = np.linspace(-0.9, -1.7, 20)
# Generated size
gen_size = 10000
# Number of samples
num_samples = 10
# The length of subsamples / mtld thresholds
subsample_lens = [50, 100, 500, 1000, 2000]
mtld_thresholds = [0.84, 0.78, 0.72, 0.66, 0.6]
# Number of subsamples
num_subsamples = 10

# -------------------------------
# --- CODE
# -------------------------------

# The number of cpu available
n_cpu = mp.cpu_count()

# Get the job number
job_id = int(sys.argv[1])

# Select the job to complete 
tested_slope = tested_slopes[job_id]

# Function for multiprocess computing
def compute_ld_indice(parameters):
    (sample, slope, id_draw), \
        (subsample_len, mtld_threshold, num_subsamples) = parameters
    smple_entropy = sample_entropy(sample)
    sample_counter = Counter(sample)
    if subsample_len > len(sample):
        subsample_entropy_rdm = np.nan
        subsample_entropy_mav = np.nan
        exp_variety = np.nan
    else:
        subsample_entropy_rdm, _ = subsample_entropy(sample, 
                                                     subsample_len, 
                                                     num_subsamples)
        subsample_entropy_mav, _ = subsample_entropy(sample, 
                                                     subsample_len, 
                                                     num_subsamples,
                                                     mode="window")
        exp_variety = get_expected_subsample_variety(sample_counter,
                                                     subsample_len)
    if len(sample_counter) / len(sample) > mtld_threshold:
        mtld = np.nan
    else:
        mtld = MTLD(sample, mtld_threshold)
    return smple_entropy, subsample_entropy_rdm, subsample_entropy_mav, \
        exp_variety, mtld

# Load dataset
ld_stat_df = pd.read_csv(input_file_path, index_col=0)
    
# Extract quantities of interest
slopes = ld_stat_df["zipf_slope"].to_numpy()
intercepts = ld_stat_df["zipf_intercept"].to_numpy()
shifts = ld_stat_df["zipf_shift"].to_numpy()
    
# Fit the generator
my_generator = TextGenerator()
my_generator.fit(slopes, intercepts, shifts)

# Generate the samples
slope_samples = my_generator.generate_samples(tested_slope, gen_size, 
                                            num_samples)
samples = [(list(sample), tested_slope, id_draw) 
           for id_draw, sample in enumerate(slope_samples)]
# Build the srest of the parameters
method_parameters = [(subsample_lens[i], mtld_thresholds[i], 
                        num_subsamples)
                    for i in range(len(subsample_lens))]
parameters_list = list(product(samples, method_parameters))

# Safeguard for mac version
# if __name__ == '__main__':

# Compute the results
with mp.Pool(n_cpu) as my_pool:
    pool_results = my_pool.map(compute_ld_indice, parameters_list)

# Build a dataframe from results and write it
meta_var = [(p[0][1], p[1][0], p[1][1], p[0][2]) for p in parameters_list]
results_df = pd.concat([pd.DataFrame(meta_var), 
                        pd.DataFrame(pool_results)], 
                        axis=1, ignore_index=True)
results_df.columns = ["slope", "subsample_len", "mtld_thresholds", 
                      "id_draw", "sample_entropy", "subsample_entropy_rdm", 
                      "subsample_entropy_mav", "exp_variety", "mtld"]
results_file_path = f"{results_folder_path}/{results_file_prefix}_" \
                    f"{tested_slope}_{gen_size}_{min(subsample_lens)}-" \
                    f"{max(subsample_lens)}.csv"

if not os.path.exists(results_folder_path):   
    os.makedirs(results_folder_path) 
results_df.to_csv(results_file_path, index=False)

