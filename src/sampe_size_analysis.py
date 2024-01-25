import os
from collections import Counter
from LTTL.Utils import get_expected_subsample_variety
from lexical_diversity import tokenize, sample_entropy, subsample_entropy, MTLD,  draw_reduced_sample
import numpy as np

# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

# The corpora folder path
corpora_folder_path = "../data/real_corpora/cleaned"
# The results folder path
results_folder_path = "../results"
# File prefix 
results_file_prefix = "sample_size_analysis"
# Parameters
reduced_sample_sizes = np.array(list(range(2, 31))) * 1000
num_reduce_sample = 10
subsample_lens = [50, 100, 500, 1000, 2000]
mtld_thresholds = [0.65, 0.70, 0.75, 0.80, 0.85]
num_subsamples = 10



# -------------------------------
# --- CODE
# -------------------------------

# Create result file
results_file_path = f"{results_folder_path}/{results_file_prefix}_" \
                    f"{min(reduced_sample_sizes)}-{max(reduced_sample_sizes)}_" \
                    f"{min(subsample_lens)}-{max(subsample_lens)}.csv"
with open(results_file_path, "w") as output_file:
    pass
        
# List the subfolders
subfolder_names = os.listdir(corpora_folder_path)[0]
for subfolder_name in subfolder_names:
    
    # Print status
    print("="*80)
    print(f"Folder: {subfolder_name}")

    # List the files
    file_names = os.listdir(f"{corpora_folder_path}/{subfolder_name}")[0]
    for file_name in file_names:
    
        # Print status
        print(f"File: {file_name}")

        # Open file and tokenize it
        with open(f"{corpora_folder_path}/{subfolder_name}/{file_name}") as input_file:
            content = input_file.read()
        sample = tokenize(content.lower())
    
        # Loop on reduced sample sizes
        for reduced_sample_size in reduced_sample_sizes:
            
            # To store results 
            sample_entropy_res, subsample_entropy_rdm_res, subsample_entropy_mav_res, \
                exp_variety_res, mtld_res = \
                np.zeros((num_reduce_sample, )), \
                np.zeros((num_reduce_sample, len(subsample_lens))), \
                np.zeros((num_reduce_sample, len(subsample_lens))), \
                np.zeros((num_reduce_sample, len(subsample_lens))), \
                np.zeros((num_reduce_sample, len(subsample_lens)))
                
            for i in range(num_reduce_sample):
                
                # Draw reduced sample
                reduced_sample = draw_reduced_sample(sample, reduced_sample_size)
                
                # Compute sample entropy
                sample_entropy_res[i] = sample_entropy(reduced_sample)
                
                for j, subsample_len in subsample_lens:
                    
                    # Compute ld measures
                    subsample_entropy_rdm_res[i, j], _ = subsample_entropy(reduced_sample, 
                                                                           subsample_len, 
                                                                           num_subsamples)
                    subsample_entropy_mav_res[i, j], _ = subsample_entropy(reduced_sample, 
                                                                           subsample_len, 
                                                                           num_subsamples,
                                                                           mode="window")
                    exp_variety_res[i, j] = get_expected_subsample_variety(Counter(reduced_sample), 
                                                                           subsample_len)
                    mtld_res[i, j] = MTLD(reduced_sample, mtld_thresholds[j])
