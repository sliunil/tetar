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
reduced_sample_sizes = np.linspace(2000, 29000, 10).astype(int)
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
    output_file.write("genre,file,reduced_sample_size,subsample_len,mtld_thresholds,id_draw,"
                      "sample_entropy,subsample_entropy_rdm,subsample_entropy_mav,exp_variety,mtld\n")
        
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
        with open(f"{corpora_folder_path}/{subfolder_name}/{file_name}") as input_file:
            content = input_file.read()
        sample = tokenize(content.lower())

        # Open the file
        with open(results_file_path, "a") as input_file:
            
            # Loop on reduced sample sizes
            for reduced_sample_size in reduced_sample_sizes:
                    
                for id_draw in range(num_reduce_sample):
                    
                    # Draw reduced sample
                    reduced_sample = draw_reduced_sample(sample, reduced_sample_size)
                    
                    # Compute sample entropy
                    smple_entropy = sample_entropy(reduced_sample)
                    
                    for id_len, subsample_len in enumerate(subsample_lens):
                        
                        # Compute ld measures
                        subsample_entropy_rdm, _ = subsample_entropy(reduced_sample, 
                                                                    subsample_len, 
                                                                    num_subsamples)
                        subsample_entropy_mav, _ = subsample_entropy(reduced_sample, 
                                                                    subsample_len, 
                                                                    num_subsamples,
                                                                    mode="window")
                        exp_variety = get_expected_subsample_variety(Counter(reduced_sample), 
                                                                    subsample_len)
                        mtld = MTLD(reduced_sample, mtld_thresholds[id_len])
                        
                        # Write in file 
                        input_file.write(f"{subfolder_name},{file_name},{reduced_sample_size},{subsample_len},"
                                         f"{mtld_thresholds[id_len]},{id_draw},{smple_entropy},"
                                         f"{subsample_entropy_rdm},{subsample_entropy_mav},{exp_variety},{mtld}\n")
