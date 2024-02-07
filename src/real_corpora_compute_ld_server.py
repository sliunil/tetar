import os
import sys
from collections import Counter
from LTTL.Utils import get_expected_subsample_variety
from lexical_diversity import tokenize, sample_entropy, subsample_entropy, \
    MTLD,  counter_to_zipf_data
import pandas as pd



# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

# The corpora folder path
corpora_folder_path = "../data/real_corpora/cleaned"
# The results folder path
results_folder_path = "../results/all_files"
# File prefix 
results_file_prefix = "ld_"
# Parameters...
subsample_len = 1000
num_subsamples = 10


# -------------------------------
# --- CODE
# -------------------------------

# To store results
results_df = pd.DataFrame(columns=["name",
                                   "group",
                                   "length",
                                   "variety",
                                   "sample_entropy",
                                   "m_subsample_entropy_rdm",
                                   "sd_subsample_entropy_rdm",
                                   "m_subsample_entropy_mav", 
                                   "sd_subsample_entropy_mav", 
                                   "exp_variety",
                                   "MTLD", 
                                   "zipf_intercept", 
                                   "zipf_slope", 
                                   "zipf_shift"])

# List the subfolders and the files
subfolder_names = os.listdir(corpora_folder_path)
file_subfolder_names = []
file_names = []
for subfolder_name in subfolder_names:
    names = os.listdir(f"{corpora_folder_path}/{subfolder_name}")
    file_names.extend(names)
    file_subfolder_names.extend([subfolder_name] * len(names))

# Get the job number
job_id = int(sys.argv[1])

# Get the folder and file name
subfolder_name = file_subfolder_names[job_id]
file_name = file_subfolder_names[job_id]

# Open the file 
with open(f"{corpora_folder_path}/{subfolder_name}/{file_name}") \
    as file:
    content = file.read()
    
# Compute results
sample = tokenize(content.lower())
counter = Counter(sample)
m_entrpy_rdm, sd_entrpy_rdm = subsample_entropy(sample, subsample_len, 
                                                num_subsamples)
m_entrpy_mav, sd_entrpy_mav = \
    subsample_entropy(sample, subsample_len, num_subsamples, 
                        mode="window")
# Set the shift to 0
_, _, zipf_param = counter_to_zipf_data(counter)
        
# Store results 
results_df = \
    pd.DataFrame.from_dict({
        "name": [file_name],
        "group": [subfolder_name],
        "length": [len(sample)],
        "variety": [len(counter)],
        "sample_entropy": [sample_entropy(sample)],
        "m_subsample_entropy_rdm": [m_entrpy_rdm],
        "sd_subsample_entropy_rdm": [sd_entrpy_rdm],
        "m_subsample_entropy_mav": [m_entrpy_mav], 
        "sd_subsample_entropy_mav": [sd_entrpy_mav], 
        "exp_variety": [get_expected_subsample_variety(counter, 
                                                        subsample_len)],
        "MTLD": [MTLD(sample)], 
        "zipf_intercept": [1], 
        "zipf_slope": [zipf_param[0]],
        "zipf_shift": [zipf_param[1]]})
        
# Save results 
results_df.to_csv(f"{results_folder_path}/{results_file_prefix}_" 
                  f"{subsample_len}_{file_names[0][:-4]}.csv")
