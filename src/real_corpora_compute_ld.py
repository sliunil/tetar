import os
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
results_folder_path = "../results/"
# Parameters...
subsample_len = 5000
num_subsamples = 10
compute_all = False

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
                                   "zipf_slope", 
                                   "zipf_shift"])

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

        with open(f"{corpora_folder_path}/{subfolder_name}/{file_name}") \
            as file:
            content = file.read()
        
        # Compute results
        sample = tokenize(content.lower())
        counter = Counter(sample)
        if compute_all:
            m_entrpy_rdm, sd_entrpy_rdm = subsample_entropy(sample, subsample_len, 
                                                            num_subsamples)
            m_entrpy_mav, sd_entrpy_mav = \
                subsample_entropy(sample, subsample_len, num_subsamples, 
                                mode="window")
            exp_variety = get_expected_subsample_variety(counter, subsample_len)
            mtld = MTLD(sample)
        else:
            m_entrpy_rdm, sd_entrpy_rdm, m_entrpy_mav, sd_entrpy_mav, \
                exp_variety, mtld = 0, 0, 0, 0, 0, 0
        # Set the shift to 0
        _, _, zipf_param = counter_to_zipf_data(counter)
        
        # Store results 
        doc_result_df = \
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
                "exp_variety": [exp_variety],
                "MTLD": [mtld], 
                "zipf_intercept": [1], 
                "zipf_slope": [zipf_param[0]],
                "zipf_shift": [zipf_param[1]]})
        results_df = pd.concat([results_df, doc_result_df], ignore_index=True)
        
# Save dataset 
results_df.to_csv(f"{results_folder_path}/real_corpora_indices_"
                  f"{subsample_len}_{num_subsamples}.csv")

