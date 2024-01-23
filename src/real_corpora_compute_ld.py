import os
from collections import Counter
from LTTL.Utils import get_expected_subsample_variety
from lexical_diversity import tokenize, sample_entropy, subsample_entropy, MTLD,  counter_to_zipf_data
import pandas as pd

# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

# The corpora folder path
corpora_folder_path = "../data/real_corpora/cleaned"
# The results folder path
results_folder_path = "../results"
# Parameters...
subsample_len = 1000
num_subsamples = 20



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
                                   "zipf_slope"])

# List the subfolders
subfolder_names = os.listdir(corpora_folder_path)

for subfolder_name in subfolder_names:
    
    # Print status
    print("="*80)
    print(f"Folder: {subfolder_name}")

    # Get files
    file_names = os.listdir(f"{corpora_folder_path}/{subfolder_name}")

    for file_name in file_names:
        
        # Print status
        print(f"File: {file_name}")

        with open(f"{corpora_folder_path}/{subfolder_name}/{file_name}") as file:
            content = file.read()
        
        # Compute results
        sample = tokenize(content.lower())
        counter = Counter(sample)
        m_entrpy_rdm, sd_entrpy_rdm = subsample_entropy(sample, subsample_len, num_subsamples)
        m_entrpy_mav, sd_entrpy_mav = \
            subsample_entropy(sample, subsample_len, num_subsamples, mode="window")
        _, _, z_intercept, z_slope = counter_to_zipf_data(counter)
        
        # Store results 
        doc_result = {"name": file_name,
                      "group": subfolder_name,
                      "length": len(sample),
                      "variety": len(counter),
                      "sample_entropy": sample_entropy(sample),
                      "m_subsample_entropy_rdm": m_entrpy_rdm,
                      "sd_subsample_entropy_rdm": sd_entrpy_rdm,
                      "m_subsample_entropy_mav": m_entrpy_mav, 
                      "sd_subsample_entropy_mav": sd_entrpy_mav, 
                      "exp_variety": get_expected_subsample_variety(counter, subsample_len),
                      "MTLD": MTLD(sample), 
                      "zipf_intercept": z_intercept, 
                      "zipf_slope": z_slope}
        results_df = pd.concat([results_df, pd.Series(doc_result)], axis=0)
        
# Save dataset 
results_df.to_csv(f"{results_folder_path}/real_corpora_indices_{subsample_len}_{num_subsamples}.csv")