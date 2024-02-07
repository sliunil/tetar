import os
import pandas as pd

# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_folder_path = "all_files"
output_file_path = "real_corpora_indices_3_z.csv"

# -------------------------------
# --- CODE
# -------------------------------

files = os.listdir(input_folder_path)
for i, file in enumerate(files):
    if i == 0:
        results_df = pd.read_csv(f"{input_folder_path}/{file}")
    else:
        new_file_df = pd.read_csv(f"{input_folder_path}/{file}")
        results_df = pd.concat([results_df, new_file_df])
               
results_df.to_csv(output_file_path, index=None)
