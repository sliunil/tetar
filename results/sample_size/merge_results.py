import os
import pandas as pd

# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_folder_path = "results_2/all_files"
output_file_path = "results_2/sample_size_2000-29000_50-2000.csv"

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
