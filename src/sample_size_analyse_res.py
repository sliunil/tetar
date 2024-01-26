import pandas as pd
import numpy as np



# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/sample_size/sample_size_2000-29000_50-2000.csv"
output_folder_path = "../results"



# -------------------------------
# --- CODE
# -------------------------------

# Load dataset
sample_size_df = pd.read_csv(input_file_path)