import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from lexical_diversity import tokenize, counter_to_zipf_data, TextGenerator


# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/real_corpora/real_corpora_indices_5000_10.csv"
corpora_folder_path = "../data/real_corpora/cleaned"
output_folder_path = "../results/artificial_corpora"


# -------------------------------
# --- CODE
# -------------------------------

# Load dataset
ld_stat_df = pd.read_csv(input_file_path, index_col=0)

# Extract quantities of interest
slopes = ld_stat_df["zipf_slope"].to_numpy()
intercepts = ld_stat_df["zipf_intercept"].to_numpy()
shifts = ld_stat_df["zipf_shift"].to_numpy()

# Get document with min_slope 
min_slope_doc = ld_stat_df[
    ld_stat_df.zipf_slope==
    ld_stat_df["zipf_slope"].min()][['group', 'name']].values[0]
# Get document with max_slope
max_slope_doc = ld_stat_df[
    ld_stat_df.zipf_slope==
    ld_stat_df["zipf_slope"].max()][['group', 'name']].values[0]

# Open the files
with open(f"{corpora_folder_path}/{min_slope_doc[0]}/{min_slope_doc[1]}") \
    as file:
        min_slope_content = file.read()
with open(f"{corpora_folder_path}/{max_slope_doc[0]}/{max_slope_doc[1]}") \
    as file:
        max_slope_content = file.read()
        
# Compute zipf from them 
min_slope_sample = tokenize(min_slope_content.lower())
min_slope_counter = Counter(min_slope_sample)
min_slope_ranks, min_slope_frequencies, min_slope_lm_model, min_slope_shift = \
    counter_to_zipf_data(min_slope_counter)
max_slope_sample = tokenize(max_slope_content.lower())
max_slope_counter = Counter(max_slope_sample)
max_slope_ranks, max_slope_frequencies, max_slope_lm_model, max_slope_shift = \
    counter_to_zipf_data(max_slope_counter)
    



# Fit the generator
my_generator = TextGenerator()
my_generator.fit(slopes, intercepts, shifts)