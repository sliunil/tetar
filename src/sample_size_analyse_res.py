import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/sample_size/sample_size_2000-29000_50-2000.csv"
output_folder_path = "../results"
output_file_prefix = "sample_size_plot"


# -------------------------------
# --- CODE
# -------------------------------

# Load dataset
sample_size_df = pd.read_csv(input_file_path)

# Get measure names 
measure_names = sample_size_df.columns.to_numpy()[6:]

# Get genres (color)
genres = sample_size_df.genre.unique()
color_map = cm.get_cmap("hsv", len(genres) + 1)

# Get subsample lengths (alpha)
subsample_lens = sample_size_df.subsample_len.unique()
alphas = np.linspace(0.5, 1, len(subsample_lens))

# Get reduced sample size
reduced_sample_sizes = sample_size_df.reduced_sample_size.unique()

# for mesure names
measure_name = measure_names[0]

# for genres
genre = genres[0]
genre_id = 0

# for subsample lengths
subsample_len = subsample_lens[0]
subsample_len_id = 0

# Restricted dataframe
sample_size_df[(sample_size_df.genre == genre) &
               (sample_size_df.subsample_len == subsample_len)]

# Groupby mean and sd
grpd_sample_size_df = sample_size_df.groupby(["reduced_sample_size"])
measure_mean = grpd_sample_size_df[measure_name].mean()
measure_std = grpd_sample_size_df[measure_name].std()
n_tests = int(len(sample_size_df) / len(measure_mean))