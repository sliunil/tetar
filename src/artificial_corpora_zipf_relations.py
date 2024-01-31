import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lexical_diversity import TextGenerator


# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/real_corpora/real_corpora_indices_5000_10.csv"
output_folder_path = "../results/real_corpora"


# -------------------------------
# --- CODE
# -------------------------------

# Load dataset
ld_stat_df = pd.read_csv(input_file_path, index_col=0)

# Get groups
groups = ld_stat_df["group"]

# Keep numerical data
num_data = ld_stat_df.drop(["name", "group"], axis=1)

# Extract quantities of interest
slopes = num_data["zipf_slope"].to_numpy()
intercepts = num_data["zipf_intercept"].to_numpy()
shifts = num_data["zipf_shift"].to_numpy()

# Fit the generator
my_generator = TextGenerator()
my_generator.fit(slopes, intercepts, shifts)

# Plot the relationships
in_sl_fig, _, sh_sl_fig, _, sh_in_fig, _ = my_generator.plot(groups)
in_sl_fig.savefig(f"{output_folder_path}/slope_intercept")
sh_sl_fig.savefig(f"{output_folder_path}/slope_shift")
sh_in_fig.savefig(f"{output_folder_path}/intercept_shift")