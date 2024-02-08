import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from collections import Counter
from lexical_diversity import tokenize, counter_to_zipf_data


# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/real_corpora/real_corpora_indices_4_zm.csv"
corpora_folder_path = "../data/real_corpora/cleaned"
output_folder_path = "../results/artificial_corpora"


# -------------------------------
# --- CODE
# -------------------------------

# Load dataset and sort it
ld_stat_df = pd.read_csv(input_file_path, index_col=0)
ld_stat_df = ld_stat_df.sort_values("zipf_slope")

# Get sorted name and group
sorted_names = ld_stat_df["name"].values
sorted_groups = ld_stat_df["group"].values

# Create cmap
color_map = cm.get_cmap("binary", len(sorted_names))

# Loop on all files
real_fig, real_ax = plt.subplots()
esti_fig, esti_ax = plt.subplots()
for i, name in enumerate(sorted_names):
    with open(f"{corpora_folder_path}/{sorted_groups[i]}/{name}") as file:
        content = file.read()
    
    sample = tokenize(content.lower())
    counter = Counter(sample)
    ranks, frequencies, zipf_params = counter_to_zipf_data(counter)
    real_ax.plot(ranks, frequencies, color=color_map(i), alpha=0.5)
    predicted_freq = np.exp(np.log(ranks + zipf_params[1])*zipf_params[0])
    predicted_freq = predicted_freq / sum(predicted_freq)*sum(frequencies)
    esti_ax.plot(ranks, predicted_freq, color=color_map(i), alpha=0.5)
    
# Extract quantities of interest
slopes = ld_stat_df["zipf_slope"].to_numpy()
intercepts = ld_stat_df["zipf_intercept"].to_numpy()
shifts = ld_stat_df["zipf_shift"].to_numpy()
    
# Compute the mean values for the interpolations
mean_slope = np.mean(slopes)
mean_shift = np.mean(shifts)
mean_length = int(ld_stat_df["length"].mean())
mean_variety = int(ld_stat_df["variety"].mean())
tested_ranks = np.arange(1, mean_variety+1)
    
# The slope space
tested_slopes = np.linspace(np.min(slopes), np.max(slopes), 
                            len(sorted_names))
slope_fig, slope_ax = plt.subplots()
for i, tested_slope in enumerate(tested_slopes):
    predicted_freq = (tested_ranks + mean_shift)**tested_slope
    predicted_freq = predicted_freq / sum(predicted_freq) * mean_length 
    slope_ax.plot(tested_ranks, predicted_freq, color=color_map(i), 
                  alpha=0.5)
    
# The shift space 
tested_shifts = np.linspace(np.min(shifts), np.max(shifts), 
                            len(sorted_names))
shift_fig, shift_ax = plt.subplots()
for i, tested_shift in enumerate(tested_shifts):
    predicted_freq = (tested_ranks + tested_shift)**mean_slope
    predicted_freq = predicted_freq / sum(predicted_freq) * mean_length 
    shift_ax.plot(tested_ranks, predicted_freq, color=color_map(i), alpha=0.5)
    
# The variety 
tested_varieties = np.linspace(ld_stat_df["variety"].min(), 
                               ld_stat_df["variety"].max(), 
                               len(sorted_names)).astype(int)
variety_fig, variety_ax = plt.subplots()
for i, tested_variety in enumerate(tested_varieties):
    ranks = np.arange(1, tested_variety+1)
    predicted_freq = (ranks + mean_shift)**mean_slope
    predicted_freq = predicted_freq / sum(predicted_freq) * mean_length 
    variety_ax.plot(ranks, predicted_freq, color=color_map(i), alpha=0.5)
    
# Update and save plots 
plt.colorbar(cm.ScalarMappable(Normalize(np.min(tested_slopes), 
                                         np.max(tested_slopes)), 
                               cmap=color_map), ax=slope_ax, label="Power")
plt.colorbar(cm.ScalarMappable(Normalize(np.min(tested_shifts), 
                                         np.max(tested_shifts)), 
                               cmap=color_map), ax=shift_ax, label="Shift")
plt.colorbar(cm.ScalarMappable(Normalize(np.min(tested_varieties), 
                                         np.max(tested_varieties)), 
                               cmap=color_map), ax=variety_ax, label="Variety")
real_ax.set_xlabel("Rank")
real_ax.set_ylabel("Frequency")
esti_ax.set_xlabel("Rank")
esti_ax.set_ylabel("Frequency")
slope_ax.set_xlabel("Rank")
slope_ax.set_ylabel("Frequency")
shift_ax.set_xlabel("Rank")
shift_ax.set_ylabel("Frequency")
variety_ax.set_xlabel("Rank")
variety_ax.set_ylabel("Frequency")
real_ax.set_xscale("log")
real_ax.set_yscale("log")
esti_ax.set_xscale("log")
esti_ax.set_yscale("log")
slope_ax.set_xscale("log")
slope_ax.set_yscale("log")
shift_ax.set_xscale("log")
shift_ax.set_yscale("log")
variety_ax.set_xscale("log")
variety_ax.set_yscale("log")
real_fig.savefig(f"{output_folder_path}/real_distrib.png", dpi=300)
esti_fig.savefig(f"{output_folder_path}/interpolated_distrib.png", dpi=300)
slope_fig.savefig(f"{output_folder_path}/slope_distrib.png", dpi=300)
shift_fig.savefig(f"{output_folder_path}/shift_distrib.png", dpi=300)
variety_fig.savefig(f"{output_folder_path}/variety_distrib.png", dpi=300)
