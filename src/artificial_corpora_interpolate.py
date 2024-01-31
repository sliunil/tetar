import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from collections import Counter
from lexical_diversity import tokenize, counter_to_zipf_data, TextGenerator


# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/real_corpora/real_corpora_indices_5000_10.csv"
corpora_folder_path = "../data/real_corpora/cleaned"
output_folder_path = "../results/artificial_corpora"
# generated size
gen_size = 10000


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
color_map = cm.get_cmap("cool", len(sorted_names))

# Loop on all files
real_fig, real_ax = plt.subplots()
esti_fig, esti_ax = plt.subplots()
tested_ranks = np.arange(1, gen_size+1)
for i, name in enumerate(sorted_names):
    with open(f"{corpora_folder_path}/{sorted_groups[i]}/{name}") as file:
        content = file.read()
    
    sample = tokenize(content.lower())
    counter = Counter(sample)
    ranks, frequencies, lm_model, shift = counter_to_zipf_data(counter)
    real_ax.plot(np.log(ranks), np.log(frequencies), color=color_map(i), 
                 alpha=0.5)
    predicted_log_freq = np.log(tested_ranks + shift)*lm_model.coef_[0] \
        + lm_model.intercept_
    esti_ax.plot(np.log(tested_ranks), predicted_log_freq, color=color_map(i),
                 alpha=0.5)
    
# Extract quantities of interest
slopes = ld_stat_df["zipf_slope"].to_numpy()
intercepts = ld_stat_df["zipf_intercept"].to_numpy()
shifts = ld_stat_df["zipf_shift"].to_numpy()
    
# Fit the generator
my_generator = TextGenerator()
my_generator.fit(slopes, intercepts, shifts)
    
# Slope space to explore
tested_slopes = np.linspace(np.min(slopes), np.max(slopes), len(sorted_names))
arti_fig, arti_ax = plt.subplots()
for i, tested_slope in enumerate(tested_slopes):
    slope, intercept, shift = my_generator.get_parameters(tested_slope)
    predicted_log_freq = np.log(tested_ranks + shift)*slope + intercept
    arti_ax.plot(np.log(tested_ranks), predicted_log_freq, color=color_map(i), 
                 alpha=0.5)
    
# Update and save plots 
plt.colorbar(cm.ScalarMappable(Normalize(np.min(slopes), np.max(slopes)), 
                               cmap=color_map), ax=real_ax, label="Slope")
plt.colorbar(cm.ScalarMappable(Normalize(np.min(slopes), np.max(slopes)), 
                               cmap=color_map), ax=esti_ax, label="Slope")
plt.colorbar(cm.ScalarMappable(Normalize(np.min(slopes), np.max(slopes)), 
                               cmap=color_map), ax=arti_ax, label="Slope")
real_ax.set_xlabel("log(rank)")
real_ax.set_ylabel("log(frequency)")
esti_ax.set_xlabel("log(rank)")
esti_ax.set_ylabel("log(frequency)")
arti_ax.set_xlabel("log(rank)")
arti_ax.set_ylabel("log(frequency)")
real_fig.savefig(f"{output_folder_path}/real_distrib.png", dpi=1200)
esti_fig.savefig(f"{output_folder_path}/interpolated_distrib.png", dpi=1200)
arti_fig.savefig(f"{output_folder_path}/artificial_distrib.png", dpi=1200)


