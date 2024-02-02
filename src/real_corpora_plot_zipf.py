import os
from collections import Counter
from lexical_diversity import tokenize, counter_to_zipf_data
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
import numpy as np



# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

# The corpora folder path
corpora_folder_path = "../data/real_corpora/cleaned"
# The results folder path
results_file_path = "../results/real_corpora_zipf_plot.png"



# -------------------------------
# --- CODE
# -------------------------------

# List the subfolders
subfolder_names = os.listdir(corpora_folder_path)

# Define colors
color_map = cm.get_cmap("hsv", len(subfolder_names) + 1)

    
fig, axs = plt.subplots(len(subfolder_names))

for grp_id, subfolder_name in enumerate(subfolder_names):

    # Get files
    file_names = os.listdir(f"{corpora_folder_path}/{subfolder_name}")

    for file_name in file_names:

        with open(f"{corpora_folder_path}/{subfolder_name}/{file_name}") \
            as file:
            content = file.read()
        
        # Compute results
        sample = tokenize(content.lower())
        counter = Counter(sample)
        ranks, frequencies, lm_model, z_shift = counter_to_zipf_data(counter)
        
        # Plot the file
        axs[grp_id].scatter(np.log(ranks), np.log(frequencies), alpha=0.5, 
                            color=color_map(grp_id), marker=".", s=0.01)
        axs[grp_id].plot(np.log(ranks), 
                         lm_model.predict(
                             np.log(ranks + z_shift).reshape(-1, 1)),
                         alpha=0.8, linewidth=0.8, color="gray")
    
# Save figure
plt.savefig(results_file_path, dpi=1200)

