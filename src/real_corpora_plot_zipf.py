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
results_folder_path = "../results"



# -------------------------------
# --- CODE
# -------------------------------

# List the subfolders
subfolder_names = os.listdir(corpora_folder_path)

# Define colors
color_map = cm.get_cmap("hsv", len(subfolder_names) + 1)


for grp_id, subfolder_name in enumerate(subfolder_names):
    
    # To store group caracteristics
    grp_log_ranks, grp_log_freq = [], []

    # Get files
    file_names = os.listdir(f"{corpora_folder_path}/{subfolder_name}")

    for file_name in file_names:

        with open(f"{corpora_folder_path}/{subfolder_name}/{file_name}") as file:
            content = file.read()
        
        # Compute results
        sample = tokenize(content.lower())
        counter = Counter(sample)
        log_rank, log_freq, _, _ = counter_to_zipf_data(counter)
        
        # Store for group slopes  
        grp_log_ranks.extend(log_rank)
        grp_log_freq.extend(log_freq)
        
        # Plot the file
        plt.plot(log_rank, log_freq, alpha=0.5, linewidth=0.5, color=color_map(grp_id))
        
    # Linear regression model for the group
    lm_model = LinearRegression()
    lm_model.fit(np.array(grp_log_ranks).reshape(-1, 1), grp_log_freq)
    plt.axline(xy1=(0, lm_model.intercept_), slope=lm_model.coef_[0], 
               color=color_map(grp_id), label=subfolder_name)
    plt.legend()
    
# Save figure
plt.savefig(f"{results_folder_path}/real_corpora_zipf_plot.png", dpi=1200)
