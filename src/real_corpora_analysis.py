import os
from collections import Counter
from LTTL.Utils import get_expected_subsample_variety
from lexical_diversity import tokenize, sample_entropy, subsample_entropy, MTLD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression


# The corpora folder path§
corpora_folder_path = "../data/real_corpora/cleaned"

# To store results
names, groups, log_ranks, log_freqs, intercepts, slopes \
    = [], [], [], [], [], []

# List the subfolders
subfolder_names = os.listdir(corpora_folder_path)

for subfolder_name in subfolder_names:

    # Get files
    file_names = os.listdir(f"{corpora_folder_path}/{subfolder_name}")

    for file_name in file_names:

        with open(f"{corpora_folder_path}/{subfolder_name}/{file_name}") as file:
            content = file.read()
            
        sample = tokenize(content.lower())

        # Get the frequencies
        counter = Counter(sample)
        frequencies = list(counter.values())
        frequencies.sort(reverse=True)

        # Compute log_rank and log_freq 
        log_rank = np.log(list(range(1, len(frequencies) + 1)))
        log_freq = np.log(frequencies)
        
        # Linear regression model 
        lm_model = LinearRegression()
        lm_model.fit(log_rank.reshape(-1, 1), log_freq)

        # Store results 
        names.append(file_name)
        groups.append(subfolder_name)
        log_ranks.append(log_rank)
        log_freqs.append(log_freq)
        intercepts.append(lm_model.intercept_)
        slopes.append(lm_model.coef_[0])
        
# Compute group_slope 
for 

# To manage colors
color_map = cm.get_cmap("hsv", len(subfolder_names))

