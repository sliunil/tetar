import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/sample_size/results_large/large_merge.csv"
output_folder_path = "../results"
output_file_prefix = "sample_size_plot"
min_prop_to_compute_mtld = 0.6


# -------------------------------
# --- CODE
# -------------------------------

# Load dataset
sample_size_df = pd.read_csv(input_file_path)
# Rename literature in novels
sample_size_df.loc[sample_size_df.genre == "literature", "genre"]= "novels"

# Get genres (color)
genres = sample_size_df.genre.unique()
genres.sort()
color_map = cm.get_cmap("binary", len(genres) + 1)
markers = ['o', 'x', 'v']
fillstyles = ['full', 'none', 'none']

# Get measure names 
measure_names = sample_size_df.columns.to_numpy()[6:]
measure_clean_names = ["Sample Entropy", "Subsample Entropy (rdm)", 
                       "Subsample Entropy (mav)", "HD-D", 
                       "MTLD"]

for measure_name_id, measure_name in enumerate(measure_names):

    # Get subsample lengths (linewidth)
    if not measure_name == 'mtld':
        subsample_lens = sample_size_df.subsample_len.unique()
        subsample_lens = subsample_lens[::2]
    else:
        subsample_lens = sample_size_df.mtld_thresholds.unique()
        subsample_lens = subsample_lens[1:-1]
    linestyles = [(0, (3*i+1, 0.5*i+1)) for i, _ in enumerate(subsample_lens)]
    linestyles[-1] = 'solid'
    
    plt.figure()

    # for genres
    for genre_id, genre in enumerate(genres):

        # for subsample lengths
        for subsample_len_id, subsample_len in enumerate(subsample_lens):

            # Selected dataframe
            if not measure_name == 'mtld':
                selected_df = sample_size_df[(sample_size_df.genre == genre) &
                                            (sample_size_df.subsample_len == 
                                            subsample_len)]
            else: 
                selected_df = sample_size_df[(sample_size_df.genre == genre) &
                                            (sample_size_df.mtld_thresholds == 
                                            subsample_len)]

            # Groupby mean and sd
            grpd_selected_df = selected_df.groupby(["reduced_sample_size"])
            measure_mean = grpd_selected_df[measure_name].mean()
            measure_std = grpd_selected_df[measure_name].std()
            n_theo_tests = int(len(selected_df) / len(measure_mean))
            n_tests = grpd_selected_df[measure_name].count()
            measure_mean.loc[n_tests / n_theo_tests 
                             < min_prop_to_compute_mtld] = np.nan

            # Plot it 
            if genre_id == 0 and not (measure_name == 'sample_entropy'):
                plt.plot(measure_mean.index, measure_mean.values,
                        color="black",
                        linestyle=linestyles[subsample_len_id],
                        label=subsample_len)
            if subsample_len_id == len(subsample_lens) - 1:
                plt.plot(measure_mean.index, measure_mean.values,
                        color=color_map(genre_id+1),
                        linestyle=linestyles[subsample_len_id],
                        label=genre,
                        marker=markers[genre_id], 
                        fillstyle=fillstyles[genre_id])
            plt.errorbar(measure_mean.index, measure_mean.values, 
                        yerr=measure_std.values*1.96/np.sqrt(n_theo_tests), 
                        color=color_map(genre_id+1),
                        linestyle=linestyles[subsample_len_id],
                        marker=markers[genre_id], 
                        fillstyle=fillstyles[genre_id])
    plt.xlabel("Sample length")
    plt.ylabel(measure_clean_names[measure_name_id])
    plt.legend(loc='upper right', fontsize='10')
    plt.savefig(f"{output_folder_path}/{output_file_prefix}_{measure_name}.png", 
                dpi=300, bbox_inches='tight')
