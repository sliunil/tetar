import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/artificial_corpora/zm_res_1/zm_1_merge.csv"
output_folder_path = "../results"
output_file_prefix = "zm_plot"
min_prop_to_compute_mtld = 0.6


# -------------------------------
# --- CODE
# -------------------------------

# Load dataset
ac_df = pd.read_csv(input_file_path)

# Get measure names and define drawing 
measure_names = ac_df.columns.to_numpy()[5:]
measure_names = measure_names[np.array([0, 1, 3, 4])]
measure_clean_names = ["Sample Entropy", "Subsample Entropy (both)", 
                       "HD-D", "MTLD"]
color_map = cm.get_cmap("binary", len(measure_names) + 1)
markers = ['o', 'v', 's', 'd']
    
# Get the graph_type
graph_types = ac_df["type"].unique()
graph_type_names = ["Variety", "Power", "Shift"]

# Normalization
for measure_name in measure_names:
    min_measure = ac_df[measure_name].min()
    max_measure = ac_df[measure_name].max()
    ac_df.loc[:, measure_name] = ac_df[measure_name] - min_measure
    ac_df.loc[:, measure_name] = ac_df[measure_name] \
        / (max_measure - min_measure)

for id_type, graph_type in enumerate(graph_types):

    ac_type_df = ac_df[ac_df.type == graph_type]

    plt.figure()

    for measure_name_id, measure_name in enumerate(measure_names):
        
        # Groupby mean and sd
        grpd_selected_df = ac_type_df.groupby([graph_type])
        measure_mean = grpd_selected_df[measure_name].mean()
        measure_std = grpd_selected_df[measure_name].std()
        n_theo_tests = int(len(ac_type_df) / len(measure_mean))
        n_tests = grpd_selected_df[measure_name].count()
        measure_mean.loc[n_tests / n_theo_tests 
                            < min_prop_to_compute_mtld] = np.nan

        # Plot it 
        plt.plot(measure_mean.index, measure_mean.values, 
                color=color_map(measure_name_id+1),
                marker=markers[measure_name_id], 
                label=measure_clean_names[measure_name_id])
            
    plt.legend(loc='lower right', fontsize='8')
    plt.xlabel(graph_type_names[id_type])
    plt.ylim((0, 1))
    plt.savefig(f"{output_folder_path}/{output_file_prefix}_{graph_type}.png", 
                dpi=300)
