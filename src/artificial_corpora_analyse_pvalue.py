import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/artificial_corpora/acz_results_3/acz_3_merge.csv"
output_folder_path = "../results"
output_file_prefix = "acz_pvalue_plot"
min_prop_to_compute_mtld = 0.6
shift_value = -1
rolling_value = 20


# -------------------------------
# --- CODE
# -------------------------------

# Load dataset
ac_df = pd.read_csv(input_file_path)

# Get measure names 
measure_names = ac_df.columns.to_numpy()[4:]
measure_clean_names = ["Sample Entropy", "Subsample Entropy (rdm)", 
                       "Subsample Entropy (mav)", "HD-D", 
                       "MTLD"]

for measure_name_id, measure_name in enumerate(measure_names):

    # Get subsample lengths (linewidth)
    if not measure_name == 'mtld':
        subsample_lens = ac_df.subsample_len.unique()
    else:
        subsample_lens = ac_df.mtld_thresholds.unique()
    linestyles = [(0, (3*i+1, 0.5*i+1)) for i, _ in enumerate(subsample_lens)]
    linestyles[-1] = 'solid'
    
    plt.figure()

    # for subsample lengths
    for subsample_len_id, subsample_len in enumerate(subsample_lens):

        # Selected dataframe
        if not measure_name == 'mtld':
            selected_df = ac_df[ac_df.subsample_len == subsample_len]
        else: 
            selected_df = ac_df[ac_df.mtld_thresholds == subsample_len]

        # Groupby mean and sd
        grpd_selected_df = selected_df.groupby(["slope"])
        measure_mean = grpd_selected_df[measure_name].mean()
        measure_std = grpd_selected_df[measure_name].std()
        n_theo_tests = int(len(selected_df) / len(measure_mean))
        n_tests = grpd_selected_df[measure_name].count()
        measure_mean.loc[n_tests / n_theo_tests 
                            < min_prop_to_compute_mtld] = np.nan
        
        # Pvalue computation
        mx = measure_mean
        mx_s = measure_mean.shift(shift_value)
        nx = n_tests
        nx_s = n_tests.shift(shift_value)
        sx = measure_std
        sx_s = measure_std.shift(shift_value)
        val = np.abs(mx - mx_s) * np.sqrt(nx + nx_s - 2) \
            / np.sqrt((1/nx + 1/nx_s)*(nx*sx**2 + nx_s*sx_s**2))
        val = val.rolling(rolling_value).mean()
        pval = 2*(1 - t.cdf(val, df=(nx + nx_s - 2)))
        
        # Plot it 
        if not measure_name == 'sample_entropy':
            plt.plot(measure_mean.index, pval,
                    color="black",
                    linestyle=linestyles[subsample_len_id],
                    label=subsample_len)
        else:
            plt.plot(measure_mean.index, pval,
                    color="black",
                    linestyle=linestyles[-1])
            
        # plt.errorbar(measure_mean.index, measure_mean.values, 
        #             yerr=measure_std.values*1.96/np.sqrt(n_theo_tests), 
        #             color="black",
        #             linestyle=linestyles[subsample_len_id])
        
    
    if not measure_name == 'sample_entropy':
        plt.legend(loc='upper right', fontsize='8')
    plt.xlabel("Slope")
    plt.ylabel(f"p-value for {measure_clean_names[measure_name_id]}")
    plt.yscale("log")
    plt.axhline(y=0.01, color="black")
    plt.ylim([0, 0.1])
    plt.savefig(f"{output_folder_path}/{output_file_prefix}_{measure_name}.png", 
                dpi=1200)
