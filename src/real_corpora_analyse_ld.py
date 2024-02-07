import pandas as pd
import numpy as np



# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/real_corpora/real_corpora_indices_1_5000.csv"
output_cor_path = "../results/real_corpora_correlation.xlsx"
output_mean_path = "../results/real_corpora_group_mean.csv"



# -------------------------------
# --- CODE
# -------------------------------

# ---- Correlations

# Load dataset
ld_stat_df = pd.read_csv(input_file_path, index_col=0)

# Keep numerical data
num_data = ld_stat_df.drop(["name", "group"], axis=1)

# Compute correlation and save them with color
correlations_df = pd.DataFrame(np.corrcoef(num_data.to_numpy().T), 
                               columns=num_data.columns, 
                               index=num_data.columns)
correlations_df.style.background_gradient(cmap="coolwarm").to_excel(
    output_cor_path)

# ---- Groups means

# Compute de mean by group
ld_stat_df_gpd = ld_stat_df.drop("name", axis=1).groupby("group")
ld_mean_df = ld_stat_df_gpd.mean()

# Save it
ld_mean_df.to_csv(output_mean_path)