import pandas as pd
from lexical_diversity import TextGenerator


# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/real_corpora/real_corpora_indices_20_10.csv"
output_folder_path = "../results/artificial_corpora"


# -------------------------------
# --- CODE
# -------------------------------

# Load dataset
ld_stat_df = pd.read_csv(input_file_path, index_col=0)

# Get groups
groups = ld_stat_df["group"]

# Extract quantities of interest
slopes = ld_stat_df["zipf_slope"].to_numpy()
intercepts = ld_stat_df["zipf_intercept"].to_numpy()
shifts = ld_stat_df["zipf_shift"].to_numpy()

# Fit the generator
my_generator = TextGenerator()
my_generator.fit(slopes, intercepts, shifts)

# Plot the relationships
in_sl_fig, _, sh_sl_fig, _, sh_in_fig, _ = my_generator.plot(groups)
in_sl_fig.savefig(f"{output_folder_path}/slope_intercept.png", dpi=1200)
sh_sl_fig.savefig(f"{output_folder_path}/slope_shift.png", dpi=1200)
sh_in_fig.savefig(f"{output_folder_path}/intercept_shift.png", dpi=1200)
