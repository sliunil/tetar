import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/real_corpora/real_corpora_indices_5000_10.csv"
output_folder_path = "../results/real_corpora"


# -------------------------------
# --- CODE
# -------------------------------

# Load dataset
ld_stat_df = pd.read_csv(input_file_path, index_col=0)

# Keep numerical data
num_data = ld_stat_df.drop(["name", "group"], axis=1)

# Extract quantities of interest
slopes = num_data["zipf_slope"].to_numpy()
intercepts = num_data["zipf_intercept"].to_numpy()
shifts = num_data["zipf_shift"].to_numpy()

# Produce models 
model_sl_in = LinearRegression()
model_sl_in.fit(slopes.reshape(-1, 1), intercepts)

model_sl_sh = LinearRegression()
model_sl_sh.fit(np.log(-slopes).reshape(-1, 1), np.log(shifts))

model_in_sh = LinearRegression()
model_in_sh.fit(np.log(intercepts).reshape(-1, 1), np.log(shifts))

# Produce graphs
sorted_sl = np.sort(slopes)
in_from_sl = model_sl_in.predict(sorted_sl.reshape(-1, 1))
plt.figure()
plt.scatter(slopes, intercepts)
plt.plot(sorted_sl, in_from_sl, color="red")
plt.xlabel("Slope")
plt.ylabel("Intercept")
plt.savefig(f"{output_folder_path}/slope_intercept")

sh_from_sl = np.exp(model_sl_sh.predict(np.log(-sorted_sl).reshape(-1, 1)))
plt.figure()
plt.scatter(slopes, shifts)
plt.xlabel("Slope")
plt.ylabel("Shift")
plt.plot(sorted_sl, sh_from_sl, color="red")
plt.savefig(f"{output_folder_path}/slope_shift")

sorted_in = np.sort(intercepts)
sh_from_in = np.exp(model_in_sh.predict(np.log(sorted_in).reshape(-1, 1)))
plt.figure()
plt.scatter(intercepts, shifts)
plt.xlabel("Intercept")
plt.ylabel("Shift")
plt.plot(sorted_in, sh_from_in, color="red")
plt.savefig(f"{output_folder_path}/intercept_shift")

# pente entre -1 et -2

# p = -1.5
# intercept = (p - model_isl.intercept_) / model_isl.coef_[0]
# shift = np.exp(model_slsh.predict(np.array(np.log(-p)).reshape(-1, 1)))[0]
# shift_2 = np.exp(model_ish.predict(np.array(np.log(intercept)).reshape(-1, 1)))[0]
