import pandas as pd
import numpy as np



# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/real_corpora/real_corpora_indices_5000_10.csv"
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

# ---- Zipf study 

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

model_isl = LinearRegression()
model_isl.fit(num_data["zipf_intercept"].to_numpy().reshape(-1, 1), 
             num_data["zipf_slope"])
plt.figure()
plt.scatter(num_data["zipf_intercept"], num_data["zipf_slope"])
plt.axline((0, model_isl.intercept_), 
           (1, model_isl.intercept_ + model_isl.coef_[0]),
           color="red")

model_ish = LinearRegression()
model_ish.fit(np.log(num_data["zipf_intercept"]).to_numpy().reshape(-1, 1), 
              np.log(num_data["zipf_shift"]))
sorted_i = np.sort(num_data["zipf_intercept"])
sh_from_i = np.exp(model_ish.predict(np.log(sorted_i).reshape(-1, 1)))
plt.figure()
plt.scatter(num_data["zipf_intercept"], num_data["zipf_shift"])
plt.plot(sorted_i, sh_from_i, color="red")

model_slsh = LinearRegression()
model_slsh.fit(np.log(-num_data["zipf_slope"]).to_numpy().reshape(-1, 1), 
               np.log(num_data["zipf_shift"]))
sorted_sl = np.sort(-num_data["zipf_slope"])
sh_from_sl = np.exp(model_slsh.predict(np.log(sorted_sl).reshape(-1, 1)))
plt.figure()
plt.scatter(-num_data["zipf_slope"], num_data["zipf_shift"])
plt.plot(sorted_sl, sh_from_sl, color="red")

print(f"Model intercept-slope : {model_isl.intercept_}, {model_isl.coef_[0]}")
print(f"Model intercept-shift : {model_ish.intercept_}, {model_ish.coef_[0]}")
print(f"Model slope-shift : {model_slsh.intercept_}, {model_slsh.coef_[0]}")

# pente entre -1 et -2

p = -1.5
intercept = (p - model_isl.intercept_) / model_isl.coef_[0]
shift = np.exp(model_slsh.predict(np.array(np.log(-p)).reshape(-1, 1)))[0]
shift_2 = np.exp(model_ish.predict(np.array(np.log(intercept)).reshape(-1, 1)))[0]
