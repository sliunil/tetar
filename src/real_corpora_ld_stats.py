import pandas as pd
import numpy as np

input_file_path = "../results/real_corpora_indices_5000_10.csv"
output_file_path = "../results/correlation_matrix_new.xlsx"

data = pd.read_csv(input_file_path, index_col=0)

num_data = data.drop(["name", "group"], axis=1)

correlations = np.corrcoef(num_data.to_numpy().T)

correlations_df = pd.DataFrame(correlations, columns=num_data.columns, index=num_data.columns)

correlations_df.style.background_gradient(cmap="Reds").to_excel(output_file_path)
