import pandas as pd
import numpy as np
from collections import Counter
from LTTL.Utils import get_expected_subsample_variety
from lexical_diversity import TextGenerator, sample_entropy, \
    subsample_entropy, MTLD
import multiprocessing as mp
from itertools import product

# -------------------------------
# --- SCRIPT PARAMETERS
# -------------------------------

input_file_path = "../results/real_corpora/real_corpora_indices_200_10.csv"
corpora_folder_path = "../data/real_corpora/cleaned"
output_folder_path = "../results/artificial_corpora"
# Tested slopes
tested_slopes = np.linspace(-1, -2, 20)
# Generated size
gen_size = 10000
# Number of samples
num_samples = 10
# The length of subsamples / mtld thresholds
subsample_lens = [50, 100, 500, 1000, 2000]
mtld_thresholds = [0.84, 0.78, 0.72, 0.66, 0.6]

# -------------------------------
# --- CODE
# -------------------------------

# The number of cpu available
n_cpu = mp.cpu_count()

# Load dataset
ld_stat_df = pd.read_csv(input_file_path, index_col=0)
    
# Extract quantities of interest
slopes = ld_stat_df["zipf_slope"].to_numpy()
intercepts = ld_stat_df["zipf_intercept"].to_numpy()
shifts = ld_stat_df["zipf_shift"].to_numpy()
    
# Fit the generator
my_generator = TextGenerator()
my_generator.fit(slopes, intercepts, shifts)

# Function for multiprocess computing
def compute_ld_indice(parameters):
    (sample, slope, id_draw), \
        (subsample_len, mtld_threshold, num_subsamples) = parameters
    smple_entropy = sample_entropy(sample)
    sample_counter = Counter(sample)
    if subsample_len > len(sample):
        subsample_entropy_rdm = np.nan
        subsample_entropy_mav = np.nan
        exp_variety = np.nan
    else:
        subsample_entropy_rdm, _ = subsample_entropy(sample, 
                                                     subsample_len, 
                                                     num_subsamples)
        subsample_entropy_mav, _ = subsample_entropy(sample, 
                                                     subsample_len, 
                                                     num_subsamples,
                                                     mode="window")
        exp_variety = get_expected_subsample_variety(sample_counter,
                                                     subsample_len)
    if len(sample_counter) / len(sample) > mtld_threshold:
        mtld = np.nan
    else:
        mtld = MTLD(sample, mtld_threshold)
    return smple_entropy, subsample_entropy_rdm, subsample_entropy_mav, \
        exp_variety, mtld

# For slopes 
tested_slope = tested_slopes[0]
# Generate the samples
samples = my_generator.generate_samples(tested_slope, gen_size, 
                                        num_samples)



# For subsample_len
subsample_len = subsample_lens[0]

product(tested_slopes, range(num_samples))

samples = []

[my_generator.generate_samples(tested_slope, gen_size, 
                                        num_samples)
                    for tested_slope, id_draw in 
                    product(tested_slopes, range(num_samples))]

