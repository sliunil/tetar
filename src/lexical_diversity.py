from collections import Counter
import math
from pathlib import Path
import random
import re
import numpy as np
from sklearn.linear_model import LinearRegression

__version__ = 0.2
__authors__ = ["aris.xanthos@unil.ch", "guillaume.guex@unil.ch"]

def tokenize(my_string, token_regex=r"\w+"):
    """Tokenize string into list of tokens based on regex describing tokens."""
    return re.findall(token_regex, my_string)

def draw_reduced_sample(sample, reduced_sample_size):
    """Draw a subsamble from a sample, where token order is preserved"""
    sample_size = len(sample)
    if sample_size < reduced_sample_size:
        raise ValueError("Can't draw a subsample larger than sample size")
    reduce_samble_start = random.randint(0, sample_size - reduced_sample_size)
    return sample[reduce_samble_start:(reduce_samble_start + 
                                       reduced_sample_size)]

def sample_entropy(sample, base=2):
    """Compute sample entropy based on a list of items."""
    if len(sample) == 0:
        raise ValueError("Can't compute sample entropy with empty list.")
    return(counter_to_sample_entropy(Counter(sample), base))

def counter_to_sample_entropy(counter, base=2):
    """Compute sample entropy based on a Counter."""
    if len(counter) == 0:
        raise ValueError("Can't compute sample entropy with empty Counter.")
    # my_sum = 0
    # weighted_sum_of_logs = 0
    # for freq in counter.values():
    #     if freq:
    #         my_sum += freq
    #         weighted_sum_of_logs += freq * math.log(freq, base)
    # return math.log(my_sum, base) - weighted_sum_of_logs/my_sum
    np_freqs = np.array(list(counter.values()))
    np_freqs = np_freqs[np_freqs > 0]
    np_sum = np.sum(np_freqs)
    return np.emath.logn(base, np_sum) \
        - np.sum(np_freqs * np.emath.logn(base, np_freqs))/np_sum

def subsample_entropy(sample, subsample_len, num_subsamples=1000, 
                      mode="random", base=2):
    """Compute subsample entropy (and standard deviation) based on a sample."""
    sample_len = len(sample)

    # Raise exception if subsample length > sample length...
    if subsample_len > sample_len:
        raise ValueError("Subsample length must be less than or equal to "
                         "sample length.")

    # Raise exception if subsample length <= 0...
    elif sample_len <= 0:
        raise ValueError("Subsample length must be greater than 0.")

    # Return entropy of sample is subsample length = sample length...
    elif subsample_len == sample_len:
        return sample_entropy(sample, base), 0

    # If 0 < subsample length < sample length, compute subsample entropy...
    my_sum = sum_of_squares = 0

    # Random subsampling...
    if mode == "random":
        my_num_subsamples = num_subsamples
        for _ in range(num_subsamples):
            entropy = sample_entropy(get_random_subsample(sample, 
                                                          subsample_len),
                                     base)
            my_sum += entropy
            sum_of_squares += entropy**2

    # Sliding window (NB: arg num_samples is overriden)...
    elif mode == "window":
        my_num_subsamples = sample_len-subsample_len+1

        # Get Counter and sample entropy for first window.
        counter = Counter(sample[:subsample_len])
        entropy = counter_to_sample_entropy(counter, base)
        my_sum += entropy
        sum_of_squares += entropy**2

        # For each consecutive window...
        for pos in range(sample_len-subsample_len):

            # Update counter...
            counter[sample[pos]] -= 1
            counter[sample[pos+subsample_len]] += 1

            # Compute entropy
            entropy = counter_to_sample_entropy(counter, base)
            my_sum += entropy
            sum_of_squares += entropy**2

    # Compute and return average entropy and standard deviation...
    average = my_sum/my_num_subsamples
    standard_deviation = sum_of_squares/my_num_subsamples - average**2
    return average, standard_deviation

def get_random_subsample(sample, subsample_len):
    """Return a random subsample with subsample_len tokens picked without
    replacement from sample (NB: original ordering of tokens is preserved).
    """
    indices = list(range(len(sample)))
    random.shuffle(indices)
    subsample = [sample[i] for i in sorted(indices[:subsample_len])]
    return subsample

def import_taaled_silently():
    """Import taaled without displaying annoying messages (based on
    https://stackoverflow.com/questions/60324614/suppress-output-on-library-import-in-python
    """
    import io
    import sys
    text_trap = io.StringIO()
    sys.stdout = text_trap
    from taaled import ld
    sys.stdout = sys.__stdout__
    return ld

LD_OBJECT = import_taaled_silently().lexdiv()

def MTLD(sample, ttr_threshold=.72):
    """Wrapper for the MTLD method in taaled."""
    return LD_OBJECT.MTLD(sample, ttrval=ttr_threshold)

def counter_to_zipf_data(counter, shifts=np.linspace(0, 200, 401)):
    """Compute zipf data from a counter: give ranks, frequencies
    linear model fitted on (shifted) log-rank vs log-freq, and shift of ranks"""
    
    # Get the frequencies
    frequencies = list(counter.values())
    frequencies.sort(reverse=True)

    # Compute log_rank and log_freq 
    ranks = np.array(range(1, len(frequencies) + 1))
    log_freqs = np.log(frequencies)
        
    # Linear regression model 
    lm_model = LinearRegression()
    mses = []
    for shift in shifts:
        shifted_log_rank = np.log(ranks + shift)
        lm_model.fit(shifted_log_rank.reshape(-1, 1), log_freqs)
        #scores.append(lm_model.score(shifted_log_rank.reshape(-1, 1), 
        #                             np.log(frequencies)))
        freq_estimates = lm_model.predict(shifted_log_rank.reshape(-1, 1))
        mses.append(np.mean((freq_estimates - log_freqs)**2))
    estimated_shift = shifts[np.where(mses == np.min(mses))[0][0]]
    lm_model.fit(np.log(ranks + estimated_shift).reshape(-1, 1), 
                 np.log(frequencies))
    
    return ranks, frequencies, lm_model, estimated_shift
    
    
