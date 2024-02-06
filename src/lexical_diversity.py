from collections import Counter
import math
from pathlib import Path
import random
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

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
    scores = []
    mses = []
    for shift in shifts:
        shifted_log_rank = np.log(ranks + shift)
        lm_model.fit(shifted_log_rank.reshape(-1, 1), log_freqs)
        scores.append(lm_model.score(shifted_log_rank.reshape(-1, 1), 
                                     np.log(frequencies)))
        freq_estimates = np.exp(
            lm_model.predict(shifted_log_rank.reshape(-1, 1)))
        mses.append(np.mean((freq_estimates - frequencies)**2))
    estimated_shift = shifts[np.where(mses == np.min(mses))[0][0]]
    lm_model.fit(np.log(ranks + estimated_shift).reshape(-1, 1), 
                 np.log(frequencies))
    
    return ranks, frequencies, lm_model, estimated_shift

class TextGenerator:
    """A class to create artificial texts, following a zipf-mandelbrot 
    distribution parametrized by the slope only. The intercept and the shift 
    are deduced from data"""
    
    # Constructor
    def __init__(self):
        self.slopes = np.empty((0))
        self.intercepts = np.empty((0))
        self.shifts = np.empty((0))
        self.model_sl_in = LinearRegression()
        self.model_sl_sh = LinearRegression()
        self.model_in_sh = LinearRegression()
        
    # Fit the models
    def fit(self, slopes, intercepts, shifts):
        self.slopes = slopes
        self.intercepts = intercepts
        self.shifts = shifts
        self.model_sl_in.fit(slopes.reshape(-1, 1), intercepts)
        if np.sum(self.shifts) > 1e-10:
            self.model_sl_sh.fit(np.log(-slopes).reshape(-1, 1), 
                                 np.log(shifts))
            self.model_in_sh.fit(np.log(intercepts).reshape(-1, 1), 
                                 np.log(shifts))
            

    # Plot the relationships
    def plot(self, groups=None, which_cmap="binary"):
        
        # Group colors 
        if groups is not None:
            gr_fact = np.unique(groups, return_inverse=True)
            markers = ['o', 'x', 'v']
            fillstyles = ['full', 'none', 'none']
            cmap = cm.get_cmap(which_cmap, len(gr_fact[0]) + 1)
            
        # Intercepts from slopes
        sorted_sl = np.sort(self.slopes)
        in_from_sl = self.model_sl_in.predict(sorted_sl.reshape(-1, 1))
        in_sl_fig, in_sl_ax = plt.subplots()
        if groups is not None:
            for id_gr, gr in enumerate(gr_fact[0]): 
                in_sl_ax.scatter(self.slopes[gr_fact[1] == id_gr], 
                                 self.intercepts[gr_fact[1] == id_gr], 
                                 c=cmap(id_gr+1), 
                                 marker=markers[id_gr],
                                 facecolors=fillstyles[id_gr], 
                                 label=gr)
            in_sl_ax.legend()
        else:
            in_sl_ax.scatter(self.slopes, self.intercepts)
        in_sl_ax.plot(sorted_sl, in_from_sl, color="black")
        in_sl_ax.set_xlabel("Slope")
        in_sl_ax.set_ylabel("Intercept")
        
        # Shifts from slopes
        sh_sl_fig, sh_sl_ax = plt.subplots()
        if np.sum(self.shifts) > 1e-10:
            sh_from_sl = np.exp(
                self.model_sl_sh.predict(np.log(-sorted_sl).reshape(-1, 1)))
            sh_sl_ax.plot(sorted_sl, sh_from_sl, color="black")
        if groups is not None:
            for id_gr, gr in enumerate(gr_fact[0]): 
                sh_sl_ax.scatter(self.slopes[gr_fact[1] == id_gr], 
                                self.shifts[gr_fact[1] == id_gr], 
                                c=cmap(id_gr+1), 
                                marker=markers[id_gr],
                                facecolors=fillstyles[id_gr], 
                                label=gr)
            sh_sl_ax.legend()
        else:
            sh_sl_ax.scatter(self.slopes, self.shifts)
        sh_sl_ax.set_xlabel("Slope")
        sh_sl_ax.set_ylabel("Shift")
            

        # Shifts from intercepts
        sh_in_fig, sh_in_ax = plt.subplots()
        sorted_in = np.sort(self.intercepts)
        if np.sum(self.shifts) > 1e-10:
            sh_from_in = np.exp(
                self.model_in_sh.predict(np.log(sorted_in).reshape(-1, 1)))
            sh_in_ax.plot(sorted_in, sh_from_in, color="black")
        if groups is not None:
            for id_gr, gr in enumerate(gr_fact[0]): 
                sh_in_ax.scatter(self.intercepts[gr_fact[1] == id_gr], 
                                 self.shifts[gr_fact[1] == id_gr], 
                                 c=cmap(id_gr+1), 
                                 marker=markers[id_gr],
                                 facecolors=fillstyles[id_gr], 
                                 label=gr)
            sh_in_ax.legend()
        else:
            sh_in_ax.scatter(self.intercepts, self.shifts)
        sh_in_ax.set_xlabel("Intercept")
        sh_in_ax.set_ylabel("Shift")
        
        return in_sl_fig, in_sl_ax, sh_sl_fig, sh_sl_ax, sh_in_fig, sh_sl_ax
    
    # Get zipf-mandelbrot parameters
    def get_parameters(self, slope):
        intercept = self.model_sl_in.predict(
            np.array(slope).reshape(-1, 1))[0]
        if np.sum(self.shifts) > 1e-10:
            shift = np.exp(self.model_sl_sh.predict(
                np.array(np.log(-slope)).reshape(-1, 1)))[0]
        else:
            shift = np.repeat(0, len(slope))
        return slope, intercept, shift
    
    # Generate samples with defined slope
    def generate_samples(self, slope, sample_size, num_samples=1):
        _, intercept, shift = self.get_parameters(slope)
        types = np.arange(1, sample_size+1)
        estimated_freq = np.exp(np.log(types + shift)*slope + intercept)
        probabilities = estimated_freq / np.sum(estimated_freq)
        samples = np.array([np.where(
            np.random.multinomial(1, probabilities, sample_size) > 0)[1] + 1 
                            for _ in range(num_samples)])
        return samples
        
            
    
