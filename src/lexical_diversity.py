
from collections import Counter
import math
from pathlib import Path
import random
import re


def main():
    """Main program."""

    # Parameters...
    subsample_len = 50
    num_subsamples = 1000
    test_data_path = Path("./data/test")
    
    # Open and read test files...
    file_name_to_content = {file_path.name: file_path.read_text(encoding="UTF-8")
                            for file_path in test_data_path.iterdir()}
    
    # Process each test file...
    for file_name, content in file_name_to_content.items():
        
        print(f"{'='*80}\nFile {file_name}")

        # Tokenize file content.
        sample = tokenize(content)
        print(f"Length: {len(sample)} tokens")
        
        # Compute and display sample entropy.
        print(f"Sample entropy: {sample_entropy(sample):.3f}")
        
        # Compute and display subsample entropy (random and window).
        print(f"Subsample entropy ({subsample_len} tokens):")
        average, stdev = subsample_entropy(sample, subsample_len)
        print(f"- random subsampling: {average:.3f}",
              f"(SD: {stdev:.3f}), n={num_subsamples}")
        average, stdev = subsample_entropy(sample, subsample_len, mode="window")
        print(f"- moving average: {average:.3f}",
              f"(SD: {stdev:.3f}), n={len(sample)-subsample_len+1}")


def tokenize(my_string, token_regex=r"\w+"):
    """Tokenize string into list of tokens based on regex describing tokens."""
    return re.findall(token_regex, my_string)
    
def sample_entropy(sample, base=2):
    """Compute sample entropy based on a list of items."""
    if len(sample) == 0:
        raise ValueError("Can't compute sample entropy with empty list.")
    return(counter_to_sample_entropy(Counter(sample), base))

def counter_to_sample_entropy(counter, base=2):
    """Compute sample entropy based on a Counter."""
    if len(counter) == 0:
        raise ValueError("Can't compute sample entropy with empty Counter.")
    my_sum = 0
    weighted_sum_of_logs = 0
    for freq in counter.values():
        if freq:
            my_sum += freq
            weighted_sum_of_logs += freq * math.log(freq, base)
    return math.log(my_sum, base) - weighted_sum_of_logs/my_sum

def subsample_entropy(sample, subsample_len, num_subsamples=1000, mode="random", 
                      base=2):
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
            entropy = sample_entropy(get_random_subsample(sample, subsample_len),
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


if __name__ == "__main__":
    main()
