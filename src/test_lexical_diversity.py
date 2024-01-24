
from collections import Counter
from pathlib import Path

from LTTL.Utils import get_expected_subsample_variety

from lexical_diversity import tokenize, sample_entropy, subsample_entropy, MTLD

__version__ = 0.1
__authors__ = ["aris.xanthos@unil.ch"]

def main():
    """Main program."""

    # Parameters...
    subsample_len = 50
    num_subsamples = 1000
    test_data_path = Path("../data/test")

    # Open and read test files...
    file_name_to_content = {file_path.name: file_path.read_text(encoding="UTF-8")
                            for file_path in test_data_path.iterdir()}

    # Process each test file...
    for file_name, content in file_name_to_content.items():

        print(f"\n{'='*80}\nFile {file_name}")

        # Tokenize lower-cased file content and display length...
        sample = tokenize(content.lower())
        print(f"Length: {len(sample)} tokens")

        # Build counter and display variety...
        counter = Counter(sample)
        print(f"Variety: {len(counter)} types")

        # Compute and display sample entropy.
        print(f"Sample entropy: {sample_entropy(sample):.3f}")

        # Compute and display subsample entropy (random and window)...
        print(f"Subsample entropy ({subsample_len} tokens):")
        average, stdev = subsample_entropy(sample, subsample_len)
        print(f"- random subsampling: {average:.3f}",
              f"(SD: {stdev:.3f}, n={num_subsamples})")
        average, stdev = subsample_entropy(sample, subsample_len, mode="window")
        print(f"- moving average: {average:.3f}",
              f"(SD: {stdev:.3f}, n={len(sample)-subsample_len+1})")

        # Compute and display expected subsample variety...
        exp_variety = get_expected_subsample_variety(counter, subsample_len)
        print(f"Expected subsample variety "
              f"({subsample_len} tokens): {exp_variety:.3f}")

       # Compute and display MTLD (standard and custom threshold)...
        print(f"MTLD (std): {MTLD(sample):.3f}")
        print(f"MTLD (threshold=0.6): {MTLD(sample, 0.6):.3f}")

if __name__ == "__main__":
    main()
