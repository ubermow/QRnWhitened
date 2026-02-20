import numpy as np

def generate_simulated_data(num_bits=10000, prob_one=0.55):
    """
    Simulates raw bits in anticipation of the real experiment.
    We introduce an intentional 'bias': prob_one=0.55 means
    that 55% of '1's and 45% of '0's will be produced.
    Our Toeplitz extractor will need to handle this defect.
    """
    print(f"-> Simulating {num_bits} raw bits...")
    print(f"-> Intentional defect introduced: probability of '1' at {prob_one * 100}%")
    
    # np.random.choice chooses 0 or 1, weighting them with the probabilities we give
    simulated_data = np.random.choice([0, 1], size=num_bits, p=[1 - prob_one, prob_one])
    
    # Quick check to see if it worked
    count_ones = np.sum(simulated_data)
    actual_percentage = (count_ones / num_bits) * 100
    print(f"-> Generation completed! Actual percentage of '1's: {actual_percentage:.2f}%\n")
    
    return simulated_data

# If we run this file directly, do a small test
if __name__ == "__main__":
    bit_finti = generate_simulated_data(num_bits=50000, prob_one=0.55)
