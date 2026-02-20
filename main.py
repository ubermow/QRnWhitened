import numpy as np

# Import our custom modules
from data_loader import generate_simulated_data
from toeplitz_extractor import ToeplitzExtractor

def main():
    print("=== QRNG Post-Processing Pipeline ===")

    # 1. Configuration parameters
    # Using Zhang et al. parameters as our baseline
    N_INPUT = 1520
    M_OUTPUT = 1024
    
    # We want to process exactly 100 blocks to test our batching logic
    TOTAL_RAW_BITS = N_INPUT * 100 

    # 2. Load/Generate Raw Data (Modulo 1)
    raw_data = generate_simulated_data(num_bits=TOTAL_RAW_BITS, prob_one=0.55)

    # 3. Initialize the Extractor (Modulo 2)
    print("\n-> Initializing Toeplitz Extractor...")
    extractor = ToeplitzExtractor(n_input=N_INPUT, m_output=M_OUTPUT)
    
    # Generate the seed and matrix ONCE for the entire session
    extractor.generate_matrix() 

    # 4. The Batching Process (Software equivalent of FPGA FIFO)
    print("-> Starting batch extraction...")
    pure_blocks = [] # List to store our purified chunks
    
    # Calculate how many full blocks we can process
    num_blocks = len(raw_data) // N_INPUT
    
    for i in range(num_blocks):
        # Slice the array: extract exactly 1520 bits
        start_idx = i * N_INPUT
        end_idx = start_idx + N_INPUT
        raw_block = raw_data[start_idx:end_idx]
        
        # Purify the block using modulo 2 matrix multiplication
        pure_block = extractor.extract(raw_block)
        
        # Save the purified block
        pure_blocks.append(pure_block)

    # 5. Reassemble the final pure bitstream
    # np.concatenate joins all the small arrays back into one long vector
    final_pure_data = np.concatenate(pure_blocks)

    # 6. Final Validation (The Moment of Truth)
    print("\n=== Extraction Results ===")
    print(f"Processed {num_blocks} blocks of {N_INPUT} bits.")
    print(f"Total pure bits generated: {len(final_pure_data)}")
    
    final_ones = np.sum(final_pure_data)
    final_percentage = (final_ones / len(final_pure_data)) * 100
    
    print(f"Initial raw bias : 55.00% '1's")
    print(f"Final pure bias  : {final_percentage:.2f}% '1's (Target: ~50.00%)")
    print("=====================================")

if __name__ == "__main__":
    main()