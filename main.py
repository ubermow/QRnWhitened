import numpy as np

# Importiamo le nostre funzioni dai moduli creati
from data_loader import generate_simulated_data, load_real_data
from toeplitz_extractor import ToeplitzExtractor

def main():
    print("=== QRNG Post-Processing Pipeline ===")

    # 1. Configuration parameters (Zhang et al.)
    N_INPUT = 1520
    M_OUTPUT = 1024
    
    # --- IL SELETTORE DI MODALITÀ ---
    # change this to False if you want to test with simulated data
    USE_REAL_DATA = True 

    # 2. Data Loading
    if USE_REAL_DATA:
        print("\n[Mode] REAL DATA: Loading from NIST Beacon file...")
        # Assuming the NIST data has already been downloaded and saved as 'nist_random_data.csv'
        raw_data = load_real_data("nist_random_data.csv")
    else:
        print("\n[Mode] SIMULATED DATA: Generating 55% bias stream...")
        TOTAL_RAW_BITS = N_INPUT * 100 
        raw_data = generate_simulated_data(num_bits=TOTAL_RAW_BITS, prob_one=0.55)

    # 3. Initialize the Extractor
    print("\n-> Initializing Toeplitz Extractor...")
    extractor = ToeplitzExtractor(n_input=N_INPUT, m_output=M_OUTPUT)
    extractor.generate_matrix() 

    # 4. The Batching Process
    print("-> Starting batch extraction...")
    pure_blocks = [] 
    
    # Calculate dynamically how many blocks we can process
    num_blocks = len(raw_data) // N_INPUT
    
    for i in range(num_blocks):
        start_idx = i * N_INPUT
        end_idx = start_idx + N_INPUT
        raw_block = raw_data[start_idx:end_idx]
        
        pure_block = extractor.extract(raw_block)
        pure_blocks.append(pure_block)

    # 5. Reassemble
    final_pure_data = np.concatenate(pure_blocks)

    # 6. Final Validation
    print("\n=== Extraction Results ===")
    print(f"Processed {num_blocks} full blocks of {N_INPUT} bits.")
    print(f"Total pure bits generated: {len(final_pure_data)}")
    
    final_ones = np.sum(final_pure_data)
    final_percentage = (final_ones / len(final_pure_data)) * 100
    
    # calculate initial bias if we are using real data
    if USE_REAL_DATA:
        raw_ones = np.sum(raw_data)
        raw_percentage = (raw_ones / len(raw_data)) * 100
        print(f"Initial raw bias : {raw_percentage:.2f}% '1's")
        
    print(f"Final pure bias  : {final_percentage:.2f}% '1's (Target: ~50.00%)")
    print("=====================================")

if __name__ == "__main__":
    main()