import time
import numpy as np
from scipy.linalg import toeplitz

class ToeplitzExtractor:
    def __init__(self, n_input, m_output):
        """
        Initializes the Toeplitz Strong Extractor.
        n_input: length of the raw input bit sequence (n)
        m_output: length of the extracted output bit sequence (m)
        """
        self.n = n_input
        self.m = m_output
        self.seed_length = self.m + self.n - 1
        self.matrix = None

    def generate_matrix(self, seed=None):
        """
        Generates the Toeplitz matrix based on the provided seed.
        """
        if seed is None:
            # We use an internal pseudo-random seed for the matrix.
            # In a strict QKD setup, this seed would be shared publicly between Alice and Bob.
            seed = np.random.randint(2, size=self.seed_length, dtype=np.int8)
        elif len(seed) != self.seed_length:
            raise ValueError(f"Seed must be exactly {self.seed_length} bits long.")

        first_col = seed[:self.m]
        first_row = seed[self.m - 1:]
        
        # We enforce np.int8 to save RAM during the massive matrix multiplications
        self.matrix = toeplitz(c=first_col, r=first_row).astype(np.int8)
        print(f"[ToeplitzExtractor] Generated {self.m}x{self.n} matrix.")
        
        return self.matrix

    def extract_batch(self, raw_bit_matrix):
        """
        Performs extraction on thousands of blocks simultaneously.
        raw_bit_matrix shape: (num_blocks, n_input)
        """
        if self.matrix is None:
            raise RuntimeError("Matrix not generated.")
        
        # We transpose the raw_bit_matrix to align with the Toeplitz matrix,
        # perform the dot product, transpose back, and apply Modulo 2.
        # This simulates the parallel XOR gates of an FPGA.
        extracted_matrix = np.dot(self.matrix, raw_bit_matrix.T).T % 2
        return extracted_matrix.astype(np.uint8)

# ==========================================
# FILE PROCESSING PIPELINE
# ==========================================
def whiten_quantum_file(input_filepath, output_filepath, n_in, m_out):
    print(f"\n=== QUANTUM TOEPLITZ WHITENING ===")
    print(f"-> Source: {input_filepath}")
    
    start_time = time.time()
    
    # 1. Load the raw bitstream
    try:
        raw_bytes = np.fromfile(input_filepath, dtype=np.uint8)
        bitstream = np.unpackbits(raw_bytes)
        total_raw_bits = len(bitstream)
        print(f"-> Loaded {total_raw_bits:,} raw bits.")
    except FileNotFoundError:
        print(f"[ERROR] Could not find {input_filepath}.")
        return

    # 2. Calculate Block Mathematics
    # We drop the tiny remainder of bits at the end of the file that don't fit into a full block
    num_blocks = total_raw_bits // n_in
    usable_bits = num_blocks * n_in
    
    print(f"-> Slicing into {num_blocks:,} blocks of {n_in} bits...")
    bitstream_chopped = bitstream[:usable_bits].reshape((num_blocks, n_in))

    # 3. Initialize the Extractor
    extractor = ToeplitzExtractor(n_input=n_in, m_output=m_out)
    extractor.generate_matrix()

    # 4. Perform the massive parallel matrix multiplication
    print(f"-> Crushing blocks through the matrix (Compression Ratio: {m_out/n_in:.4f})...")
    # For a 33MB file, Numpy can do this in one giant vectorized operation
    whitened_blocks = extractor.extract_batch(bitstream_chopped)

    # 5. Reassemble and Save
    print("-> Reassembling purified bitstream...")
    whitened_flat = whitened_blocks.flatten()
    
    packed_whitened_bytes = np.packbits(whitened_flat)
    
    with open(output_filepath, 'wb') as f_out:
        f_out.write(packed_whitened_bytes.tobytes())

    total_whitened_bytes = len(packed_whitened_bytes)
    
    elapsed_time = time.time() - start_time
    print(f"\n=== WHITENING COMPLETE ===")
    print(f"-> Yielded {total_whitened_bytes:,} Bytes of Cryptographic Key.")
    print(f"-> Output File: {output_filepath}")
    print(f"-> Processing Time: {elapsed_time:.2f} seconds.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    INPUT_FILE = "quantum_bitstream.bin"
    OUTPUT_FILE = "whitened_quantum_keys.bin"
    
    # We use n=2048 to process large chunks efficiently.
    # We use m=2000 based on the ML Min-Entropy (H_inf = 0.9969).
    # 2048 * 0.9969 = 2041. We use 2000 to leave a strict safety margin.
    whiten_quantum_file(INPUT_FILE, OUTPUT_FILE, n_in=2048, m_out=2000)