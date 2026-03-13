import time
import numpy as np
import os
from scipy.linalg import toeplitz

# ==========================================
# 1. INDUSTRIAL TOEPLITZ EXTRACTOR
# ==========================================
class StreamingToeplitzExtractor:
    def __init__(self, n_input, m_output):
        """
        Initializes a RAM-efficient, high-speed Toeplitz Extractor.
        n_input: length of raw block (n)
        m_output: length of output block (m)
        """
        self.n = n_input
        self.m = m_output
        self.seed_length = self.m + self.n - 1
        self.matrix = None

    def generate_matrix(self, seed=None):
        if seed is None:
            # Deterministic pseudo-random seed for reproducibility during testing
            np.random.seed(42) 
            seed = np.random.randint(2, size=self.seed_length, dtype=np.int8)
        elif len(seed) != self.seed_length:
            raise ValueError(f"Seed must be exactly {self.seed_length} bits long.")

        first_col = seed[:self.m]
        first_row = seed[self.m - 1:]
        
        self.matrix = toeplitz(c=first_col, r=first_row).astype(np.int8)
        print(f"[Extractor] Generated {self.m}x{self.n} Toeplitz Matrix.")
        return self.matrix

    def process_chunk(self, raw_bit_chunk):
        """
        Processes a single block of RAM-safe data using vectorized BLAS operations.
        """
        # np.dot is extremely fast here because it calls underlying C-level BLAS routines.
        extracted_matrix = np.dot(raw_bit_chunk, self.matrix.T) % 2
        return extracted_matrix.astype(np.uint8)

# ==========================================
# 2. STREAMING PIPELINE ENGINE
# ==========================================
def whiten_gigabyte_file(input_filepath, output_filepath, n_in=2048, m_out=1920, chunk_mb=50):
    """
    Reads huge files in chunks to prevent RAM explosion.
    """
    print(f"\n=== QUANTUM TOEPLITZ STREAMING ENGINE ===")
    print(f"-> Source: {input_filepath}")
    
    if not os.path.exists(input_filepath):
        print(f"[CRITICAL] File missing: {input_filepath}")
        return

    file_size_bytes = os.path.getsize(input_filepath)
    print(f"-> Target Size: {file_size_bytes / (1024*1024):.2f} MB")
    
    extractor = StreamingToeplitzExtractor(n_input=n_in, m_output=m_out)
    extractor.generate_matrix()
    
    # Calculate bytes per chunk
    bytes_per_chunk = chunk_mb * 1024 * 1024
    total_processed_bits = 0
    total_whitened_bytes = 0
    
    start_time = time.time()
    
    # Open both files: Read raw in binary, write whitened in binary appending
    with open(input_filepath, 'rb') as f_in, open(output_filepath, 'wb') as f_out:
        print(f"-> Commencing stream (Chunk size: {chunk_mb} MB)...")
        
        while True:
            # Read a safe chunk of bytes
            raw_bytes = f_in.read(bytes_per_chunk)
            if not raw_bytes:
                break  # EOF Reached
                
            chunk_array = np.frombuffer(raw_bytes, dtype=np.uint8)
            bitstream = np.unpackbits(chunk_array)
            
            # Drop remainder bits that don't fit into a perfect block
            num_blocks = len(bitstream) // n_in
            if num_blocks == 0:
                continue
                
            usable_bits = num_blocks * n_in
            bitstream_chopped = bitstream[:usable_bits].reshape((num_blocks, n_in))
            
            # 1. Crush blocks through the Toeplitz matrix
            whitened_blocks = extractor.process_chunk(bitstream_chopped)
            
            # 2. Flatten, pack back into bytes, and stream to SSD
            whitened_flat = whitened_blocks.flatten()
            packed_whitened_bytes = np.packbits(whitened_flat)
            f_out.write(packed_whitened_bytes.tobytes())
            
            # Metrics update
            total_processed_bits += usable_bits
            total_whitened_bytes += len(packed_whitened_bytes)
            
            print(f"   ... Processed {total_processed_bits / 1_000_000:.2f} M bits | "
                  f"Extracted {total_whitened_bytes / (1024*1024):.2f} MB", end='\r')

    elapsed_time = time.time() - start_time
    print(f"\n\n=== STREAMING COMPLETE ===")
    print(f"-> Final Yield: {total_whitened_bytes / (1024*1024):.2f} MB of Cryptographic Key.")
    print(f"-> Compression Ratio: {m_out / n_in:.4f}")
    print(f"-> Execution Time: {elapsed_time:.2f} seconds.")
    print(f"-> Output File: {output_filepath}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Point this to your new 16MB file
    INPUT_FILE = r"data\\raw\\3hours_nopeople\\spatial_3hraw_bitstream.bin"
    OUTPUT_FILE = r"pure_3hs_keys.bin"

    whiten_gigabyte_file(INPUT_FILE, OUTPUT_FILE, n_in=2048, m_out=1702, chunk_mb=50)