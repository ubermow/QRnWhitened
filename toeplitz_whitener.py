import time
import numpy as np
import os
import hashlib
from scipy.linalg import toeplitz

# ==========================================
# TRULY SECURE TOEPLITZ EXTRACTOR (SHA-256 EXPANSION)
# ==========================================
class StreamingToeplitzExtractor:
    def __init__(self, n_input, m_output):
        self.n = n_input
        self.m = m_output
        self.seed_length = self.m + self.n - 1
        
        # Pull ONE true quantum-level seed from the OS at startup
        self.master_seed = os.urandom(32) 
        self.block_counter = 0

    def generate_unique_seed(self):
        """
        Uses SHA-256 to expand the master seed into unique matrix dimensions.
        Requires ZERO additional calls to the OS entropy pool.
        """
        seed_bytes = bytearray()
        bytes_needed = (self.seed_length // 8) + 1
        
        # Cryptographically stretch the master seed using a rolling counter
        while len(seed_bytes) < bytes_needed:
            # Hash(Master_Seed + Counter) guarantees a unique output every time
            hasher = hashlib.sha256()
            hasher.update(self.master_seed)
            hasher.update(self.block_counter.to_bytes(8, byteorder='big'))
            
            seed_bytes.extend(hasher.digest())
            self.block_counter += 1
            
        # Unpack the raw hash bytes into a binary array of 0s and 1s
        seed_bits = np.unpackbits(np.frombuffer(seed_bytes, dtype=np.uint8))
        return seed_bits[:self.seed_length].astype(np.uint8)

    def process_chunk(self, raw_bit_chunk):
        """
        Processes the chunk by generating a unique matrix for every block via SHA-256.
        """
        num_blocks = raw_bit_chunk.shape[0]
        extracted_matrix = np.zeros((num_blocks, self.m), dtype=np.uint8)
        
        for i in range(num_blocks):
            # Generate a cryptographically unique seed without hitting os.urandom
            seed = self.generate_unique_seed()
            
            first_col = seed[:self.m]
            first_row = seed[self.m - 1:]
            matrix = toeplitz(c=first_col, r=first_row).astype(np.uint32)
            
            block = raw_bit_chunk[i].astype(np.uint32)
            extracted_matrix[i] = np.dot(block, matrix.T) % 2
            
        return extracted_matrix

# ==========================================
# STREAMING PIPELINE ENGINE
# ==========================================
def whiten_gigabyte_file(input_filepath, output_filepath, n_in=2048, m_out=1702, chunk_mb=1):
    print(f"\n=== CRYPTOGRAPHIC TOEPLITZ ENGINE (SHA-256 MODE) ===")
    print(f"-> Source: {input_filepath}")
    
    if not os.path.exists(input_filepath):
        print(f"[CRITICAL] File missing: {input_filepath}")
        return

    file_size_bytes = os.path.getsize(input_filepath)
    print(f"-> Target Size: {file_size_bytes / (1024*1024):.2f} MB")
    
    extractor = StreamingToeplitzExtractor(n_input=n_in, m_output=m_out)
    bytes_per_chunk = chunk_mb * 1024 * 1024
    
    total_processed_bits = 0
    total_whitened_bytes = 0
    bit_buffer = np.array([], dtype=np.uint8)  
    
    start_time = time.time()
    
    with open(input_filepath, 'rb') as f_in, open(output_filepath, 'wb') as f_out:
        while True:
            raw_bytes = f_in.read(bytes_per_chunk)
            if not raw_bytes:
                break
                
            chunk_array = np.frombuffer(raw_bytes, dtype=np.uint8)
            bitstream = np.unpackbits(chunk_array)
            
            num_blocks = len(bitstream) // n_in
            if num_blocks == 0: continue
                
            usable_bits = num_blocks * n_in
            bitstream_chopped = bitstream[:usable_bits].reshape((num_blocks, n_in))
            
            whitened_blocks = extractor.process_chunk(bitstream_chopped)
            whitened_flat = whitened_blocks.flatten()
            
            combined_bits = np.concatenate((bit_buffer, whitened_flat))
            safe_length = (len(combined_bits) // 8) * 8
            bits_to_pack = combined_bits[:safe_length]
            bit_buffer = combined_bits[safe_length:]
            
            if len(bits_to_pack) > 0:
                packed_whitened_bytes = np.packbits(bits_to_pack)
                f_out.write(packed_whitened_bytes.tobytes())
                total_whitened_bytes += len(packed_whitened_bytes)
            
            total_processed_bits += usable_bits
            print(f"   ... Processed {total_processed_bits / 1_000_000:.2f} M bits | "
                  f"Extracted {total_whitened_bytes / (1024*1024):.2f} MB", end='\r')

    elapsed_time = time.time() - start_time
    print(f"\n\n=== STREAMING COMPLETE ===")
    print(f"-> Final Yield: {total_whitened_bytes / (1024*1024):.2f} MB of Cryptographic Key.")
    print(f"-> Compression Ratio: {m_out / n_in:.4f}")
    print(f"-> Execution Time: {elapsed_time:.2f} seconds.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Point this to your physically correct RAW files
    INPUT_FILE = r"data\\raw\\3hours_nopeople\\temporal_3hraw_bitstream.bin"
    OUTPUT_FILE = r"data\\whitened\\final_attempt\\pure_3ht_keys.bin"

    whiten_gigabyte_file(INPUT_FILE, OUTPUT_FILE, n_in=2048, m_out=1918, chunk_mb=1)