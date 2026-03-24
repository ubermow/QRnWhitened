import time
import numpy as np
import os
import torch
import hashlib

# ==========================================
# 1. HARDWARE ACCELERATION
# ==========================================
# Automatically route the heavy FFT math to the GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. HYBRID FFT-TOEPLITZ & HASH_DRBG EXTRACTOR
# ==========================================
class HybridFFTToeplitzExtractor:
    def __init__(self, n_input, m_output):
        self.n = n_input
        self.m = m_output
        self.seed_length = self.m + self.n - 1
        
        # Calculate the next power of 2 for optimal FFT performance
        self.N_fft = 1 << (self.seed_length - 1).bit_length()
        
        # ---------------------------------------------------------
        # THE CRYPTOGRAPHIC ANCHOR
        # Pull exactly ONE 256-bit true-random seed from the OS.
        # The OS will never be queried for randomness again.
        # ---------------------------------------------------------
        self.master_seed = os.urandom(32) 
        self.drbg_counter = 0
        print("[+] Master Entropy Seed secured. Initializing Hash_DRBG expansion.")

    def _generate_batch_seeds(self, batch_size):
        """
        Uses SHA-256 to infinitely stretch the master seed into the exact 
        number of bytes required to build unique matrices for the entire batch.
        """
        bytes_needed = ((batch_size * self.seed_length) // 8) + 1
        seed_bytes = bytearray()
        
        while len(seed_bytes) < bytes_needed:
            hasher = hashlib.sha256()
            hasher.update(self.master_seed)
            hasher.update(self.drbg_counter.to_bytes(8, byteorder='big'))
            seed_bytes.extend(hasher.digest())
            self.drbg_counter += 1
            
        # Unpack the raw hash bytes into a flat bit array
        return np.unpackbits(np.frombuffer(seed_bytes, dtype=np.uint8))

    def process_batch(self, bit_blocks):
        """
        Processes hundreds of massive blocks simultaneously in the frequency domain,
        seeded entirely by the SHA-256 expanded Hash_DRBG stream.
        """
        batch_size = bit_blocks.shape[0]

        # 1. Expand the Master Seed via SHA-256 for this specific batch
        seed_pool = self._generate_batch_seeds(batch_size)

        # 2. Allocate memory for the Circulant vectors on the GPU
        c_ext = torch.zeros((batch_size, self.N_fft), dtype=torch.float64, device=DEVICE)

        # 3. Build the Circulant embedding for every block in the batch
        for i in range(batch_size):
            start_idx = i * self.seed_length
            seed = seed_pool[start_idx : start_idx + self.seed_length]

            col = torch.tensor(seed[:self.m], dtype=torch.float64, device=DEVICE)
            row = torch.tensor(seed[self.m - 1:], dtype=torch.float64, device=DEVICE)

            c_ext[i, :self.m] = col
            # The row is placed at the end of the vector in reverse to complete the Circulant shift
            c_ext[i, - (self.n - 1):] = row[1:].flip(dims=[0])

        # 4. Prepare the raw quantum data blocks
        x_ext = torch.zeros((batch_size, self.N_fft), dtype=torch.float64, device=DEVICE)
        x_ext[:, :self.n] = torch.tensor(bit_blocks, dtype=torch.float64, device=DEVICE)

        # 5. FAST FOURIER TRANSFORM (Time Domain -> Frequency Domain)
        C_fft = torch.fft.rfft(c_ext)
        X_fft = torch.fft.rfft(x_ext)

        # 6. Point-wise Multiplication (The O(N log N) shortcut)
        Y_fft = C_fft * X_fft

        # 7. INVERSE FAST FOURIER TRANSFORM (Frequency Domain -> Time Domain)
        y_ext = torch.fft.irfft(Y_fft, n=self.N_fft)

        # 8. Extract the valid m_out bits, round precisely, and apply GF(2) math
        valid_out = y_ext[:, :self.m]
        extracted_bits = torch.round(valid_out).to(torch.int64) % 2

        return extracted_bits.cpu().numpy().astype(np.uint8)

# ==========================================
# 3. MASSIVE-SCALE PIPELINE ENGINE
# ==========================================
def whiten_gigabyte_file_hybrid(input_filepath, output_filepath, n_in=131072, m_out=108660, chunk_mb=5):
    print(f"\n=== HYBRID CRYPTOGRAPHIC ENGINE (FFT + SHA-256) ===")
    print(f"-> Hardware Target: {DEVICE.type.upper()}")
    print(f"-> Source: {input_filepath}")
    
    if not os.path.exists(input_filepath):
        print(f"[CRITICAL] File missing: {input_filepath}")
        return

    file_size_bytes = os.path.getsize(input_filepath)
    print(f"-> Target Size: {file_size_bytes / (1024*1024):.2f} MB")
    
    # Initialize the Hybrid Extractor
    extractor = HybridFFTToeplitzExtractor(n_input=n_in, m_output=m_out)
    bytes_per_chunk = chunk_mb * 1024 * 1024
    
    total_processed_bits = 0
    total_whitened_bytes = 0
    bit_buffer = np.array([], dtype=np.uint8)  
    
    start_time = time.time()
    
    with open(input_filepath, 'rb') as f_in, open(output_filepath, 'wb') as f_out:
        print(f"-> Commencing continuous extraction stream (Block Size: {n_in:,} bits)...")
        
        while True:
            raw_bytes = f_in.read(bytes_per_chunk)
            if not raw_bytes:
                break
                
            chunk_array = np.frombuffer(raw_bytes, dtype=np.uint8)
            bitstream = np.unpackbits(chunk_array)
            
            num_blocks = len(bitstream) // n_in
            if num_blocks == 0:
                continue
                
            usable_bits = num_blocks * n_in
            bitstream_chopped = bitstream[:usable_bits].reshape((num_blocks, n_in))
            
            # --- THE HYBRID CRUSH ---
            whitened_blocks = extractor.process_batch(bitstream_chopped)
            whitened_flat = whitened_blocks.flatten()
            
            # --- THE BYTE-BOUNDARY FIX ---
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
    print(f"-> Output File: {output_filepath}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Point this to your RAW hardware bitstreams
    INPUT_FILE = r"data\\raw\\3hours_nopeople\\temporal_3hraw_bitstream.bin"
    OUTPUT_FILE = r"data\\whitened\\final_attempt\\FFTw_3ht_keys.bin"

    # NOTE: The block size is now massive (131,072) to crush hardware drift.
    # Be sure to recalculate m_out using your worst-case AI min-entropy:
    # m_out = int(np.floor(131072 * min_entropy) - 128)
    
    # Example below assumes an H_min of ~0.83
    whiten_gigabyte_file_hybrid(INPUT_FILE, OUTPUT_FILE, n_in=131072, m_out=122748, chunk_mb=20)