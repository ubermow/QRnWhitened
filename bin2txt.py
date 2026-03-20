import numpy as np
import os
import time

def convert_bin_to_ascii_for_nist(input_bin_file, output_txt_file, bits_to_extract=20_000_000):
    """
    Safely translates a raw binary file into a pure ASCII text file of 0s and 1s.
    This bypasses all Endianness/hardware reading risks in the NIST C-Suite.
    """
    print(f"\n=== NIST STS ASCII CONVERTER ===")
    print(f"-> Reading from: {input_bin_file}")
    
    if not os.path.exists(input_bin_file):
        print(f"[ERROR] File not found: {input_bin_file}")
        return

    # Calculate bytes to read (20 million bits = 2.5 MB)
    bytes_to_read = bits_to_extract // 8
    
    start_time = time.time()
    
    with open(input_bin_file, 'rb') as f_in:
        raw_bytes = np.frombuffer(f_in.read(bytes_to_read), dtype=np.uint8)
        
    # Unpack bytes into an array of 0s and 1s
    bitstream = np.unpackbits(raw_bytes)
    
    print(f"-> Successfully loaded {len(bitstream):,} bits.")
    print(f"-> Writing to pure ASCII format (This may take a moment)...")
    
    # Save the 0/1 array as a continuous string of text characters
    # fmt='%d' ensures they are written as integers without decimal points
    # newline='' ensures it is one giant continuous line, which NIST prefers
    np.savetxt(output_txt_file, [bitstream], fmt='%d', delimiter='', newline='')
    
    elapsed_time = time.time() - start_time
    print(f"-> Conversion complete in {elapsed_time:.2f} seconds.")
    print(f"-> Target file ready for NIST C-Suite: {output_txt_file}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Point this to your beautifully whitened Hybrid FFT strings!
    FILE_TEMPORAL_BIN = r"data\whitened\final_attempt\FFTw_3ht_keys.bin"
    FILE_TEMPORAL_TXT = r"FFTw_3ht_keys_NIST.txt"

    FILE_SPATIAL_BIN = r"data\whitened\final_attempt\FFTw_3hs_keys.bin"
    FILE_SPATIAL_TXT = r"FFTw_3hs_keys_NIST.txt"

    # We need exactly 20 chunks of 1,000,000 bits
    convert_bin_to_ascii_for_nist(FILE_TEMPORAL_BIN, FILE_TEMPORAL_TXT, bits_to_extract=100_000_000)
    convert_bin_to_ascii_for_nist(FILE_SPATIAL_BIN, FILE_SPATIAL_TXT, bits_to_extract=100_000_000)