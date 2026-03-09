import sys
import os
import numpy as np
import time

# ==========================================
# 1. HARDWARE BRIDGE CONFIGURATION
# ==========================================
SWABIAN_PYTHON_PATH = r"C:\Program Files\Swabian Instruments\Time Tagger\driver\x64\python3.10"
SWABIAN_ROOT_PATH = r"C:\Program Files\Swabian Instruments\Time Tagger"

if SWABIAN_PYTHON_PATH not in sys.path:
    sys.path.append(SWABIAN_PYTHON_PATH)
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(SWABIAN_ROOT_PATH)
    os.add_dll_directory(SWABIAN_PYTHON_PATH)

try:
    from TimeTagger import FileReader
except ImportError:
    print("[CRITICAL ERROR] TimeTagger FileReader could not be loaded.")
    sys.exit(1)

# ==========================================
# 2. DIGITALIZATION MODULE (PARITY METHOD)
# ==========================================
def extract_bits_from_ttbin(input_file, output_file, target_channel=2, chunk_size=10000000):
    """
    Reads the raw quantum timestamps and extracts true random bits 
    using the Least Significant Bit (LSB) Parity method.
    Saves the result as a highly compressed raw binary file (.bin).
    """
    print(f"\n=== QUANTUM DIGITALIZATION ENGINE ===")
    print(f"-> Source File: {input_file}")
    print(f"-> Target Bitstream File: {output_file}")
    
    try:
        reader = FileReader(input_file)
    except Exception as e:
        print(f"[ERROR] Could not open binary file: {e}")
        return

    total_photons_processed = 0
    total_bytes_written = 0
    
    start_time = time.time()

    # Open the output file in 'wb' (Write Binary) mode
    # This prevents Windows from adding text-formatting characters
    with open(output_file, 'wb') as f_out:
        print(f"-> Extracting Parity Bits for Channel {target_channel}...")
        
        while reader.hasData():
            buffer = reader.getData(chunk_size)
            channels = buffer.getChannels()
            
            # We explicitly request the raw PICOSECONDS (64-bit integers)
            # Do not convert to seconds here. We need the integer precision 
            # to extract the Least Significant Bit.
            timestamps_ps = buffer.getTimestamps()
            
            # 1. Isolate Alice's photons using boolean masking
            mask = (channels == target_channel)
            alice_timestamps = timestamps_ps[mask]
            
            if len(alice_timestamps) == 0:
                continue
            
            total_photons_processed += len(alice_timestamps)
            
            # 2. Extract the physical randomness (The Parity Bit)
            # The modulo 2 operation isolates the LSB (0 if even, 1 if odd)
            raw_bits = alice_timestamps % 2
            
            # 3. Compress the bits into true physical bytes
            # np.packbits takes an array like [1, 0, 0, 1, 1, 1, 0, 0] 
            # and squashes it into a single 8-bit Byte to save disk space
            packed_bytes = np.packbits(raw_bits)
            
            # 4. Stream directly to the hard drive in binary format
            f_out.write(packed_bytes.tobytes())
            
            total_bytes_written += len(packed_bytes)

    elapsed_time = time.time() - start_time
    
    print("\n=== DIGITALIZATION COMPLETE ===")
    print(f"-> Processed Photons: {total_photons_processed:,}")
    print(f"-> Generated True Random Bytes: {total_bytes_written:,} Bytes")
    print(f"-> Physical File Size on Disk: {total_bytes_written / (1024*1024):.2f} MB")
    print(f"-> Execution Time: {elapsed_time:.2f} seconds")

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Ensure this points to your 1-hour recording
    INPUT_FILENAME = "half_raw_photons.ttbin"
    
    # We save as .bin because it is a raw stream of bits, not text.
    OUTPUT_FILENAME = "quantum_bitstream.bin"
    
    extract_bits_from_ttbin(INPUT_FILENAME, OUTPUT_FILENAME, target_channel=2)