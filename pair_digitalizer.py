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
# 2. HERALDED TEMPORAL DIGITALIZATION 
# ==========================================
def extract_temporal_entropy(input_file, output_file, chunk_size=10_000_000):
    """
    Extracts true random bits from the inter-arrival times (tau) of 
    STRICTLY heralded single photons, isolating SPDC Poissonian entropy.
    """
    print(f"\n=== HERALDED TEMPORAL DIGITALIZATION ENGINE ===")
    print(f"-> Source File: {input_file}")
    
    # --- HARDWARE & PHYSICS PARAMETERS ---
    CHANNEL_T = 1  # Trigger (Idler)
    CHANNEL_S = 2  # Signal (Alice)
    WINDOW_PS = 5000  # 5 ns coincidence window
    DELAY_PS = -500   # -0.5 ns electronic delay
    
    try:
        reader = FileReader(input_file)
    except Exception as e:
        print(f"[ERROR] Could not open binary file: {e}")
        return

    total_valid_photons = 0
    total_bytes_written = 0
    start_time = time.time()

    with open(output_file, 'wb') as f_out:
        print(f"-> Extracting Inter-Arrival Parity Bits...")
        
        while reader.hasData():
            buffer = reader.getData(chunk_size)
            channels = buffer.getChannels()
            timestamps_ps = buffer.getTimestamps()
            
            # 1. Isolate the channels
            t_T = timestamps_ps[channels == CHANNEL_T]
            t_S = timestamps_ps[channels == CHANNEL_S] + DELAY_PS
            
            if len(t_T) == 0 or len(t_S) == 0:
                continue
                
            # 2. Vectorized Coincidence Search (Heralding)
            # Find which Trigger photons have a corresponding Signal photon in the window
            left_S = np.searchsorted(t_S, t_T)
            right_S = np.searchsorted(t_S, t_T + WINDOW_PS)
            
            # Boolean array: True if this specific Trigger yielded a coincidence
            is_coincidence = (right_S > left_S) 
            
            # Isolate the exact timestamps of the valid, heralded trigger events
            valid_timestamps = t_T[is_coincidence]
            
            if len(valid_timestamps) < 2:
                continue
            
            total_valid_photons += len(valid_timestamps)
            
            # 3. Calculate Inter-Arrival Times (tau)
            # tau_k = t_{k+1} - t_k
            tau_intervals = np.diff(valid_timestamps)
            
            # 4. Extract the physical randomness (The LSB Parity Bit)
            # This perfectly maps to your LaTeX equation: b_temporal = tau % 2
            raw_bits = (tau_intervals % 2).astype(np.uint8)
            
            # 5. Compress and Write
            packed_bytes = np.packbits(raw_bits)
            f_out.write(packed_bytes.tobytes())
            total_bytes_written += len(packed_bytes)

    elapsed_time = time.time() - start_time
    
    print("\n=== DIGITALIZATION COMPLETE ===")
    print(f"-> Total Heralded Photons: {total_valid_photons:,}")
    print(f"-> Generated True Random Bytes: {total_bytes_written:,} Bytes")
    print(f"-> Physical File Size on Disk: {total_bytes_written / (1024*1024):.2f} MB")
    print(f"-> Execution Time: {elapsed_time:.2f} seconds")

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    INPUT_FILENAME = "3h_raw_photons.ttbin"  # Ensure this matches your recorded file
    OUTPUT_FILENAME = "time_3hraw_bitstream.bin"
    
    extract_temporal_entropy(INPUT_FILENAME, OUTPUT_FILENAME)