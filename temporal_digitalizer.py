import sys
import os
import time
import numpy as np

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
# 2. HERALDED TEMPORAL DIGITALIZATION (WANG METHOD)
# ==========================================
def extract_temporal_entropy_merged(input_file, output_file, chunk_size=10_000_000):
    """
    Extracts true random bits from the inter-arrival times (tau) of 
    STRICTLY heralded single photons across BOTH output paths.
    """
    print(f"\n=== MERGED TEMPORAL DIGITALIZATION ENGINE ===")
    print(f"-> Source File: {input_file}")
    
    # --- HARDWARE & PHYSICS PARAMETERS ---
    CHANNEL_T = 1  # Trigger (Idler)
    CHANNEL_A = 2  # Signal Path A (Reflected)
    CHANNEL_B = 3  # Signal Path B (Transmitted)
    WINDOW_PS = 5000  # 5 ns coincidence window
    DELAY_PS = -500   # -0.5 ns electronic delay
    
    try:
        reader = FileReader(input_file)
    except Exception as e:
        print(f"[ERROR] Could not open binary file: {e}")
        return

    total_valid_photons = 0
    total_bytes_written = 0
    chunks_processed = 0
    last_timestamp = None  # To carry over the delta calculation between chunks
    
    start_time = time.time()

    with open(output_file, 'wb') as f_out:
        print(f"-> Extracting Merged Inter-Arrival Parity Bits...")
        
        while reader.hasData():
            buffer = reader.getData(chunk_size)
            channels = buffer.getChannels()
            timestamps_ps = buffer.getTimestamps()
            
            # 1. Isolate the channels
            t_T = timestamps_ps[channels == CHANNEL_T]
            t_A = timestamps_ps[channels == CHANNEL_A] + DELAY_PS
            t_B = timestamps_ps[channels == CHANNEL_B] + DELAY_PS
            
            if len(t_T) == 0:
                continue
                
            # 2. Vectorized Coincidence Search (Virtual Hardware Gate)
            left_A = np.searchsorted(t_A, t_T)
            right_A = np.searchsorted(t_A, t_T + WINDOW_PS)
            coinc_A = (right_A > left_A) 
            
            left_B = np.searchsorted(t_B, t_T)
            right_B = np.searchsorted(t_B, t_T + WINDOW_PS)
            coinc_B = (right_B > left_B) 
            
            # 3. Filter for STRICT Single-Photon events (Anti-Bunching)
            strict_A = coinc_A & ~coinc_B
            strict_B = coinc_B & ~coinc_A
            
            # 4. Extract valid physical timestamps of the triggers
            valid_t_A = t_T[strict_A]
            valid_t_B = t_T[strict_B]
            
            # 5. THE WANG METHOD: Merge and chronologically sort
            merged_valid_timestamps = np.sort(np.concatenate((valid_t_A, valid_t_B)))
            
            if len(merged_valid_timestamps) == 0:
                continue
                
            total_valid_photons += len(merged_valid_timestamps)
            
            # 6. Cross-Chunk Boundary Handling
            if last_timestamp is not None:
                # Prepend the last event of the previous chunk to get the boundary tau
                processing_array = np.insert(merged_valid_timestamps, 0, last_timestamp)
            else:
                processing_array = merged_valid_timestamps
                
            # Save the last timestamp for the next chunk
            last_timestamp = merged_valid_timestamps[-1]

            # Need at least 2 photons to make a delta
            if len(processing_array) < 2:
                continue
            
            # 7. Calculate Inter-Arrival Times (tau_k = t_{k+1} - t_k)
            tau_intervals = np.diff(processing_array)
            
            # 8. Extract the physical randomness (LSB Parity Bit)
            raw_bits = (tau_intervals % 2).astype(np.uint8)
            
            # 9. Compress and Write
            packed_bytes = np.packbits(raw_bits)
            f_out.write(packed_bytes.tobytes())
            total_bytes_written += len(packed_bytes)
            
            chunks_processed += 1
            if chunks_processed % 10 == 0:
                print(f"   ... Processed {chunks_processed * (chunk_size // 1_000_000)}M raw events "
                      f"| Yielded {(total_bytes_written * 8):,} temporal bits")

    elapsed_time = time.time() - start_time
    
    print("\n=== DIGITALIZATION COMPLETE ===")
    print(f"-> Total Heralded Photons Merged: {total_valid_photons:,}")
    print(f"-> Total Temporal Bits Extracted: {(total_bytes_written * 8):,}")
    print(f"-> Physical File Size on Disk: {total_bytes_written / (1024*1024):.2f} MB")
    print(f"-> Execution Time: {elapsed_time:.2f} seconds")

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    INPUT_FILENAME = "data\\raw\\3hours_nopeople\\3h_raw_photons.ttbin"  
    OUTPUT_FILENAME = "temporal_merged_3hraw_bitstream.bin"
    
    extract_temporal_entropy_merged(INPUT_FILENAME, OUTPUT_FILENAME)