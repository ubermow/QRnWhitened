import sys
import os
import time
import numpy as np

# ==========================================
# SWABIAN HARDWARE INTEGRATION
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
    print("[CRITICAL ERROR] TimeTagger FileReader could not be loaded. Check your paths.")
    sys.exit(1)

# ==========================================
# SPATIAL DIGITALIZATION ENGINE
# ==========================================
def extract_spatial_entropy(input_filepath, output_filepath, chunk_size=10_000_000):
    print(f"\n=== STRICT HERALDED SPATIAL DIGITALIZATION ===")
    print(f"-> Source: {input_filepath}")
    
    # --- HARDWARE & PHYSICS PARAMETERS ---
    # Adjust these channel numbers if your physical BNC cables are plugged in differently
    CHANNEL_T = 1  # Trigger (Idler Photon)
    CHANNEL_A = 2  # Signal Path A (Reflected - Bit 0)
    CHANNEL_B = 3  # Signal Path B (Transmitted - Bit 1)
    
    # Time Tagger records in picoseconds
    WINDOW_PS = 5000  # 5 ns coincidence window
    DELAY_PS = -500   # -0.5 ns electronic cable delay
    
    start_time = time.time()
    total_spatial_bits = 0
    chunks_processed = 0
    
    try:
        reader = FileReader(input_filepath)
    except Exception as e:
        print(f"[ERROR] Could not open {input_filepath}: {e}")
        return

    print("-> Commencing continuous memory-stream processing...")
    
    with open(output_filepath, 'wb') as f_out:
        while reader.hasData():
            # 1. Load a manageable chunk into RAM
            buffer = reader.getData(chunk_size)
            channels = buffer.getChannels()
            timestamps_ps = buffer.getTimestamps()
            
            # 2. Isolate chronologically ordered timestamps for each physical channel
            t_T = timestamps_ps[channels == CHANNEL_T]
            t_A = timestamps_ps[channels == CHANNEL_A] + DELAY_PS
            t_B = timestamps_ps[channels == CHANNEL_B] + DELAY_PS
            
            # Skip chunk if no triggers exist
            if len(t_T) == 0:
                continue
                
            # 3. Vectorized Coincidence Search (The Virtual Hardware Gate)
            left_A = np.searchsorted(t_A, t_T)
            right_A = np.searchsorted(t_A, t_T + WINDOW_PS)
            coinc_A = (right_A > left_A)  # True if Det A fired within [t_T, t_T + 5000]
            
            left_B = np.searchsorted(t_B, t_T)
            right_B = np.searchsorted(t_B, t_T + WINDOW_PS)
            coinc_B = (right_B > left_B)  # True if Det B fired within [t_T, t_T + 5000]
            
            # 4. Filter for STRICT Single-Photon events (Anti-Bunching)
            # Discard multi-photon events (both fired) and empty pulses (neither fired)
            strict_A = coinc_A & ~coinc_B
            strict_B = coinc_B & ~coinc_A
            
            # 5. Chronological Assembly & Bit Mapping
            valid_triggers = strict_A | strict_B
            chronological_B = strict_B[valid_triggers]
            
            # Map physical paths to logical bits: Path B -> 1, Path A -> 0
            raw_bits = np.where(chronological_B, 1, 0).astype(np.uint8)
            
            # 6. Compress and Write to SSD
            if len(raw_bits) > 0:
                packed_bytes = np.packbits(raw_bits)
                f_out.write(packed_bytes.tobytes())
                total_spatial_bits += len(raw_bits)
            
            chunks_processed += 1
            if chunks_processed % 10 == 0:
                print(f"   ... Processed {chunks_processed * (chunk_size // 1_000_000)}M raw events "
                      f"| Extracted {total_spatial_bits:,} spatial bits")

    elapsed_time = time.time() - start_time
    print(f"\n=== DIGITALIZATION COMPLETE ===")
    print(f"-> Total Spatial Bits Extracted: {total_spatial_bits:,}")
    print(f"-> Output File: {output_filepath}")
    print(f"-> Processing Time: {elapsed_time:.2f} seconds.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Point this to your actual raw Swabian file
    INPUT_FILE = "3h_raw_photons.ttbin" 
    OUTPUT_FILE = "spatial_3hraw_bitstream.bin"
    
    extract_spatial_entropy(INPUT_FILE, OUTPUT_FILE)