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
    
    CHANNEL_T = 1  
    CHANNEL_A = 2  
    CHANNEL_B = 3  
    
    WINDOW_PS = 5000  
    DELAY_PS = -500   
    DEAD_TIME_PS = 45000 # 45 ns global recovery window
    
    start_time = time.time()
    total_spatial_bits = 0
    chunks_processed = 0
    
    last_accepted_t = None
    bit_buffer = np.array([], dtype=np.uint8)
    
    try:
        reader = FileReader(input_filepath)
    except Exception as e:
        print(f"[ERROR] Could not open {input_filepath}: {e}")
        return

    print("-> Commencing continuous memory-stream processing...")
    
    with open(output_filepath, 'wb') as f_out:
        while reader.hasData():
            buffer = reader.getData(chunk_size)
            channels = buffer.getChannels()
            timestamps_ps = buffer.getTimestamps()
            
            t_T = timestamps_ps[channels == CHANNEL_T]
            t_A = timestamps_ps[channels == CHANNEL_A] + DELAY_PS
            t_B = timestamps_ps[channels == CHANNEL_B] + DELAY_PS
            
            if len(t_T) == 0:
                continue
                
            left_A = np.searchsorted(t_A, t_T)
            right_A = np.searchsorted(t_A, t_T + WINDOW_PS)
            coinc_A = (right_A > left_A)  
            
            left_B = np.searchsorted(t_B, t_T)
            right_B = np.searchsorted(t_B, t_T + WINDOW_PS)
            coinc_B = (right_B > left_B)  
            
            strict_A = coinc_A & ~coinc_B
            strict_B = coinc_B & ~coinc_A
            
            valid_triggers = strict_A | strict_B
            valid_t_T = t_T[valid_triggers]
            raw_bits = np.where(strict_B[valid_triggers], 1, 0).astype(np.uint8)
            
            # --- THE GLOBAL DEAD-TIME FILTER ---
            accepted_bits = []
            for i in range(len(valid_t_T)):
                current_t = valid_t_T[i]
                # Enforce the 45ns global hold-off to prevent spatial Markov biasing
                if last_accepted_t is None or (current_t - last_accepted_t) >= DEAD_TIME_PS:
                    accepted_bits.append(raw_bits[i])
                    last_accepted_t = current_t
            
            if not accepted_bits:
                continue
                
            accepted_bits_arr = np.array(accepted_bits, dtype=np.uint8)
            
            # --- THE BYTE-BOUNDARY BUFFER FIX ---
            combined_bits = np.concatenate((bit_buffer, accepted_bits_arr))
            safe_length = (len(combined_bits) // 8) * 8
            bits_to_pack = combined_bits[:safe_length]
            bit_buffer = combined_bits[safe_length:]
            
            if len(bits_to_pack) > 0:
                packed_bytes = np.packbits(bits_to_pack)
                f_out.write(packed_bytes.tobytes())
                total_spatial_bits += len(bits_to_pack)
            
            chunks_processed += 1
            if chunks_processed % 10 == 0:
                print(f"   ... Processed {chunks_processed * (chunk_size // 1_000_000)}M raw events "
                      f"| Extracted {total_spatial_bits:,} unbiased spatial bits", end='\r')

    elapsed_time = time.time() - start_time
    print(f"\n\n=== DIGITALIZATION COMPLETE ===")
    print(f"-> Total Spatial Bits Extracted: {total_spatial_bits:,}")
    print(f"-> Output File: {output_filepath}")
    print(f"-> Processing Time: {elapsed_time:.2f} seconds.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    INPUT_FILE = r"data\\raw\\3hours_nopeople\\3h_raw_photons.ttbin"
    OUTPUT_FILE = r"spatial_3hraw_bitstream.bin"
    
    extract_spatial_entropy(INPUT_FILE, OUTPUT_FILE)