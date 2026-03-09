import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, chisquare, kstest, uniform

# ==========================================
# 1. HARDWARE BRIDGE CONFIGURATION
# ==========================================
# Injecting the Swabian C++ passports. Without these, 
# the FileReader will refuse to parse the binary .ttbin file.
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
# 2. DATA LOADING MODULE (.TTBIN BINARY)
# ==========================================
def load_ttbin_data(file_path, target_channel=2, chunk_size=10000000):
    """
    Reads the proprietary Swabian .ttbin binary file and isolates 
    photons from a specific channel. Converts picoseconds to seconds.
    """
    print(f"\n=== QUANTUM DATA LOADER ===")
    print(f"-> Opening binary stream from: {file_path}")
    
    try:
        reader = FileReader(file_path)
    except Exception as e:
        print(f"[ERROR] Could not open file. Details: {e}")
        return np.array([])

    target_timestamps = []
    total_events_read = 0

    print(f"-> Extracting photons for Channel {target_channel}...")
    
    # We read in chunks to prevent RAM overflow
    while reader.hasData():
        # getData returns a TimeTagStreamBuffer object, not a tuple!
        buffer = reader.getData(chunk_size)
        
        # Explicitly extract the numpy arrays from the buffer object
        channels = buffer.getChannels()
        timestamps_ps = buffer.getTimestamps()
        
        total_events_read += len(channels)
        
        # Logical mask: Keep ONLY the photons from our target detector (Alice)
        mask = (channels == target_channel)
        
        # Apply the mask and convert timestamps from picoseconds to seconds
        filtered_seconds = timestamps_ps[mask] * 1e-12
        target_timestamps.extend(filtered_seconds)

    # Convert the collected list back into a high-performance NumPy array
    final_array = np.array(target_timestamps)
    
    print(f"-> Total events scanned across all channels: {total_events_read}")
    print(f"-> Usable photons isolated on Channel {target_channel}: {len(final_array)}")
    
    if len(final_array) > 0:
        duration = final_array[-1] - final_array[0]
        avg_rate = len(final_array) / duration
        print(f"-> Average Count Rate (Channel {target_channel}): {avg_rate / 1000:.2f} kHz")

    return final_array

# ==========================================
# 3. STATISTICAL VALIDATION MODULES
# ==========================================
def test_poissonian_statistics(timestamps, time_window=0.01):
    """
    MACROSCOPIC TEST: Verifies if the photon emission follows a Poisson distribution.
    """
    print(f"\n=== MACRO TEST: Poissonian Statistics (Window: {time_window*1000} ms) ===")
    
    max_time = timestamps[-1]
    bins = np.arange(0, max_time + time_window, time_window)
    counts, _ = np.histogram(timestamps, bins=bins)
    
    lambda_val = np.mean(counts)
    print(f"-> Average counts per window (Lambda): {lambda_val:.2f}")
    
    max_count = int(np.max(counts))
    bin_edges = np.arange(-0.5, max_count + 1.5, 1.0)
    x_values = np.arange(0, max_count + 1)
    
    plt.figure(figsize=(10, 6))
    observed_freq, _, _ = plt.hist(counts, bins=bin_edges, density=True, 
                                   alpha=0.6, color='blue', label='Experimental Data')
    
    theoretical_pmf = poisson.pmf(x_values, lambda_val)
    plt.plot(x_values, theoretical_pmf, 'r--', linewidth=2, 
             label=fr'Theoretical Poisson ($\lambda$={lambda_val:.2f})')
    
    plt.title(f"Macroscopic Photon Statistics (Bin Window: {time_window*1000} ms)")
    plt.xlabel("Number of Photons Detected per Window")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    valid_indices = theoretical_pmf > 0
    obs = observed_freq[valid_indices]
    exp = theoretical_pmf[valid_indices]
    exp = exp * (np.sum(obs) / np.sum(exp))
    
    chi2_stat, p_val = chisquare(f_obs=obs, f_exp=exp)
    
    print(f"-> Chi-Square Test p-value: {p_val:.4e}")
    if p_val > 0.05:
        print("-> RESULT: PASSED. Distribution is consistent with Poisson.")
    else:
        print("-> RESULT: WARNING. Deviation from perfect Poisson detected.")

def test_uniform_approximation(timestamps, micro_bin_width=20e-9):
    """
    MICROSCOPIC TEST: Verifies the irreducible randomness of the photon arrival times.
    """
    print(f"\n=== MICRO TEST: Uniform Approximation (Cycle: {micro_bin_width*1e9} ns) ===")
    
    time_mod = np.mod(timestamps, micro_bin_width)
    normalized_times = time_mod / micro_bin_width
    
    plt.figure(figsize=(10, 6))
    plt.hist(normalized_times, bins=50, density=True, alpha=0.6, color='green', label='Experimental Micro-Times')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Ideal Uniform Distribution')
    
    plt.title(f"Microscopic Arrival Distribution (Modulo {micro_bin_width*1e9} ns)")
    plt.xlabel("Normalized Arrival Time (0 to 1)")
    plt.ylabel("Density")
    plt.ylim(0, 1.5) 
    plt.legend()
    plt.grid(alpha=0.3)
    
    ks_stat, p_val = kstest(normalized_times, 'uniform')
    
    print(f"-> Kolmogorov-Smirnov Test p-value: {p_val:.4e}")
    if p_val > 0.05:
        print("-> RESULT: PASSED. Distribution is statistically Uniform.")
    else:
        print("-> RESULT: WARNING. Slight deviation from Uniform detected.")

def test_system_stationarity(timestamps, chunk_duration_sec=10, micro_bin_width=20e-9):
    """
    STABILITY TEST: Slices the dataset into chunks to ensure the laser/crystal 
    did not drift or malfunction during the acquisition.
    """
    print(f"\n=== STATIONARITY TEST: Evaluating repeatability over chunks of {chunk_duration_sec}s ===")
    
    max_time = timestamps[-1]
    num_chunks = int(max_time // chunk_duration_sec)
    
    if num_chunks < 3:
        print("[WARNING] Dataset is too short (less than 3 chunks) to plot stationarity.")
        return

    chunk_times = []
    lambda_trends = []
    p_value_trends = []
    
    print(f"-> Slicing dataset into {num_chunks} independent operational blocks...")

    for i in range(num_chunks):
        start_time = i * chunk_duration_sec
        end_time = start_time + chunk_duration_sec
        
        mask = (timestamps >= start_time) & (timestamps < end_time)
        chunk_data = timestamps[mask]
        
        if len(chunk_data) < 100:
            continue 
            
        counts, _ = np.histogram(chunk_data, bins=np.arange(start_time, end_time, 0.01))
        lambda_val = np.mean(counts)
        
        time_mod = np.mod(chunk_data, micro_bin_width)
        normalized_times = time_mod / micro_bin_width
        _, p_val = kstest(normalized_times, 'uniform')
        
        chunk_times.append((start_time / 60)) 
        lambda_trends.append(lambda_val)
        p_value_trends.append(p_val)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(chunk_times, lambda_trends, marker='o', color='blue', linestyle='-')
    ax1.axhline(np.mean(lambda_trends), color='r', linestyle='--', label='Average Rate')
    ax1.set_ylabel(r"Photon Rate ($\lambda$)")
    ax1.set_title(f"QRNG Hardware Stability Over Time (Chunk: {chunk_duration_sec}s)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(chunk_times, p_value_trends, marker='s', color='green', linestyle='-')
    ax2.axhline(0.05, color='red', linestyle='-', linewidth=2, label='Failure Threshold (p=0.05)')
    ax2.set_ylabel("Uniformity p-value")
    ax2.set_xlabel("Elapsed Time (Minutes)")
    ax2.set_yscale('log') 
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()

# ==========================================
# 4. MAIN EXECUTION BLOCK (REAL DATA)
# ==========================================
if __name__ == "__main__":
    # Ensure this matches the exact name of the file you recorded
    FILENAME = "half_raw_photons.ttbin" 
    
    #P Channel 2 is typically Alice's detector arm.
    timestamps = load_ttbin_data(FILENAME, target_channel=2)
    
    if len(timestamps) > 0:
        print("\n-> Engaging Statistical Physics Validators...")
        
        # 1. Macro Test: Poisson distribution at 10 milliseconds
        test_poissonian_statistics(timestamps, time_window=0.01) 
        
        # 2. Micro Test: Uniformity at a 20 nanoseconds scale
        test_uniform_approximation(timestamps, micro_bin_width=20e-9) 
        
        # 3. Stationarity Test: 10-second blocks for a 60-second file
        test_system_stationarity(timestamps, chunk_duration_sec=60) 
        
        print("\n-> Rendering all plots. Close the windows to exit the script.")
        plt.show()
    else:
        print("\n[ERROR] No data extracted or file is empty. Pipeline halted.")