import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, poisson, chisquare, kstest

# ==========================================
# DELOITTE CORPORATE AESTHETIC CONFIGURATION
# ==========================================
DELOITTE_GREEN = '#86BC25'
DELOITTE_BLACK = '#000000'
DELOITTE_GREY  = '#53565A'
DELOITTE_LIGHT = '#D0D0CE'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': DELOITTE_GREY,
    'axes.linewidth': 1.0,
    'xtick.color': DELOITTE_GREY,
    'ytick.color': DELOITTE_GREY,
    'text.color': DELOITTE_BLACK,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white'
})

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
# 2. DATA EXTRACTION MODULES (.TTBIN)
# ==========================================
def load_ttbin_data(file_path, target_channel=2, chunk_size=10_000_000):
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
    
    while reader.hasData():
        buffer = reader.getData(chunk_size)
        channels = buffer.getChannels()
        timestamps_ps = buffer.getTimestamps()
        
        total_events_read += len(channels)
        mask = (channels == target_channel)
        filtered_seconds = timestamps_ps[mask] * 1e-12
        target_timestamps.extend(filtered_seconds)

    final_array = np.array(target_timestamps)
    
    print(f"-> Total events scanned across all channels: {total_events_read:,}")
    print(f"-> Usable photons isolated on Channel {target_channel}: {len(final_array):,}")
    
    if len(final_array) > 0:
        duration = final_array[-1] - final_array[0]
        avg_rate = len(final_array) / duration
        print(f"-> Average Count Rate (Channel {target_channel}): {avg_rate / 1000:.2f} kHz")

    return final_array

def extract_heralded_tau(file_path, chunk_size=20_000_000):
    """
    Reads the .ttbin file, applies strict coincidence gating, 
    and returns an array of inter-arrival times (tau) in seconds.
    """
    print(f"\n=== QUANTUM PHYSICS VALIDATOR (COINCIDENCES) ===")
    print(f"-> Source: {file_path}")
    
    CHANNEL_T = 1  # Trigger
    CHANNEL_S = 2  # Signal
    WINDOW_PS = 5000  # 5 ns
    DELAY_PS = -500   # -0.5 ns
    
    try:
        reader = FileReader(file_path)
    except Exception as e:
        print(f"[ERROR] Could not open file: {e}")
        return np.array([])

    all_tau_ps = []
    print("-> Applying 5ns Coincidence Gate and extracting tau...")
    
    while reader.hasData():
        buffer = reader.getData(chunk_size)
        channels = buffer.getChannels()
        timestamps_ps = buffer.getTimestamps()
        
        t_T = timestamps_ps[channels == CHANNEL_T]
        t_S = timestamps_ps[channels == CHANNEL_S] + DELAY_PS
        
        if len(t_T) == 0 or len(t_S) == 0:
            continue
            
        left_S = np.searchsorted(t_S, t_T)
        right_S = np.searchsorted(t_S, t_T + WINDOW_PS)
        is_coincidence = (right_S > left_S) 
        
        valid_timestamps = t_T[is_coincidence]
        
        if len(valid_timestamps) > 1:
            tau_ps = np.diff(valid_timestamps)
            tau_ps = tau_ps[tau_ps > 50_000] # Filter dead-time artifacts
            all_tau_ps.extend(tau_ps)

    tau_seconds = np.array(all_tau_ps) * 1e-12
    
    print(f"-> Total valid heralded intervals extracted: {len(tau_seconds):,}")
    if len(tau_seconds) > 0:
        avg_rate = 1.0 / np.mean(tau_seconds)
        print(f"-> Average Heralded Pair Rate: {avg_rate / 1000:.2f} kHz")

    return tau_seconds

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
                                   alpha=0.6, color=DELOITTE_GREY, label='Experimental Data')
    
    theoretical_pmf = poisson.pmf(x_values, lambda_val)
    plt.plot(x_values, theoretical_pmf, color=DELOITTE_GREEN, linestyle='--', linewidth=2, 
             label=fr'Theoretical Poisson ($\lambda$={lambda_val:.2f})')
    
    plt.title(f"Macroscopic Photon Statistics (Bin Window: {time_window*1000} ms)")
    plt.xlabel("Number of Photons Detected per Window")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    valid_indices = theoretical_pmf > 0
    obs = observed_freq[valid_indices]
    exp = theoretical_pmf[valid_indices]
    
    if np.sum(exp) > 0:
        exp = exp * (np.sum(obs) / np.sum(exp))
        chi2_stat, p_val = chisquare(f_obs=obs, f_exp=exp)
        print(f"-> Chi-Square Test p-value: {p_val:.4e}")
        
        if p_val > 0.01:
            print("-> RESULT: PASSED. Distribution is consistent with Poisson.")
        else:
            print("-> RESULT: WARNING. Deviation from perfect Poisson detected.")
    else:
        print("-> RESULT: ERROR. Expected frequencies are zero.")

def test_uniform_approximation(timestamps, micro_bin_width=20e-9):
    """
    MICROSCOPIC TEST: Verifies the irreducible randomness of the photon arrival times.
    Safely utilizes chunking to avoid MemoryError on large datasets.
    """
    print(f"\n=== MICRO TEST: Uniform Approximation (Cycle: {micro_bin_width*1e9} ns) ===")
    
    time_mod = np.mod(timestamps, micro_bin_width)
    normalized_times = time_mod / micro_bin_width
    
    plt.figure(figsize=(10, 6))
    plt.hist(normalized_times, bins=50, density=True, alpha=0.6, color=DELOITTE_GREY, label='Experimental Micro-Times')
    plt.axhline(y=1.0, color=DELOITTE_GREEN, linestyle='--', linewidth=2, label='Ideal Uniform Distribution')
    
    plt.title(f"Microscopic Arrival Distribution (Modulo {micro_bin_width*1e9} ns)")
    plt.xlabel("Normalized Arrival Time (0 to 1)")
    plt.ylabel("Density")
    plt.ylim(0, 1.5) 
    plt.legend()
    plt.grid(alpha=0.3)
    
    total_events = len(normalized_times)
    CHUNK_SIZE = 100_000
    num_chunks = total_events // CHUNK_SIZE
    
    if num_chunks < 1:
        ks_stat, p_val = kstest(normalized_times, 'uniform')
        pass_rate = 100.0 if p_val > 0.01 else 0.0
    else:
        print(f"-> Running memory-safe KS test across {num_chunks:,} chunks...")
        p_values = []
        for i in range(num_chunks):
            chunk = normalized_times[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
            _, p_val = kstest(chunk, 'uniform')
            p_values.append(p_val)
            
        p_values = np.array(p_values)
        pass_rate = np.mean(p_values > 0.01) * 100
        
        print(f"-> Median KS p-value across chunks: {np.median(p_values):.4e}")
        print(f"-> Chunks passing KS test (alpha=0.01): {pass_rate:.2f}%")

    if pass_rate >= 95.0:
        print("-> RESULT: PASSED. Distribution is statistically Uniform.")
    else:
        print("-> RESULT: WARNING. Slight deviation from Uniform detected.")

def test_system_stationarity(timestamps, chunk_duration_sec=60, micro_bin_width=20e-9):
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
        
        if len(chunk_data) < 100_000:
            continue 
            
        counts, _ = np.histogram(chunk_data, bins=np.arange(start_time, end_time, 0.01))
        lambda_val = np.mean(counts)
        
        subset_data = np.random.choice(chunk_data, size=100_000, replace=False)
        time_mod = np.mod(subset_data, micro_bin_width)
        normalized_times = time_mod / micro_bin_width
        _, p_val = kstest(normalized_times, 'uniform')
        
        chunk_times.append((start_time / 60)) 
        lambda_trends.append(lambda_val)
        p_value_trends.append(p_val)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(chunk_times, lambda_trends, marker='o', color=DELOITTE_BLACK, linestyle='-')
    ax1.axhline(np.mean(lambda_trends), color=DELOITTE_GREEN, linestyle='--', label='Average Rate')
    ax1.set_ylabel(r"Photon Rate ($\lambda$)")
    ax1.set_title(f"QRNG Hardware Stability Over Time (Chunk: {chunk_duration_sec}s)", fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, color=DELOITTE_LIGHT)
    
    ax2.plot(chunk_times, p_value_trends, marker='s', color=DELOITTE_GREY, linestyle='-')
    ax2.axhline(0.01, color='red', linestyle='-', linewidth=2, label='Failure Threshold (p=0.01)')
    ax2.set_ylabel("Uniformity p-value")
    ax2.set_xlabel("Elapsed Time (Minutes)")
    ax2.set_yscale('log') 
    ax2.legend()
    ax2.grid(alpha=0.3, color=DELOITTE_LIGHT)
    
    plt.tight_layout()

def test_exponential_decay(tau_seconds):
    """
    Verifies that the inter-arrival times follow an exponential distribution,
    which is the mathematical proof of a memoryless Poisson process.
    Uses chunking to avoid MemoryErrors on massive datasets.
    """
    print(f"\n=== MACRO TEST: Exponential Inter-Arrival Distribution ===")
    
    total_events = len(tau_seconds)
    if total_events < 1000:
        print("[WARNING] Not enough data for statistical testing.")
        return

    lambda_val = 1.0 / np.mean(tau_seconds)
    max_plot_time = 5.0 / lambda_val 
    filtered_tau = tau_seconds[tau_seconds < max_plot_time]
    
    plt.figure(figsize=(10, 6))
    bins = 100
    counts, bin_edges, _ = plt.hist(filtered_tau, bins=bins, density=True, 
                                    alpha=0.6, color=DELOITTE_GREY, label='Experimental Heralded $\\tau$')
    
    x_val = np.linspace(0, max_plot_time, 500)
    pdf_theoretical = expon.pdf(x_val, scale=1.0/lambda_val)
    plt.plot(x_val, pdf_theoretical, color=DELOITTE_GREEN, linewidth=2, 
             label=fr'Theoretical $f(\tau) = \lambda e^{{-\lambda \tau}}$')
    
    plt.title("Validation of Poissonian SPDC Emission Statistics", fontsize=14, pad=15)
    plt.xlabel("Inter-Arrival Time $\\tau$ (seconds)")
    plt.ylabel("Probability Density")
    plt.legend()
    
    CHUNK_SIZE = 100_000
    num_chunks = total_events // CHUNK_SIZE
    
    if num_chunks < 1:
        ks_stat, p_val = kstest(tau_seconds, 'expon', args=(0, 1.0/lambda_val))
        print(f"-> KS Test p-value: {p_val:.4e}")
        pass_rate = 100.0 if p_val > 0.01 else 0.0
    else:
        print(f"-> Splitting {total_events:,} events into {num_chunks:,} chunks for KS testing...")
        p_values = []
        
        for i in range(num_chunks):
            chunk = tau_seconds[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
            _, p_val = kstest(chunk, 'expon', args=(0, 1.0/lambda_val))
            p_values.append(p_val)
            
        p_values = np.array(p_values)
        pass_rate = np.mean(p_values > 0.01) * 100
        
        print(f"-> Median KS p-value across chunks: {np.median(p_values):.4e}")
        print(f"-> Chunks passing KS test (alpha=0.01): {pass_rate:.2f}%")

    if pass_rate >= 95.0: 
        print("-> RESULT: PASSED. Distribution is statistically Exponential and stable over time.")
    else:
        print("-> RESULT: WARNING. Deviation or drift detected across chunks.")
        
    plt.tight_layout()

# ==========================================
# 4. MAIN EXECUTION BLOCK 
# ==========================================
if __name__ == "__main__":
    # Assicurati che il nome coincida con il file generato dallo Swabian TimeTagger
    FILENAME = "3h_raw_photons.ttbin" 
    
    # ---------------------------------------------------------
    # PIPELINE 1: SINGLE CHANNEL STATISTICS (Raw Photons)
    # ---------------------------------------------------------
    print("\n" + "="*50 + "\nPIPELINE 1: RAW DETECTOR STATISTICS\n" + "="*50)
    timestamps = load_ttbin_data(FILENAME, target_channel=2)
    
    if len(timestamps) > 0:
        print("\n-> Engaging Statistical Physics Validators...")
        test_poissonian_statistics(timestamps, time_window=0.01) 
        test_uniform_approximation(timestamps, micro_bin_width=20e-9) 
        test_system_stationarity(timestamps, chunk_duration_sec=60)
    else:
        print("\n[WARNING] Pipeline 1 skipped due to lack of raw data.")

    # ---------------------------------------------------------
    # PIPELINE 2: HERALDED COINCIDENCE STATISTICS
    # ---------------------------------------------------------
    print("\n" + "="*50 + "\nPIPELINE 2: HERALDED COINCIDENCE STATISTICS\n" + "="*50)
    tau_data = extract_heralded_tau(FILENAME)
    
    if len(tau_data) > 0:
        test_exponential_decay(tau_data) 
    else:
        print("\n[WARNING] Pipeline 2 skipped due to lack of coincidence data.")

    # Render di tutti i grafici generati
    print("\n-> Rendering all plots. Close the windows to exit the script.")
    plt.show()