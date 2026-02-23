import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, chisquare, kstest, uniform

def load_thorlabs_data(file_path):
    """
    Loads timestamp data from the Thorlabs CSV export.
    Assumes the file contains a column with relative timestamps.
    """
    print(f"-> Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, comment='#') 
        # Extract the first column assuming it contains the timestamps in seconds
        timestamps = df.iloc[:, 0].values 
        print(f"-> Loaded {len(timestamps)} events.")
        return timestamps
    except Exception as e:
        print(f"[ERROR] Could not load file: {e}")
        return np.array([])

def test_poissonian_statistics(timestamps, time_window=0.01):
    """
    MACROSCOPIC REGIME VALIDATION
    Checks if the photon arrival count follows a Poisson distribution.
    
    Args:
        timestamps (array): Array of arrival times in seconds.
        time_window (float): Integration time window (T) in seconds (e.g., 10ms).
    """
    print(f"\n=== MACRO TEST: Poissonian Statistics (Window: {time_window*1000} ms) ===")
    
    max_time = timestamps[-1]
    bins = np.arange(0, max_time + time_window, time_window)
    counts, _ = np.histogram(timestamps, bins=bins)
    
    lambda_val = np.mean(counts)
    print(f"-> Average counts per window (Lambda): {lambda_val:.2f}")
    
    max_count = int(np.max(counts))
    
    # Shift bin edges by -0.5 to perfectly center the bars on integers
    bin_edges = np.arange(-0.5, max_count + 1.5, 1.0)
    x_values = np.arange(0, max_count + 1)
    
    plt.figure(figsize=(10, 6))
    observed_freq, _, _ = plt.hist(counts, bins=bin_edges, density=True, 
                                   alpha=0.6, color='blue', label='Experimental Data')
    
    theoretical_pmf = poisson.pmf(x_values, lambda_val)
    
    # Using raw f-string (fr) to avoid LaTeX escape character warnings
    plt.plot(x_values, theoretical_pmf, 'r--', linewidth=2, 
             label=fr'Theoretical Poisson ($\lambda$={lambda_val:.2f})')
    
    plt.title(f"Macroscopic Photon Statistics (Bin Window: {time_window*1000} ms)")
    plt.xlabel("Number of Photons Detected per Window")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Statistical Inference (Chi-Square Goodness of Fit)
    valid_indices = theoretical_pmf > 0
    obs = observed_freq[valid_indices]
    exp = theoretical_pmf[valid_indices]
    
    # Normalize expected to match observed sum
    exp = exp * (np.sum(obs) / np.sum(exp))
    
    chi2_stat, p_val = chisquare(f_obs=obs, f_exp=exp)
    
    print(f"-> Chi-Square Test p-value: {p_val:.4e}")
    if p_val > 0.05:
        print("-> RESULT: PASSED. Distribution is consistent with Poisson.")
    else:
        print("-> RESULT: WARNING. Deviation from perfect Poisson detected.")

def test_uniform_approximation(timestamps, micro_bin_width=20e-9):
    """
    MICROSCOPIC REGIME VALIDATION
    Checks if the arrival time within a small cycle is Uniformly distributed.
    
    Args:
        timestamps (array): Array of arrival times in seconds.
        micro_bin_width (float): The small interval T (e.g., 20ns).
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
    
    # Statistical Inference (Kolmogorov-Smirnov Test)
    ks_stat, p_val = kstest(normalized_times, 'uniform')
    
    print(f"-> Kolmogorov-Smirnov Test p-value: {p_val:.4e}")
    if p_val > 0.05:
        print("-> RESULT: PASSED. Distribution is statistically Uniform.")
    else:
        print("-> RESULT: WARNING. Slight deviation from Uniform detected.")

def test_system_stationarity(timestamps, chunk_duration_sec=300, micro_bin_width=20e-9):
    """
    STABILITY REGIME VALIDATION (STATIONARITY)
    Slices a long dataset into chunks and evaluates the stability 
    of the photon generation rate and uniformity over time.
    """
    print(f"\n=== STATIONARITY TEST: Evaluating repeatability over chunks of {chunk_duration_sec}s ===")
    
    max_time = timestamps[-1]
    num_chunks = int(max_time // chunk_duration_sec)
    
    if num_chunks < 3:
        print("[WARNING] The dataset is too short to test long-term stationarity.")
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
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    FILENAME = "thorlabs_data_sample.csv" 
    
    # --- DUMMY DATA GENERATOR ---
    print("[WARNING] Generating DUMMY data for simulation purposes...")
    avg_rate = 100000 # 100 kHz count rate
    total_time = 300  # 300 seconds (5 minutes) to allow stationarity chunking
    num_events = avg_rate * total_time
    
    intervals = np.random.exponential(1/avg_rate, num_events)
    timestamps = np.cumsum(intervals)
    # ------------------------------------

    if len(timestamps) > 0:
        # 1. Macro Test
        test_poissonian_statistics(timestamps, time_window=0.01) 
        
        # 2. Micro Test
        test_uniform_approximation(timestamps, micro_bin_width=50e-9) 
        
        # 3. Stationarity Test
        # We use 60-second chunks for the 5-minute dummy dataset
        test_system_stationarity(timestamps, chunk_duration_sec=60) 
        
        # Show all accumulated plots at the very end
        print("\n-> Rendering all plots. Close the windows to exit the script.")
        plt.show()