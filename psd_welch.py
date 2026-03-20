import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import welch

def load_raw_chunk(filepath, num_bits=5_000_000):
    """Loads a specific number of bits from the raw file for frequency analysis."""
    print(f"-> Loading {num_bits:,} bits from {os.path.basename(filepath)}...")
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None

    bytes_to_read = (num_bits // 8) + 1
    with open(filepath, 'rb') as f:
        raw_bytes = np.frombuffer(f.read(bytes_to_read), dtype=np.uint8)
        
    bitstream = np.unpackbits(raw_bytes)[:num_bits]
    
    # Convert 0/1 to -1/1 to center the signal around zero (removes the massive DC offset)
    return bitstream.astype(np.float32) * 2.0 - 1.0

def plot_power_spectral_density(bitstream, title_label):
    """
    Calculates and plots the PSD using Welch's method with a premium aesthetic.
    """
    print(f"-> Calculating Power Spectral Density for {title_label}...")
    
    # Welch's method divides the data into overlapping segments to reduce noise
    # nperseg defines the resolution of the frequency bins
    frequencies, psd = welch(bitstream, fs=1.0, nperseg=8192, scaling='density')

    # Premium Aesthetic Colors
    dark_slate = '#2A2E33'
    deloitte_green = '#86BC25'
    light_grey = '#EAEAEA'
    text_grey = '#5A5A5A'
    accent_red = '#E34053'

    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(16, 6), facecolor='#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Plot the PSD
    # We use a logarithmic scale for the Y-axis (Decibels) to make spikes obvious
    psd_db = 10 * np.log10(psd + 1e-12) 
    
    ax.plot(frequencies, psd_db, color=dark_slate, linewidth=1.2, alpha=0.85)

    # Clean axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')

    ax.yaxis.grid(True, linestyle='-', alpha=0.5, color=light_grey)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Typography
    ax.set_title(f'Hardware Frequency Diagnostic: {title_label} Dimension', 
                 fontsize=16, color=dark_slate, weight='500', pad=25, loc='left')
    ax.set_ylabel('Power Spectral Density (dB/Hz)', fontsize=11, color=text_grey, labelpad=15)
    ax.set_xlabel('Normalized Frequency', fontsize=11, color=text_grey, labelpad=15)
    
    ax.tick_params(axis='x', bottom=True, color='#CCCCCC', labelcolor=text_grey)
    ax.tick_params(axis='y', left=False, labelcolor=text_grey)
    
    ax.set_xlim(0, 0.5)  # Nyquist limit for discrete data is 0.5

    # Calculate and plot the theoretical ideal (White Noise Baseline)
    ideal_baseline = np.mean(psd_db)
    ax.axhline(y=ideal_baseline, color=deloitte_green, linestyle='--', linewidth=1.5, alpha=0.8, 
               label='Ideal Flat Baseline (White Noise)')

    # Look for aggressive peaks (Spikes that are 10dB above the baseline)
    peak_threshold = ideal_baseline + 10
    peaks = psd_db > peak_threshold
    if np.any(peaks):
        ax.scatter(frequencies[peaks], psd_db[peaks], color=accent_red, s=20, zorder=5, 
                   label='Hardware Resonances / Clock Leaks')
        print("[!] WARNING: Significant periodic frequencies detected.")
    else:
        print("[+] Good news: No massive periodic spikes detected.")

    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='white', fontsize=11)

    fig.tight_layout()
    
    filename = f"psd_diagnostic_{title_label.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"-> Saved high-res diagnostic plot to: {filename}")
    
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # CRITICAL: Point these to your RAW, UNWHITENED files
    FILE_TEMPORAL_RAW = r"data\\whitened\\final_attempt\\pure_3ht_keys.bin"
    FILE_SPATIAL_RAW  = r"data\\whitened\\final_attempt\\pure_3hs_keys.bin"

    BITS_TO_ANALYZE = 137_000_000

    # Analyze Temporal
    temporal_raw = load_raw_chunk(FILE_TEMPORAL_RAW, BITS_TO_ANALYZE)
    if temporal_raw is not None:
        plot_power_spectral_density(temporal_raw, "Temporal")

    # Analyze Spatial
    spatial_raw = load_raw_chunk(FILE_SPATIAL_RAW, BITS_TO_ANALYZE)
    if spatial_raw is not None:
        plot_power_spectral_density(spatial_raw, "Spatial")