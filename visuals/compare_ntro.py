import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# ==========================================
# DELOITTE CORPORATE AESTHETIC CONFIGURATION
# ==========================================
DELOITTE_BLUE  = '#0097A9'  # Spatial Domain
DELOITTE_GREEN = '#86BC25'  # Temporal Domain
DELOITTE_RED   = '#DA291C'  # Noise Floor/Limits
DELOITTE_GREY  = '#A6A6A6'  # Neutral/Baselines
DELOITTE_DARK  = '#2A2E33'  # Text & Axes
WHITE          = '#FFFFFF'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': DELOITTE_DARK,
    'text.color': DELOITTE_DARK,
    'figure.facecolor': WHITE,
    'axes.facecolor': WHITE,
    'axes.titlesize': 15,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.frameon': False,
})

# ==========================================
# 1. METRICS EXTRACTION ENGINE (UPGRADED)
# ==========================================
def extract_metrics(filepath, max_lags=50):
    if not os.path.exists(filepath):
        print(f"[WARNING] File not found: {filepath}")
        return None

    raw_bytes = np.fromfile(filepath, dtype=np.uint8)
    bitstream = np.unpackbits(raw_bytes)
    
    n = len(bitstream)
    if n < 100000: # Increased minimum length for rigorous deep-lag stats
        print(f"[WARNING] File {filepath} too short for deep-lag analysis.")
        return None

    p_1 = np.mean(bitstream)
    p_0 = 1.0 - p_1
    bias_deviation = abs(p_1 - 0.5)

    # ---------------------------------------------------------
    # UPGRADE: Deep-Lag Autocorrelation Extraction (Lags 1 to 50)
    # ---------------------------------------------------------
    autocorr = np.zeros(max_lags)
    # Mean centering the bitstream for faster/accurate covariance math
    x_centered = bitstream - p_1 
    variance = np.var(bitstream)
    
    if variance == 0:
        autocorr.fill(1e-8)
    else:
        for lag in range(1, max_lags + 1):
            # Calculate correlation via centered dot product
            cov = np.dot(x_centered[:-lag], x_centered[lag:]) / (n - lag)
            r = cov / variance
            autocorr[lag-1] = max(abs(r), 1e-8) # Floor at 1e-8 for log scale plotting

    return {
        'p_0': p_0,
        'p_1': p_1,
        'bias': bias_deviation,
        'autocorr': autocorr,
        'bytes': len(raw_bytes),
        'bits': n
    }

# ==========================================
# 2. ANALYTICAL PLOTTING FUNCTIONS (UPGRADED)
# ==========================================

# ... [Keep your existing plot_1_logarithmic_bias here] ...

def plot_2_deep_lag_correlogram(metrics_dict, max_lags=50):
    """
    Plot 2: Deep-Lag Autocorrelation Correlogram (Lags 1 to 50)
    Replaces the 3-lag bar chart and the visual canvas.
    """
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    lags = np.arange(1, max_lags + 1)

    n_bits = metrics_dict['Spatial NIST']['bits']
    sigma = 1.0 / np.sqrt(n_bits)
    three_sigma = 3.0 * sigma

    domains = [
        ('Spatial Domain Hardware Memory', metrics_dict['Spatial Raw'], metrics_dict['Spatial NIST'], metrics_dict['Spatial AI'], axs[0]),
        ('Temporal Domain Hardware Memory', metrics_dict['Temporal Raw'], metrics_dict['Temporal NIST'], metrics_dict['Temporal AI'], axs[1])
    ]

    for title, raw_data, nist_data, ai_data, ax in domains:
        # Plotting the raw hardware baseline
        ax.plot(lags, raw_data['autocorr'], color=DELOITTE_GREY, alpha=0.8, lw=2, 
                label='Raw Hardware Output', zorder=2)
        
        # Plotting the NIST and AI whitened outputs using scatter/stem-like markers
        ax.scatter(lags, nist_data['autocorr'], color=DELOITTE_BLUE, alpha=0.7, s=30, 
                   label='NIST SP 800-90B Extracted', zorder=3)
        ax.scatter(lags, ai_data['autocorr'], color=DELOITTE_GREEN, alpha=0.9, s=30, marker='s', 
                   label='Attention-LSTM Extracted', zorder=4)

        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_yscale('log')
        ax.set_xlim(0.5, max_lags + 0.5)

        # Statistical Thresholds
        ax.axhspan(1e-9, three_sigma, color=DELOITTE_RED, alpha=0.05, lw=0, zorder=1)
        ax.axhline(y=three_sigma, color=DELOITTE_RED, linestyle='-', alpha=0.6, lw=1.5, 
                   label=r'NIST $3\sigma$ Bound ($\approx 99.7\%$ confidence)', zorder=1)
        ax.axhline(y=sigma, color=DELOITTE_RED, linestyle='--', alpha=0.3, lw=1, 
                   label=r'Statistical Noise Floor ($1\sigma$)', zorder=1)

        despine_ax(ax)
        ax.grid(axis='both', alpha=0.15, linestyle='-')
        ax.set_ylabel('Absolute Autocorrelation Coefficient $|r_k|$', fontweight='bold', labelpad=10)
        
        # Ensure y-axis accommodates the raw hardware spikes
        max_val = max(np.max(raw_data['autocorr']), three_sigma * 5)
        ax.set_ylim(bottom=1e-8, top=max_val * 2)

    axs[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15), ncol=2)
    axs[1].set_xlabel('Lag distance $k$ (bits)', fontweight='bold', labelpad=10)

    plt.suptitle('Deep-Lag Cryptographic Autocorrelation Analysis (Lags 1 - 50)', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plot_2_deep_lag_correlogram.png', dpi=400, bbox_inches='tight')
    plt.close()

# ... [Keep your existing plot_4_global_yield here] ...

def plot_4_global_yield(metrics_dict):
    """Plot 4: Compact Grouped Bar Yield Comparison (NIST vs AI)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['NIST SP 800-90B\nExtraction', 'Attention-LSTM\nExtraction']
    x = np.arange(len(categories))
    width = 0.35
    
    s_nist_mb = metrics_dict['Spatial NIST']['bytes'] / (1024 * 1024)
    t_nist_mb = metrics_dict['Temporal NIST']['bytes'] / (1024 * 1024)
    s_ai_mb = metrics_dict['Spatial AI']['bytes'] / (1024 * 1024)
    t_ai_mb = metrics_dict['Temporal AI']['bytes'] / (1024 * 1024)
    
    spatial_yields = [s_nist_mb, s_ai_mb]
    temporal_yields = [t_nist_mb, t_ai_mb]
    
    # Unstacked grouped bars for perfect consistency with Plot 1
    ax.bar(x - width/2, spatial_yields, width, label='Spatial Yield', color=DELOITTE_BLUE, alpha=0.95, edgecolor=WHITE, lw=1.5)
    ax.bar(x + width/2, temporal_yields, width, label='Temporal Yield', color=DELOITTE_GREEN, alpha=0.95, edgecolor=WHITE, lw=1.5)
    
    ax.set_ylabel('Certified Randomness (Megabytes)', fontweight='bold', labelpad=10)
    ax.set_title('Global Entropy Harvesting Yield', pad=25, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='600')
    
    # Annotate yields directly on bars
    for i, (s, t) in enumerate(zip(spatial_yields, temporal_yields)):
        ax.text(i - width/2, s + (s * 0.02), f'{s:.2f} MB', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(i + width/2, t + (t * 0.02), f'{t:.2f} MB', ha='center', va='bottom', fontweight='bold', fontsize=10)

    despine_ax(ax)
    ax.grid(axis='y', alpha=0.15, linestyle='-')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.05))
    
    plt.tight_layout()
    plt.savefig('plot_4_global_yield.png', dpi=400, bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN SCRIPT EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== DELOITTE QRNG: GENERATING COMPACT 6-FILE ANALYTICAL SUITE ===")
    
    files = {
        'Spatial Raw': r"data\\raw\\3hours_nopeople\\spatial_3hraw_bitstream.bin",
        'Spatial NIST': r"data\\whitened\\final_attempt\\FFTw_3hs_nkeys.bin",
        'Spatial AI': r"data\\whitened\\final_attempt\\FFTw_3hs_AIkeys.bin",
        'Temporal Raw': r"data\\raw\\3hours_nopeople\\temporal_3hraw_bitstream.bin",
        'Temporal NIST': r"data\\whitened\\final_attempt\\FFTw_3ht_nkeys.bin",
        'Temporal AI': r"data\\whitened\\final_attempt\\FFTw_3ht_AIkeys.bin"
    }
    
    data = {}
    for name, path in files.items():
        metrics = extract_metrics(path)
        if metrics:
            data[name] = metrics
            print(f"[OK] Processed {name}")
            
    if len(data) == 6:
        # ... (inside if len(data) == 6:) ...
        print("\n-> All data loaded. Generating compact comparative plots...")
        plot_1_logarithmic_bias(data)
        print("   - Created: plot_1_log_bias.png")
        plot_2_deep_lag_correlogram(data) # <--- UPDATED CALL
        print("   - Created: plot_2_deep_lag_correlogram.png")
        plot_4_global_yield(data)
        print("   - Created: plot_4_global_yield.png")
        print("\n=== PLOT GENERATION COMPLETE ===")
    else:
        print("\n[ERROR] Missing files. Found", len(data), "but expected 6.")