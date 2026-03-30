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
# 1. METRICS EXTRACTION ENGINE
# ==========================================
def extract_metrics(filepath):
    if not os.path.exists(filepath):
        print(f"[WARNING] File not found: {filepath}")
        return None

    raw_bytes = np.fromfile(filepath, dtype=np.uint8)
    bitstream = np.unpackbits(raw_bytes)
    
    n = len(bitstream)
    if n < 10000: return None

    p_1 = np.mean(bitstream)
    p_0 = 1.0 - p_1
    bias_deviation = abs(p_1 - 0.5)

    autocorr = {}
    for lag in [1, 2, 3]:
        x_current = bitstream[:-lag]
        x_shifted = bitstream[lag:]
        matrix = np.corrcoef(x_current, x_shifted)
        autocorr[lag] = max(abs(matrix[0, 1]), 1e-8)

    return {
        'p_0': p_0,
        'p_1': p_1,
        'bias': bias_deviation,
        'autocorr': autocorr,
        'bytes': len(raw_bytes),
        'bits': n,
        'bitstream_sample': bitstream[:10000] 
    }

def despine_ax(ax):
    """Utility to remove top/right borders for a clean, open aesthetic."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')

# ==========================================
# 2. ANALYTICAL PLOTTING FUNCTIONS
# ==========================================

def plot_1_logarithmic_bias(metrics_dict):
    """Plot 1: High-Sensitivity Logarithmic Bias Analysis"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Raw Hardware\n(Physical Output)', 'NIST SP 800-90B\n(Standard Ratio)', 'Attention-LSTM\n(AI Ratio)']
    x = np.arange(len(categories))
    width = 0.30  # Slimmer bars for a sharper, metrological aesthetic

    def get_bias(key): return max(metrics_dict.get(key, {}).get('bias', 1e-9), 1e-9)

    spatial_vals = [get_bias('Spatial Raw'), get_bias('Spatial NIST'), get_bias('Spatial AI')]
    temporal_vals = [get_bias('Temporal Raw'), get_bias('Temporal NIST'), get_bias('Temporal AI')]
    
    bars_s = ax.bar(x - width/2, spatial_vals, width, label='Spatial Domain', color=DELOITTE_BLUE, alpha=0.9, edgecolor=WHITE, lw=1.2)
    bars_t = ax.bar(x + width/2, temporal_vals, width, label='Temporal Domain', color=DELOITTE_GREEN, alpha=0.9, edgecolor=WHITE, lw=1.2)

    ax.set_ylabel('Absolute Bias $|P(1) - 0.5|$', fontweight='bold', labelpad=10)
    ax.set_title('Cryptographic Purification: High-Precision Bias Eradication', pad=25, fontweight='bold', color=DELOITTE_DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='600')
    ax.set_yscale('log')
    
    n_bits = metrics_dict['Spatial NIST']['bits']
    noise_floor = 1.0 / np.sqrt(n_bits)
    
    # Dynamic Y-limits to maximize sensitivity and focus on the data spread
    min_val = min(min(spatial_vals), min(temporal_vals))
    max_val = max(max(spatial_vals), max(temporal_vals))
    ax.set_ylim(max(1e-8, min_val * 0.1), max_val * 3)
    
    # Noise Floor Shading
    ax.axhspan(1e-9, noise_floor, color=DELOITTE_RED, alpha=0.04, lw=0)
    ax.axhline(y=noise_floor, color=DELOITTE_RED, linestyle='--', alpha=0.8, lw=1.5,
               label=f'Statistical Limit ($1/\\sqrt{{N}}$) $\\approx$ {noise_floor:.2e}')

    # High-precision value annotations
    for bars in [bars_s, bars_t]:
        for bar in bars:
            height = bar.get_height()
            if height > 1e-8:
                ax.annotate(f'{height:.2e}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', 
                            fontsize=9, fontweight='bold', color=DELOITTE_DARK)

    despine_ax(ax)
    ax.grid(axis='y', alpha=0.2, linestyle=':')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.05))
    
    plt.tight_layout()
    plt.savefig('plot_1_log_bias.png', dpi=400, bbox_inches='tight')
    plt.close()



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

def plot_2_deep_lag_correlogram(metrics_dict, max_lags=50):
    """Plot 2: Deep-Lag Correlogram with Global Legend and Area Fills"""
    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    lags = np.arange(1, max_lags + 1)

    n_bits = metrics_dict['Spatial NIST']['bits']
    sigma = 1.0 / np.sqrt(n_bits)
    three_sigma = 3.0 * sigma

    domains = [
        ('Spatial Domain Hardware Memory Suppression', metrics_dict['Spatial Raw'], metrics_dict['Spatial NIST'], metrics_dict['Spatial AI'], axs[0]),
        ('Temporal Domain Hardware Memory Suppression', metrics_dict['Temporal Raw'], metrics_dict['Temporal NIST'], metrics_dict['Temporal AI'], axs[1])
    ]

    for title, raw_data, nist_data, ai_data, ax in domains:
        # Raw hardware baseline with subtle fill to visually ground the background noise
        ax.plot(lags, raw_data['autocorr'], color=DELOITTE_GREY, alpha=0.6, lw=1.5, 
                label='Raw Hardware Output', zorder=2)
        ax.fill_between(lags, 1e-9, raw_data['autocorr'], color=DELOITTE_GREY, alpha=0.1, zorder=1)
        
        # NIST and AI whitened outputs with distinct, clear geometric markers
        ax.scatter(lags, nist_data['autocorr'], color=DELOITTE_BLUE, alpha=0.85, s=40, marker='o',
                   label='NIST SP 800-90B Extracted', zorder=4)
        ax.scatter(lags, ai_data['autocorr'], color=DELOITTE_GREEN, alpha=0.95, s=40, marker='d', 
                   label='Attention-LSTM Extracted', zorder=5)

        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_yscale('log')
        ax.set_xlim(0, max_lags + 1)

        # Statistical Thresholds
        ax.axhspan(1e-9, three_sigma, color=DELOITTE_RED, alpha=0.04, lw=0, zorder=0)
        ax.axhline(y=three_sigma, color=DELOITTE_RED, linestyle='-', alpha=0.6, lw=1.5, 
                   label=r'NIST $3\sigma$ Bound', zorder=3)
        ax.axhline(y=sigma, color=DELOITTE_RED, linestyle='--', alpha=0.4, lw=1, 
                   label=r'Noise Floor ($1\sigma$)', zorder=3)

        # ... [keep everything inside the 'for' loop above this unchanged] ...

        despine_ax(ax)
        ax.grid(axis='both', alpha=0.15, linestyle=':')
        ax.set_ylabel('Absolute Autocorrelation $|r_k|$', fontweight='bold', labelpad=10)
        
        # [!] CORRECTED: Restored the absolute floor to 1e-8 so no data points are cut off
        max_val = max(np.max(raw_data['autocorr']), three_sigma * 5)
        ax.set_ylim(bottom=1e-8, top=max_val * 2)

    axs[1].set_xlabel('Lag distance $k$ (bits)', fontweight='bold', labelpad=10)

    # Global Unified Legend placed strictly in the bottom right empty space
    handles, labels = axs[0].get_legend_handles_labels()
    axs[1].legend(handles, labels, loc='lower right', ncol=1, frameon=True, 
                  facecolor=WHITE, edgecolor='#E0E0E0', fontsize=10, framealpha=0.95)

    # Clean layout with no artificial top padding
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    plt.savefig('plot_2_deep_lag_correlogram.png', dpi=400, bbox_inches='tight')
    plt.close()


    
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