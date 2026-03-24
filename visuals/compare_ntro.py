import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# DELOITTE CORPORATE AESTHETIC CONFIGURATION
# ==========================================
DELOITTE_GREEN = '#86BC25'
DELOITTE_BLACK = '#000000'
DELOITTE_GREY  = '#53565A'
DELOITTE_RED   = '#DA291C' 
DELOITTE_BLUE  = '#0097A9' 

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': DELOITTE_GREY,
    'text.color': DELOITTE_BLACK,
    'figure.facecolor': 'white',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# ==========================================
# 1. METRICS EXTRACTION ENGINE (MULTI-LAG)
# ==========================================
def extract_metrics(filepath):
    if not os.path.exists(filepath):
        print(f"[WARNING] File not found: {filepath}")
        return None

    raw_bytes = np.fromfile(filepath, dtype=np.uint8)
    bitstream = np.unpackbits(raw_bytes)
    
    n = len(bitstream)
    if n < 10000: return None

    # Distribution and Bias
    p_1 = np.mean(bitstream)
    p_0 = 1.0 - p_1
    bias_deviation = abs(p_1 - 0.5)

    # Multi-Lag Autocorrelation (Lags 1, 2, 3)
    autocorr = {}
    for lag in [1, 2, 3]:
        x_current = bitstream[:-lag]
        x_shifted = bitstream[lag:]
        matrix = np.corrcoef(x_current, x_shifted)
        # Floor to 1e-8 to prevent log(0) graphing errors for perfect strings
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

# ==========================================
# 2. ANALYTICAL PLOTTING FUNCTIONS
# ==========================================

def plot_1_logarithmic_bias(metrics_dict):
    """Plot 1: Logarithmic Bias Analysis (The 'Microscope' view)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Spatial\nRaw', 'Spatial\nWhitened', 'Temporal\nRaw', 'Temporal\nWhitened']
    
    bias_vals = [
        max(metrics_dict['Spatial Raw']['bias'], 1e-8),
        max(metrics_dict['Spatial Whitened']['bias'], 1e-8),
        max(metrics_dict['Temporal Raw']['bias'], 1e-8),
        max(metrics_dict['Temporal Whitened']['bias'], 1e-8)
    ]
    
    x = np.arange(len(labels))
    colors = [DELOITTE_GREY, DELOITTE_GREEN, DELOITTE_GREY, DELOITTE_GREEN]

    bars = ax.bar(x, bias_vals, width=0.5, color=colors)

    ax.set_ylabel('Absolute Bias |P(1) - 0.5|')
    ax.set_title('Cryptographic Purification: Logarithmic Bias Eradication', pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_yscale('log')
    
    n_bits = metrics_dict['Spatial Whitened']['bits']
    noise_floor = 1.0 / np.sqrt(n_bits)
    ax.axhline(y=noise_floor, color=DELOITTE_RED, linestyle='--', alpha=0.7, 
               label=f'Finite Sample Limit ($1/\\sqrt{{N}}$) $\\approx$ {noise_floor:.2e}')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2e}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('plot_1_log_bias.png', dpi=300)
    plt.close()

def plot_2_multi_lag_autocorr(metrics_dict):
    """Plot 2: Multi-Lag Autocorrelation (Depths 1, 2, 3) with NIST Bounds"""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    lags = [1, 2, 3]
    x = np.arange(len(lags))
    width = 0.35

    # Target number of bits for statistical bounds
    n_bits = metrics_dict['Spatial Whitened']['bits']
    sigma = 1.0 / np.sqrt(n_bits)
    three_sigma = 3.0 * sigma

    domains = [
        ('Spatial', metrics_dict['Spatial Raw'], metrics_dict['Spatial Whitened'], axs[0]),
        ('Temporal', metrics_dict['Temporal Raw'], metrics_dict['Temporal Whitened'], axs[1])
    ]

    for title, raw_data, white_data, ax in domains:
        raw_vals = [raw_data['autocorr'][l] for l in lags]
        white_vals = [white_data['autocorr'][l] for l in lags]

        bars_raw = ax.bar(x - width/2, raw_vals, width, label=f'Raw', color=DELOITTE_GREY)
        bars_white = ax.bar(x + width/2, white_vals, width, label=f'Whitened', color=DELOITTE_GREEN)

        ax.set_title(f'{title} Domain Memory Suppression', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Lag {l}' for l in lags], fontweight='bold')
        ax.set_yscale('log')

        ax.axhline(y=sigma, color=DELOITTE_GREY, linestyle='--', alpha=0.7, label=f'1$\\sigma$ Fluctuation')
        ax.axhline(y=three_sigma, color=DELOITTE_RED, linestyle='-', alpha=0.8, label=f'3$\\sigma$ NIST Limit')

        for bars in [bars_raw, bars_white]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        ax.grid(axis='y', alpha=0.3, which='both')
        ax.legend(loc='upper right')

    axs[0].set_ylabel('Absolute Serial Autocorrelation')
    # Set y-limits to ensure everything is visible
    axs[0].set_ylim(bottom=1e-8, top=1e-1)

    plt.suptitle('Deep-Lag Cryptographic Autocorrelation Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plot_2_multilag_autocorr.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_3_visual_bitstream(metrics_dict):
    """Plot 3: The 'TV Static' 2D Canvas of the Spatial bits"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    raw_matrix = metrics_dict['Spatial Raw']['bitstream_sample'].reshape(100, 100)
    whitened_matrix = metrics_dict['Spatial Whitened']['bitstream_sample'].reshape(100, 100)
    
    cmap = plt.cm.colors.ListedColormap(['#FFFFFF', DELOITTE_GREEN])
    
    axs[0].imshow(raw_matrix, cmap=cmap, interpolation='none')
    axs[0].set_title('Spatial Raw (Hardware Artifacts)', fontweight='bold')
    axs[0].axis('off')
    
    axs[1].imshow(whitened_matrix, cmap=cmap, interpolation='none')
    axs[1].set_title('Spatial Whitened (Cryptographic Uniformity)', fontweight='bold')
    axs[1].axis('off')

    plt.suptitle('Bitstream Canvas Representation (10,000 bits)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_3_visual_canvas.png', dpi=300)
    plt.close()

def plot_4_global_yield(metrics_dict):
    """Plot 4: The Combined Spatial + Temporal Yield"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    spatial_mb = metrics_dict['Spatial Whitened']['bytes'] / (1024 * 1024)
    temporal_mb = metrics_dict['Temporal Whitened']['bytes'] / (1024 * 1024)
    
    categories = ['Cryptographic Yield']
    
    p1 = ax.bar(categories, [spatial_mb], color=DELOITTE_BLUE, label=f'Spatial Yield', width=0.4)
    p2 = ax.bar(categories, [temporal_mb], bottom=[spatial_mb], color=DELOITTE_GREEN, label=f'Temporal Yield', width=0.4)
    
    total_mb = spatial_mb + temporal_mb
    
    ax.set_ylabel('Certified Randomness (Megabytes)')
    ax.set_title('Global Entropy Harvesting (Multi-Dimensional)', pad=20, fontweight='bold')
    ax.set_ylim(0, total_mb * 1.25)
    
    ax.text(0, total_mb + (total_mb * 0.03), f'Total Valid Entropy:\n{total_mb:.2f} MB', 
            ha='center', va='bottom', fontweight='bold', color=DELOITTE_BLACK)
    
    if spatial_mb > 0:
        ax.text(0, spatial_mb / 2, f'{spatial_mb:.2f} MB', ha='center', va='center', color='white', fontweight='bold')
    if temporal_mb > 0:
        ax.text(0, spatial_mb + (temporal_mb / 2), f'{temporal_mb:.2f} MB', ha='center', va='center', color='white', fontweight='bold')

    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot_4_global_yield.png', dpi=300)
    plt.close()

# ==========================================
# MAIN SCRIPT EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== DELOITTE QRNG: GENERATING FULL ANALYTICAL SUITE ===")
    
    # Check your specific file paths
    files = {
        'Spatial Raw': r"data\\raw\\3hours_nopeople\\spatial_3hraw_bitstream.bin",
        'Spatial Whitened': r"data\\whitened\\final_attempt\\pure_3hs_keys.bin",
        'Temporal Raw': r"data\\raw\\3hours_nopeople\\time_2c_3hraw_bitstream.bin",
        'Temporal Whitened': r"data\\whitened\\final_attempt\\pure_3ht_keys.bin"
    }
    
    data = {}
    for name, path in files.items():
        metrics = extract_metrics(path)
        if metrics:
            data[name] = metrics
            print(f"[OK] Processed {name}")
            
    if len(data) == 4:
        print("\n-> All data loaded. Generating plots...")
        plot_1_logarithmic_bias(data)
        print("   - Created: plot_1_log_bias.png")
        plot_2_multi_lag_autocorr(data)
        print("   - Created: plot_2_multilag_autocorr.png")
        plot_3_visual_bitstream(data)
        print("   - Created: plot_3_visual_canvas.png")
        plot_4_global_yield(data)
        print("   - Created: plot_4_global_yield.png")
        print("\n=== PLOT GENERATION COMPLETE ===")
    else:
        print("\n[ERROR] Missing files. Cannot generate complete comparative plots.")