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
    """Plot 1: 3-Column Compact Bias Analysis"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Raw Hardware\n(Physical Output)', 'NIST SP 800-90B\n(Standard Ratio)', 'Attention-LSTM\n(AI Ratio)']
    x = np.arange(len(categories))
    width = 0.35  

    def get_bias(key): return max(metrics_dict.get(key, {}).get('bias', 1e-8), 1e-8)

    spatial_vals = [get_bias('Spatial Raw'), get_bias('Spatial NIST'), get_bias('Spatial AI')]
    temporal_vals = [get_bias('Temporal Raw'), get_bias('Temporal NIST'), get_bias('Temporal AI')]
    
    # White edgecolor creates a crisp, professional separation
    bars_s = ax.bar(x - width/2, spatial_vals, width, label='Spatial Domain', color=DELOITTE_BLUE, alpha=0.95, edgecolor=WHITE, lw=1.5)
    bars_t = ax.bar(x + width/2, temporal_vals, width, label='Temporal Domain', color=DELOITTE_GREEN, alpha=0.95, edgecolor=WHITE, lw=1.5)

    ax.set_ylabel('Absolute Bias $|P(1) - 0.5|$', fontweight='bold', labelpad=10)
    ax.set_title('Cryptographic Purification: Bias Eradication Performance', pad=25, fontweight='bold', color=DELOITTE_DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='600')
    ax.set_yscale('log')
    
    n_bits = metrics_dict['Spatial NIST']['bits']
    noise_floor = 1.0 / np.sqrt(n_bits)
    
    # Noise Floor Shading
    ax.axhspan(1e-9, noise_floor, color=DELOITTE_RED, alpha=0.05, lw=0)
    ax.axhline(y=noise_floor, color=DELOITTE_RED, linestyle='--', alpha=0.6, lw=1.5,
               label=f'Statistical Limit ($1/\\sqrt{{N}}$) $\\approx$ {noise_floor:.2e}')

    # Value Annotations
    for bars in [bars_s, bars_t]:
        for bar in bars:
            height = bar.get_height()
            if height > 1e-8:
                ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', 
                            fontsize=9, fontweight='600', color=DELOITTE_DARK)

    despine_ax(ax)
    ax.grid(axis='y', alpha=0.15, linestyle='-')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.05))
    
    plt.tight_layout()
    plt.savefig('plot_1_log_bias.png', dpi=400, bbox_inches='tight')
    plt.close()

def plot_2_multi_lag_autocorr(metrics_dict):
    """Plot 2: Compact Autocorrelation grouped by Lags"""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    lags = [1, 2, 3]
    x = np.arange(len(lags))
    width = 0.25

    n_bits = metrics_dict['Spatial NIST']['bits']
    sigma = 1.0 / np.sqrt(n_bits)
    three_sigma = 3.0 * sigma

    # We maintain Spatial/Temporal split here because combining 6 bars per lag is visually overwhelming
    domains = [
        ('Spatial Domain', metrics_dict['Spatial Raw'], metrics_dict['Spatial NIST'], metrics_dict['Spatial AI'], axs[0]),
        ('Temporal Domain', metrics_dict['Temporal Raw'], metrics_dict['Temporal NIST'], metrics_dict['Temporal AI'], axs[1])
    ]

    for title, raw_data, nist_data, ai_data, ax in domains:
        raw_vals = [raw_data['autocorr'][l] for l in lags]
        nist_vals = [nist_data['autocorr'][l] for l in lags]
        ai_vals = [ai_data['autocorr'][l] for l in lags]

        # Use Greys for Raw, and the standard Deloitte Palette for the methodologies
        ax.bar(x - width, raw_vals, width, label='Raw Hardware', color=DELOITTE_GREY, alpha=0.8)
        ax.bar(x, nist_vals, width, label='NIST SP 800-90B', color=DELOITTE_BLUE, alpha=0.9, edgecolor=WHITE, lw=1)
        ax.bar(x + width, ai_vals, width, label='Attention-LSTM', color=DELOITTE_GREEN, alpha=0.9, edgecolor=WHITE, lw=1)

        ax.set_title(f'{title} Memory Suppression', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Lag {l}' for l in lags], fontweight='600')
        ax.set_yscale('log')

        ax.axhspan(1e-9, three_sigma, color=DELOITTE_RED, alpha=0.04, lw=0)
        ax.axhline(y=three_sigma, color=DELOITTE_RED, linestyle='-', alpha=0.5, lw=1.5, label='NIST $3\\sigma$ Bound')

        despine_ax(ax)
        ax.grid(axis='y', alpha=0.15, linestyle='-')
        if title == 'Temporal Domain':
            ax.legend(loc='upper right')

    axs[0].set_ylabel('Absolute Serial Autocorrelation', fontweight='bold', labelpad=10)
    axs[0].set_ylim(bottom=1e-8, top=1e-1)

    plt.suptitle('Deep-Lag Cryptographic Autocorrelation Analysis', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('plot_2_multilag_autocorr.png', dpi=400, bbox_inches='tight')
    plt.close()

def plot_3_visual_bitstream(metrics_dict):
    """Plot 3: 3-Panel Canvas for Spatial (Unchanged, as it is already 3 columns)"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    raw_mat = metrics_dict['Spatial Raw']['bitstream_sample'].reshape(100, 100)
    nist_mat = metrics_dict['Spatial NIST']['bitstream_sample'].reshape(100, 100)
    ai_mat = metrics_dict['Spatial AI']['bitstream_sample'].reshape(100, 100)
    
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_edgecolor('#E0E0E0')
            spine.set_linewidth(1)
        ax.set_xticks([])
        ax.set_yticks([])
    
    axs[0].imshow(raw_mat, cmap='Greys', interpolation='none')
    axs[0].set_title('Spatial Raw (Hardware)', fontweight='bold', pad=10)
    
    # We use Blue/Green here to map back to the AI vs NIST methodology color scheme for this specific plot
    cmap_nist = plt.cm.colors.ListedColormap([WHITE, DELOITTE_BLUE])
    axs[1].imshow(nist_mat, cmap=cmap_nist, interpolation='none')
    axs[1].set_title('Spatial NIST (Standard)', fontweight='bold', pad=10)
    
    cmap_ai = plt.cm.colors.ListedColormap([WHITE, DELOITTE_GREEN])
    axs[2].imshow(ai_mat, cmap=cmap_ai, interpolation='none')
    axs[2].set_title('Spatial AI (Deep Learning)', fontweight='bold', pad=10)

    plt.suptitle('Bitstream Canvas Representation (First 10,000 bits)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plot_3_visual_canvas.png', dpi=400, bbox_inches='tight')
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
        print("\n-> All data loaded. Generating compact comparative plots...")
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
        print("\n[ERROR] Missing files. Found", len(data), "but expected 6.")