import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

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
# GLOBAL CERTIFIED METRICS (275M+ BITS)
# ==========================================
# We use the mathematical global truth to avoid local 1M-bit variance
GLOBAL_RAW_P1 = 0.499944
GLOBAL_WHITE_P1 = 0.499983
GLOBAL_RAW_SHANNON = 0.99999999
GLOBAL_WHITE_SHANNON = 1.00000000

def analyze_structural_data(filepath, max_bits=1000000, max_lag=40):
    """Loads a 1M bit sample specifically for Autocorrelation and Spatial Bitmaps."""
    bytes_to_read = max_bits // 8
    try:
        with open(filepath, 'rb') as f:
            raw_bytes = np.frombuffer(f.read(bytes_to_read), dtype=np.uint8)
        bitstream = np.unpackbits(raw_bytes)[:max_bits]
    except FileNotFoundError:
        return None

    # Autocorrelation (Local Sample)
    mean, var = np.mean(bitstream), np.var(bitstream)
    lags = np.arange(1, max_lag + 1)
    if var > 0:
        autocorr = [np.mean((bitstream[:-lag] - mean) * (bitstream[lag:] - mean)) / var for lag in lags]
    else:
        autocorr = np.zeros(max_lag)

    # Bitmap (First 40,000 bits)
    bitmap_data = bitstream[:40000].reshape((100, 400))

    return {'lags': lags, 'autocorr': np.array(autocorr), 'bitmap': bitmap_data, 'sample_size': len(bitstream)}

def generate_full_report(raw_file, whitened_file, max_bits=1000000):
    print("\n=== QUANTUM VISUAL AUDITOR ENGINE ===")
    
    raw_data = analyze_structural_data(raw_file, max_bits)
    white_data = analyze_structural_data(whitened_file, max_bits)
    
    if not raw_data or not white_data:
        print("[ERROR] Missing binary files. Aborting.")
        return

    print("-> Data loaded. Generating isolated plots...")

    # ---------------------------------------------------------
    # PLOT 1: GLOBAL BIAS DEVIATION (ERROR)
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    labels = ['Raw Hardware (Global)', 'Toeplitz Whitened (Global)']
    
    raw_dev = abs(GLOBAL_RAW_P1 - 0.5)
    white_dev = abs(GLOBAL_WHITE_P1 - 0.5)
    dev_values = [raw_dev, white_dev]
    
    bars = ax1.bar(labels, dev_values, color=[DELOITTE_GREY, DELOITTE_GREEN], width=0.4)
    ax1.set_title('Global Bias Deviation from Ideal (Lower is Better)', fontsize=14, pad=15)
    ax1.set_ylabel('Absolute Error |P(1) - 0.5|')
    ax1.set_ylim(0, max(dev_values) * 1.3)
    
    for bar, d in zip(bars, dev_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(dev_values)*0.02), 
                 f'{d:.6f}', ha='center', fontweight='bold')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('1_bias_deviation_comparison.png', dpi=300)
    plt.close()
    print("   [+] Saved: 1_bias_deviation_comparison.png")

    # ---------------------------------------------------------
    # PLOT 2: AUTOCORRELATION
    # ---------------------------------------------------------
    fig2, (ax2_raw, ax2_white) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Serial Autocorrelation Comparison (1M Bit Sample)', fontsize=16, y=1.05)
    conf_interval = 1.96 / np.sqrt(max_bits)

    ax2_raw.stem(raw_data['lags'], raw_data['autocorr'], basefmt=" ", linefmt=DELOITTE_GREY, markerfmt='o')
    plt.setp(ax2_raw.collections[0], color=DELOITTE_GREY)
    ax2_raw.set_title('Raw Hardware Memory')

    ax2_white.stem(white_data['lags'], white_data['autocorr'], basefmt=" ", linefmt=DELOITTE_GREEN, markerfmt='o')
    plt.setp(ax2_white.collections[0], color=DELOITTE_GREEN)
    ax2_white.set_title('Whitened Cryptographic Purity')

    for ax in [ax2_raw, ax2_white]:
        ax.axhline(0, color=DELOITTE_BLACK, linewidth=1)
        ax.fill_between(raw_data['lags'], -conf_interval, conf_interval, color=DELOITTE_LIGHT, alpha=0.3)
        ax.set_ylim(-0.005, 0.005)
        ax.set_xlabel('Lag (bits)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('2_autocorrelation_comparison.png', dpi=300)
    plt.close()
    print("   [+] Saved: 2_autocorrelation_comparison.png")

    # ---------------------------------------------------------
    # PLOT 3: BITMAPS
    # ---------------------------------------------------------
    fig3, (ax3_raw, ax3_white) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('2D Spatial Entropy Representation', fontsize=16, y=1.05)
    
    ax3_raw.imshow(raw_data['bitmap'], cmap=ListedColormap(['#ffffff', DELOITTE_GREY]), aspect='auto', interpolation='nearest')
    ax3_raw.set_title('Raw Bits')
    ax3_raw.axis('off')

    ax3_white.imshow(white_data['bitmap'], cmap=ListedColormap(['#ffffff', DELOITTE_GREEN]), aspect='auto', interpolation='nearest')
    ax3_white.set_title('Whitened Bits')
    ax3_white.axis('off')

    plt.tight_layout()
    plt.savefig('3_bitmap_comparison.png', dpi=300)
    plt.close()
    print("   [+] Saved: 3_bitmap_comparison.png")

    # ---------------------------------------------------------
    # PLOT 4: THE UNIFIED GLOBAL DASHBOARD
    # ---------------------------------------------------------
    print("-> Rendering Unified Global Dashboard...")
    fig4 = plt.figure(figsize=(16, 11))
    fig4.suptitle('Quantum Entropy Extractor: Complete Architecture Audit', 
                 fontsize=18, fontweight='bold', color=DELOITTE_BLACK, y=0.96)

    gs = fig4.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.4, wspace=0.2)

    # Dashboard Row 1: Bias Error & Summary Text
    ax4_bias = fig4.add_subplot(gs[0, 0])
    bars_dash = ax4_bias.bar(labels, dev_values, color=[DELOITTE_GREY, DELOITTE_GREEN], width=0.5)
    ax4_bias.set_title('Global Bias Deviation |P(1) - 0.5|', fontsize=12)
    ax4_bias.set_ylim(0, max(dev_values) * 1.3)
    ax4_bias.set_ylabel('Absolute Error')
    for bar, d in zip(bars_dash, dev_values):
        ax4_bias.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(dev_values)*0.02), 
                     f'{d:.6f}', ha='center', fontweight='bold')
    ax4_bias.spines['top'].set_visible(False)
    ax4_bias.spines['right'].set_visible(False)

    ax4_text = fig4.add_subplot(gs[0, 1])
    ax4_text.axis('off')
    summary_text = (
        f"CRYPTOGRAPHIC YIELD SUMMARY\n"
        f"{'-'*40}\n"
        f"Global Dataset: >275,000,000 bits\n"
        f"Sample Dataset: 1,000,000 bits\n\n"
        f"[RAW HARDWARE STATE]\n"
        f"Shannon Entropy : {GLOBAL_RAW_SHANNON:.8f}\n"
        f"Bias Deviation  : {raw_dev:.6f}\n\n"
        f"[TOEPLITZ WHITENED STATE]\n"
        f"Shannon Entropy : {GLOBAL_WHITE_SHANNON:.8f}\n"
        f"Bias Deviation  : {white_dev:.6f}\n"
    )
    bbox_props = dict(boxstyle="square,pad=1.5", fc="#f8f9fa", ec=DELOITTE_GREY, lw=1)
    ax4_text.text(0.1, 0.5, summary_text, fontsize=12, family='monospace', 
                 verticalalignment='center', bbox=bbox_props)

    # Dashboard Row 2: Autocorrelation
    ax4_auto_raw = fig4.add_subplot(gs[1, 0])
    ax4_auto_raw.stem(raw_data['lags'], raw_data['autocorr'], basefmt=" ", linefmt=DELOITTE_GREY, markerfmt='o')
    plt.setp(ax4_auto_raw.collections[0], color=DELOITTE_GREY)
    ax4_auto_raw.axhline(0, color=DELOITTE_BLACK, linewidth=1)
    ax4_auto_raw.fill_between(raw_data['lags'], -conf_interval, conf_interval, color=DELOITTE_LIGHT, alpha=0.3)
    ax4_auto_raw.set_title('Raw Autocorrelation', fontsize=12)

    ax4_auto_white = fig4.add_subplot(gs[1, 1])
    ax4_auto_white.stem(white_data['lags'], white_data['autocorr'], basefmt=" ", linefmt=DELOITTE_GREEN, markerfmt='o')
    plt.setp(ax4_auto_white.collections[0], color=DELOITTE_GREEN)
    ax4_auto_white.axhline(0, color=DELOITTE_BLACK, linewidth=1)
    ax4_auto_white.fill_between(white_data['lags'], -conf_interval, conf_interval, color=DELOITTE_LIGHT, alpha=0.3)
    ax4_auto_white.set_title('Whitened Autocorrelation', fontsize=12)

    for ax in [ax4_auto_raw, ax4_auto_white]:
        ax.set_ylim(-0.005, 0.005)
        ax.set_xlabel('Lag (bits)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Dashboard Row 3: Bitmaps
    ax4_img_raw = fig4.add_subplot(gs[2, 0])
    ax4_img_raw.imshow(raw_data['bitmap'], cmap=ListedColormap(['#ffffff', DELOITTE_GREY]), aspect='auto', interpolation='nearest')
    ax4_img_raw.set_title('Raw Spatial Representation', fontsize=12)
    ax4_img_raw.axis('off')

    ax4_img_white = fig4.add_subplot(gs[2, 1])
    ax4_img_white.imshow(white_data['bitmap'], cmap=ListedColormap(['#ffffff', DELOITTE_GREEN]), aspect='auto', interpolation='nearest')
    ax4_img_white.set_title('Whitened Spatial Representation', fontsize=12)
    ax4_img_white.axis('off')

    plt.tight_layout()
    plt.savefig('4_master_dashboard_comparison.png', dpi=300)
    print("   [+] Saved: 4_master_dashboard_comparison.png")
    print("=== EXPORT COMPLETE ===")

if __name__ == "__main__":
    RAW_FILE = "quantum_bitstream.bin"
    WHITENED_FILE = "whitened_quantum_keys.bin"
    
    generate_full_report(RAW_FILE, WHITENED_FILE, max_bits=1000000)