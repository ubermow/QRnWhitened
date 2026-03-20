import numpy as np
import os
import time
import matplotlib.pyplot as plt

# ==========================================
# STATISTICAL MIN-ENTROPY EVALUATOR
# ==========================================
def calculate_chunk_entropy(bitstream_chunk):
    n = len(bitstream_chunk)
    if n < 1000: return None

    count_1 = np.sum(bitstream_chunk)
    count_0 = n - count_1
    p_guess_iid = max(count_0 / n, count_1 / n)
    h_min_iid = -np.log2(p_guess_iid)

    x_current = bitstream_chunk[:-1]
    x_next = bitstream_chunk[1:]
    
    idx_0 = (x_current == 0)
    idx_1 = (x_current == 1)
    
    count_current_0 = np.sum(idx_0)
    count_current_1 = np.sum(idx_1)
    
    p_00 = np.sum(x_next[idx_0] == 0) / count_current_0 if count_current_0 > 0 else 0
    p_01 = np.sum(x_next[idx_0] == 1) / count_current_0 if count_current_0 > 0 else 0
    p_10 = np.sum(x_next[idx_1] == 0) / count_current_1 if count_current_1 > 0 else 0
    p_11 = np.sum(x_next[idx_1] == 1) / count_current_1 if count_current_1 > 0 else 0
    
    p_guess_markov = max(p_00, p_01, p_10, p_11)
    h_min_markov = -np.log2(p_guess_markov)

    return min(h_min_iid, h_min_markov)

def process_file_in_chunks(filepath, chunk_size=5_000_000):
    print(f"\n-> Loading raw bitstream from: {os.path.basename(filepath)}")
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return []

    with open(filepath, 'rb') as f:
        raw_bytes = np.fromfile(f, dtype=np.uint8)
    
    full_bitstream = np.unpackbits(raw_bytes)
    total_bits = len(full_bitstream)
    num_chunks = total_bits // chunk_size
    
    if num_chunks == 0:
        return []

    print(f"-> Processing {total_bits:,} bits into {num_chunks} chunks of {chunk_size:,} bits...")

    entropies = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = full_bitstream[start_idx:end_idx]
        
        h_min = calculate_chunk_entropy(chunk)
        if h_min is not None:
            entropies.append(h_min)

    worst_case = np.min(entropies)
    mean_val = np.mean(entropies)
    std_val = np.std(entropies)
    
    print(f"   Completed. \n   Mean (\u03bc): {mean_val:.6f} | Std Dev (\u03c3): {std_val:.6f} | Absolute Worst: {worst_case:.6f}")
    return entropies

# ==========================================
# VISUALIZATION BLOCK (EDITORIAL AESTHETIC)
# ==========================================
def plot_entropy_stability(entropies_t, entropies_s):
    chunks_count = min(len(entropies_t), len(entropies_s))
    x = np.arange(1, chunks_count + 1)
    
    t_data = entropies_t[:chunks_count]
    s_data = entropies_s[:chunks_count]

    mean_t, std_t = np.mean(t_data), np.std(t_data)
    mean_s, std_s = np.mean(s_data), np.std(s_data)
    
    worst_t = np.min(t_data)
    worst_s = np.min(s_data)
    overall_worst = min(worst_t, worst_s)

    # Elegant Color Palette
    deloitte_green = '#86BC25'
    dark_slate = '#2A2E33'
    light_grey = '#EAEAEA'
    title_color = '#1A1A1A'
    text_grey = '#5A5A5A'
    accent_red = '#E34053'

    # Set up figure
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(16, 7), facecolor='#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Main Data Lines (Smooth and clean)
    ax.plot(x, t_data, color=dark_slate, linewidth=2.0, alpha=0.9,
            label=f'Temporal Dimension  (\u03bc={mean_t:.4f}, \u03c3={std_t:.4f})', zorder=3)
    ax.plot(x, s_data, color=deloitte_green, linewidth=2.0, alpha=0.9,
            label=f'Spatial Dimension     (\u03bc={mean_s:.4f}, \u03c3={std_s:.4f})', zorder=3)

    # Subtle Mean Lines
    ax.axhline(y=mean_t, color=dark_slate, linestyle=':', linewidth=1.0, alpha=0.4, zorder=2)
    ax.axhline(y=mean_s, color=deloitte_green, linestyle=':', linewidth=1.0, alpha=0.4, zorder=2)

    # Highlight points (softer edges)
    idx_worst_t = np.argmin(t_data)
    idx_worst_s = np.argmin(s_data)
    ax.scatter(idx_worst_t + 1, worst_t, color=accent_red, s=70, edgecolor='white', linewidth=1.5, zorder=5)
    ax.scatter(idx_worst_s + 1, worst_s, color=accent_red, s=70, edgecolor='white', linewidth=1.5, zorder=5)

    # Minimalist Axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')

    ax.yaxis.grid(True, linestyle='-', alpha=0.5, color=light_grey)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Elegant Typography
    ax.set_title('Quantum Hardware Stability: Raw Min-Entropy Tracking', 
                 fontsize=17, color=title_color, weight='500', pad=30, loc='left')
    ax.set_ylabel('Certified Min-Entropy (bits/bit)', fontsize=11, color=text_grey, labelpad=15)
    ax.set_xlabel('Time Progression (Chunk Index)', fontsize=11, color=text_grey, labelpad=15)
    
    ax.tick_params(axis='x', bottom=True, color='#CCCCCC', labelcolor=text_grey, labelsize=10, pad=5)
    ax.tick_params(axis='y', left=False, labelcolor=text_grey, labelsize=10, pad=5)
    
    # Dynamic Boundaries
    y_min = overall_worst - (overall_worst * 0.005)
    y_max = max(np.max(t_data), np.max(s_data)) + (overall_worst * 0.005)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, chunks_count + 1)

    # Worst-Case Line & Protected Annotation
    ax.axhline(y=overall_worst, color=accent_red, linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)
    
    # White bounding box around text prevents ANY overlap clashes with data lines
    text_box_style = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85)
    ax.text(chunks_count - 0.2, overall_worst + (overall_worst * 0.0003), 
            f'Global Worst-Case: {overall_worst:.4f}', 
            color=accent_red, fontsize=10, ha='right', va='bottom', bbox=text_box_style, zorder=6)

    # Refined Legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='white', 
              fontsize=11, labelcolor=dark_slate, borderpad=1)

    fig.tight_layout()

    # Automatically save high-res version for presentations
    save_path = "min_entropy_stability_report.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] High-resolution chart automatically saved to: {save_path}")

    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    FILE_TEMPORAL_RAW = r"data\\raw\\3hours_nopeople\\temporal_3hraw_bitstream.bin"
    FILE_SPATIAL_RAW  = r"data\\raw\\3hours_nopeople\\spatial_3hraw_bitstream.bin"

    CHUNK_SIZE = 5_000_000

    start_time = time.time()
    print("\n--- INITIATING MIN-ENTROPY AUDIT (TEMPORAL) ---")
    entropies_t = process_file_in_chunks(FILE_TEMPORAL_RAW, chunk_size=CHUNK_SIZE)
    
    print("\n--- INITIATING MIN-ENTROPY AUDIT (SPATIAL) ---")
    entropies_s = process_file_in_chunks(FILE_SPATIAL_RAW, chunk_size=CHUNK_SIZE)

    if entropies_t and entropies_s:
        print("\n-> Generating and saving aesthetic stability visualization...")
        plot_entropy_stability(entropies_t, entropies_s)