import matplotlib.pyplot as plt
import numpy as np
import os
import re

def parse_nist_report(filepath):
    """Parses the official NIST C-Suite finalAnalysisReport.txt for Uniformity P-Values"""
    print(f"-> Parsing {os.path.basename(filepath)}...")
    if not os.path.exists(filepath):
        print(f"[ERROR] Could not find {filepath}")
        return None

    results = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            if re.search(r'\d+/\d+', line) and len(line.strip()) > 50:
                parts = line.strip().split()
                try:
                    test_name = parts[-1]
                    prop_str = parts[-2]
                    # This captures the Uniformity P-Value (the 'P-VALUE' column)
                    uniformity_p_val = float(parts[-3])
                    
                    passes, total = map(int, prop_str.split('/'))
                    proportion = passes / total
                    
                    if test_name not in results:
                        results[test_name] = {'proportions': [], 'p_values': []}
                    
                    results[test_name]['proportions'].append(proportion)
                    results[test_name]['p_values'].append(uniformity_p_val)
                except ValueError:
                    continue

    final_stats = {}
    for test, data in results.items():
        # Averages the proportions and uniformity p-values for multi-row tests
        final_stats[test] = {
            'avg_prop': np.mean(data['proportions']),
            'avg_p': np.mean(data['p_values'])
        }
        
    return final_stats

def generate_wang_deloitte_plot(temporal_data, spatial_data):
    """
    Generates an elegant two-panel NIST validation dashboard.
    Each panel (Temporal/Spatial) shows Pass Proportion vs Average P-Value independently.
    Scientifically rigorous while maintaining contemporary Deloitte aesthetic.
    """
    print("-> Generating Deloitte-compliant Dual-Dimension Dashboard...")
    
    # Isolate common tests
    common_tests = [t for t in temporal_data.keys() if t in spatial_data.keys()]
    x = np.arange(len(common_tests))
    width = 0.35

    # Extract data
    t_props = [temporal_data[t]['avg_prop'] for t in common_tests]
    s_props = [spatial_data[t]['avg_prop'] for t in common_tests]
    t_avgs = [temporal_data[t]['avg_p'] for t in common_tests]
    s_avgs = [spatial_data[t]['avg_p'] for t in common_tests]

    # Deloitte Brand Palette - Contemporary & Professional
    deloitte_green = '#86BC25'
    deloitte_blue = '#0097A9'
    dark_slate = '#2A2E33'
    accent_red = '#DA291C'
    light_grey = '#E8E8E8'
    bg_light = '#FAFAFA'
    text_grey = '#4A4A4A'

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.linewidth': 0.8,
    })

    # Create figure with two side-by-side subplots
    fig = plt.figure(figsize=(17, 8), facecolor='white')
    fig.suptitle('Quantum Randomness Validation: NIST SP 800-22 Analysis', 
                 fontsize=18, color=dark_slate, weight='bold', y=0.98)

    # Create grid for better control
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, top=0.92, bottom=0.12, left=0.07, right=0.97)
    
    # --- TEMPORAL PANEL (Left) ---
    ax_t_prop = fig.add_subplot(gs[0, 0])
    ax_t_pval = fig.add_subplot(gs[1, 0])
    
    # --- SPATIAL PANEL (Right) ---
    ax_s_prop = fig.add_subplot(gs[0, 1])
    ax_s_pval = fig.add_subplot(gs[1, 1])

    # Helper function to style axes
    def style_ax(ax):
        ax.set_facecolor(bg_light)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(light_grey)
        ax.spines['bottom'].set_color(light_grey)
        ax.yaxis.grid(True, linestyle='-', alpha=0.5, color=light_grey, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=8.5, colors=text_grey, labelcolor=text_grey)

    # --- TEMPORAL TOP: Pass Proportions ---
    style_ax(ax_t_prop)
    bars_t_prop = ax_t_prop.bar(x, t_props, width*1.2, color=dark_slate, alpha=0.85, 
                                edgecolor=light_grey, linewidth=0.6)
    ax_t_prop.set_ylabel('Pass Proportion', fontsize=10, color=text_grey, weight='500', labelpad=10)
    ax_t_prop.set_ylim(0, 1.15)
    ax_t_prop.axhline(y=0.96, color=accent_red, linestyle='--', linewidth=1.8, alpha=0.6, zorder=1)
    ax_t_prop.text(-0.5, 0.925, 'Min. Threshold (0.96)', fontsize=8.5, color=accent_red, 
                   style='italic', weight='500', va='top')
    ax_t_prop.set_title('TEMPORAL DIMENSION', fontsize=14, color=dark_slate, weight='bold', pad=12, loc='left')
    ax_t_prop.set_xticks([])

    # --- TEMPORAL BOTTOM: P-Values (Log Scale) ---
    style_ax(ax_t_pval)
    bars_t_pval = ax_t_pval.bar(x, t_avgs, width*1.2, color=deloitte_blue, alpha=0.75, 
                                edgecolor=light_grey, linewidth=0.6)
    ax_t_pval.set_yscale('log')
    ax_t_pval.set_ylim(1e-5, 10)
    ax_t_pval.set_ylabel('Uniformity P-Value (Log)', fontsize=10, color=text_grey, weight='500', labelpad=10)
    ax_t_pval.axhline(y=0.01, color=accent_red, linestyle=':', linewidth=1.8, alpha=0.6, zorder=1)
    ax_t_pval.text(-0.5, 0.00015, 'Significance α=0.01', fontsize=8.5, color=accent_red, 
                   style='italic', weight='500', va='bottom')
    clean_labels = [re.sub(r'([A-Z])', r' \1', t).strip() for t in common_tests]
    ax_t_pval.set_xticks(x)
    ax_t_pval.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=8.5, color=text_grey)

    # --- SPATIAL TOP: Pass Proportions ---
    style_ax(ax_s_prop)
    bars_s_prop = ax_s_prop.bar(x, s_props, width*1.2, color=deloitte_green, alpha=0.85, 
                                edgecolor=light_grey, linewidth=0.6)
    ax_s_prop.set_ylabel('Pass Proportion', fontsize=10, color=text_grey, weight='500', labelpad=10)
    ax_s_prop.set_ylim(0, 1.15)
    ax_s_prop.axhline(y=0.96, color=accent_red, linestyle='--', linewidth=1.8, alpha=0.6, zorder=1)
    ax_s_prop.text(-0.5, 0.925, 'Min. Threshold (0.96)', fontsize=8.5, color=accent_red, 
                   style='italic', weight='500', va='top')
    ax_s_prop.set_title('SPATIAL DIMENSION', fontsize=14, color=deloitte_green, weight='bold', pad=12, loc='right')
    ax_s_prop.set_xticks([])

    # --- SPATIAL BOTTOM: P-Values (Log Scale) ---
    style_ax(ax_s_pval)
    bars_s_pval = ax_s_pval.bar(x, s_avgs, width*1.2, color='#5EC8C8', alpha=0.75, 
                                edgecolor=light_grey, linewidth=0.6)
    ax_s_pval.set_yscale('log')
    ax_s_pval.set_ylim(1e-5, 10)
    ax_s_pval.set_ylabel('Uniformity P-Value (Log)', fontsize=10, color=text_grey, weight='500', labelpad=10)
    ax_s_pval.axhline(y=0.01, color=accent_red, linestyle=':', linewidth=1.8, alpha=0.6, zorder=1)
    ax_s_pval.text(-0.5, 0.00015, 'Significance α=0.01', fontsize=8.5, color=accent_red, 
                   style='italic', weight='500', va='bottom')
    ax_s_pval.set_xticks(x)
    ax_s_pval.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=8.5, color=text_grey)

    # --- GLOBAL FOOTER NOTE ---
    #footer_text = (
     #   "Scientific Integrity: All tests employ NIST SP 800-22 Rev. 1a with uniformly distributed p-values as the primary validity indicator.\n"
      #  "Left (Temporal): Time-tagged photon arrivals. Right (Spatial): Quantum dot array positional detection. Both streams undergo Toeplitz bit extraction.\n"
     #   "Interpretation: High pass proportions + uniformly distributed p-values indicate cryptographically viable randomness sources."
    #)
   # fig.text(0.5, 0.02, footer_text, fontsize=8.5, color='#666666', ha='center', va='bottom',
   #           linespacing=1.7, style='italic',
   #           bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5', edgecolor='#DDDDDD',
   #                    linewidth=0.8, alpha=0.85))

    filename = "aiNIST_Dual_Dimension_Validation.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[+] High-resolution dual-panel visualization saved: {filename}")
    plt.show()

# ==========================================
if __name__ == "__main__":
    FILE_TEMPORAL = r"data\\whitened\\final_attempt\\nnNIST\\nn_temporal_finalAnalysisReport.txt"
    FILE_SPATIAL = r"data\\whitened\\final_attempt\\nnNIST\\nn_spatial_finalAnalysisReport.txt"

    data_t = parse_nist_report(FILE_TEMPORAL)
    data_s = parse_nist_report(FILE_SPATIAL)

    if data_t and data_s:
        generate_wang_deloitte_plot(data_t, data_s)