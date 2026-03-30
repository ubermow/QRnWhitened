import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ==========================================
# DELOITTE CORPORATE AESTHETIC CONFIGURATION
# ==========================================
DELOITTE_BLUE  = '#0097A9'  
DELOITTE_GREEN = '#86BC25'  
DELOITTE_RED   = '#DA291C'  
DELOITTE_DARK  = '#2A2E33'  
WHITE          = '#FFFFFF'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': DELOITTE_DARK,
    'text.color': DELOITTE_DARK,
    'figure.facecolor': WHITE,
    'axes.facecolor': WHITE,
    'axes.titlesize': 15,
    'axes.labelsize': 12,
    'legend.frameon': False,
})

def despine_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')

# ==========================================
# HARDCODED DATA FROM 30H TERMINAL OUTPUT
# ==========================================
# Extracted directly from your provided logs
t_val = [0.69320203, 0.69316026, 0.69317895, 0.69321331, 0.69318899, 0.69314607, 
         0.69314512, 0.69316044, 0.69314456, 0.69314621, 0.69314426, 0.69316065, 
         0.69314951, 0.69314943, 0.69314961, 0.69314766, 0.69315552, 0.69315794, 
         0.69315137, 0.69314420, 0.69314556, 0.69314538, 0.69314747, 0.69314909, 
         0.69314839, 0.69314858]

s_val = [0.69012447, 0.69013542, 0.69016584, 0.69012762, 0.69014707, 0.69011276, 
         0.69011527, 0.69011108, 0.69011157, 0.69011239, 0.69012909, 0.69012639, 
         0.69011107, 0.69011114, 0.69011157, 0.69011264, 0.69011203, 0.69011620, 
         0.69011255, 0.69011436, 0.69011382, 0.69011483, 0.69011513, 0.69011249, 
         0.69011237, 0.69011270]

# Metrics calculated from the final loss values
t_min_ent = 0.9999  # Derived from ~0.693144
s_min_ent = 0.8943  # Derived from ~0.690111
t_ratio = 0.9989    # Assuming 128-bit penalty on 131072 block
s_ratio = 0.8933


# ==========================================
# PLOT 5: UPGRADED CONVERGENCE WITH DUAL INSETS
# ==========================================
def plot_upgraded_convergence():
    epochs = np.arange(1, len(t_val) + 1)
    theoretical_limit = 0.693147
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Main plot (Macro view)
    ax.axhline(theoretical_limit, color=DELOITTE_DARK, linestyle='--', lw=1.5, label='Theoretical Guessing Limit (0.6931)', alpha=0.7)
    ax.plot(epochs, t_val, marker='o', color=DELOITTE_GREEN, lw=2.5, markersize=6, label='Temporal Validation Loss')
    ax.plot(epochs, s_val, marker='s', color=DELOITTE_BLUE, lw=2.5, markersize=6, label='Spatial Validation Loss')
    
    # Give the title plenty of padding to accommodate the legend above it
    ax.set_title('Adversarial AI Convergence with Dual Micro-Variance Insets', pad=40, fontweight='bold', color=DELOITTE_DARK)
    ax.set_xlabel('Training Epochs', fontweight='bold', labelpad=10)
    ax.set_ylabel('Binary Cross-Entropy Loss', fontweight='bold', labelpad=10)
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, prop={'size': 11, 'weight': 'bold'})
    despine_ax(ax)
    ax.grid(axis='y', alpha=0.15, linestyle='-')
    
    # Lock the main y-axis limits to ensure data doesn't collide with the insets
    ax.set_ylim(0.6890, 0.6950)
    
    # --------------------------------------------------
    # INSET 1: Temporal Zoom (Epochs 16-26)
    # --------------------------------------------------
    # Coordinates: [left, bottom, width, height] as fractions of the figure
    ax_inset_t = fig.add_axes([0.62, 0.55, 0.26, 0.20]) 
    ax_inset_t.plot(epochs[15:], t_val[15:], marker='o', color=DELOITTE_GREEN, lw=2, markersize=4)
    ax_inset_t.axhline(theoretical_limit, color=DELOITTE_DARK, linestyle='--', lw=1.0, alpha=0.7)
    ax_inset_t.set_title("Temporal Zoom ($10^{-5}$ scale)", fontsize=10, fontweight='bold')
    ax_inset_t.tick_params(axis='both', labelsize=8)
    ax_inset_t.grid(True, alpha=0.2, linestyle=':')
    
    # --------------------------------------------------
    # INSET 2: Spatial Zoom (Epochs 16-26)
    # --------------------------------------------------
    ax_inset_s = fig.add_axes([0.62, 0.20, 0.26, 0.20])
    ax_inset_s.plot(epochs[15:], s_val[15:], marker='s', color=DELOITTE_BLUE, lw=2, markersize=4)
    ax_inset_s.set_title("Spatial Zoom ($10^{-5}$ scale)", fontsize=10, fontweight='bold')
    ax_inset_s.tick_params(axis='both', labelsize=8)
    ax_inset_s.grid(True, alpha=0.2, linestyle=':')
    
    # DO NOT use tight_layout() here to prevent coordinate shifting.
    # bbox_inches='tight' in savefig will handle the outer margins safely.
    plt.savefig('plot_5_convergence_upgraded.png', dpi=400, bbox_inches='tight')
    plt.close()

# ==========================================
# PLOT 6: UPGRADED SAFETY MARGIN (TRUNCATED Y-AXIS ZOOM)
# ==========================================
def plot_upgraded_safety_margin():
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['Raw Shannon\nEntropy', 'AI Certified\nMin-Entropy', 'Toeplitz Ratio\n(LHL)']
    
    # Exact data from 30h run
    t_vals = [1.0000, 0.9999, 0.9989]
    s_vals = [0.9956, 0.8943, 0.8933]
    
    x = np.arange(len(labels))
    width = 0.5
    
    # --- LEFT SUBPLOT: TEMPORAL ZOOM ---
    bars_t = axs[0].bar(x, t_vals, width, color=DELOITTE_GREEN, alpha=0.9, edgecolor=WHITE, lw=1.5)
    axs[0].set_title('Temporal Domain Yield\n(Micro-Variance Zoom)', fontweight='bold', color=DELOITTE_DARK, pad=15)
    axs[0].set_ylabel('Information (Bits / Raw Bit)', fontweight='bold', labelpad=10)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, fontweight='bold', fontsize=11)
    
    # Cut the bottom of the bins to magnify the 0.001 drop
    axs[0].set_ylim(0.9980, 1.0005) 
    
    # --- RIGHT SUBPLOT: SPATIAL ZOOM ---
    bars_s = axs[1].bar(x, s_vals, width, color=DELOITTE_BLUE, alpha=0.9, edgecolor=WHITE, lw=1.5)
    axs[1].set_title('Spatial Domain Yield\n(Macro-Variance Zoom)', fontweight='bold', color=DELOITTE_DARK, pad=15)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, fontweight='bold', fontsize=11)
    
    # Cut the bottom of the bins to magnify the 0.100 drop
    axs[1].set_ylim(0.8850, 1.0000)
    
    # --- ANNOTATIONS AND AESTHETICS ---
    text_box_style = dict(boxstyle="round,pad=0.2", facecolor=WHITE, edgecolor="none", alpha=0.9)
    
    def add_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 6), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color=DELOITTE_DARK, bbox=text_box_style)

    add_labels(axs[0], bars_t)
    add_labels(axs[1], bars_s)
    
    for ax in axs:
        despine_ax(ax)
        ax.grid(axis='y', alpha=0.2, linestyle=':')

    # Add a subtle reference line to show where the LHL 128-bit penalty drops from
    axs[0].axhline(y=t_vals[1], color=DELOITTE_RED, linestyle='--', alpha=0.4, lw=1.5)
    axs[1].axhline(y=s_vals[1], color=DELOITTE_RED, linestyle='--', alpha=0.4, lw=1.5)

    plt.suptitle('Cryptographic Yield vs. Adversarial Predictability (AI Audit)', fontsize=18, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.savefig('plot_6_safety_margin_ai.png', dpi=400, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_upgraded_convergence()
    plot_upgraded_safety_margin()  # <--- Updated execution call
    print("Upgraded plots generated successfully.")