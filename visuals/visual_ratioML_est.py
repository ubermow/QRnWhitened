import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# DELOITTE CORPORATE AESTHETIC CONFIGURATION
# ==========================================
DELOITTE_GREEN = '#86BC25'
DELOITTE_BLACK = '#000000'
DELOITTE_GREY  = '#53565A'
WARNING_RED    = '#DA291C' 
DELOITTE_BLUE  = '#0097A9' 

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': DELOITTE_GREY,
    'axes.linewidth': 1.0,
    'text.color': DELOITTE_BLACK,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# ==========================================
# PLOT 1: AI LEARNING CURVE COMPARISON
# ==========================================
def plot_learning_curves(temporal_losses, spatial_losses):
    epochs = np.arange(1, len(temporal_losses) + 1)
    theoretical_limit = 0.693147 # -ln(0.5)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.axhline(theoretical_limit, color=DELOITTE_BLACK, linestyle='--', linewidth=2, 
               label='Theoretical Guessing Limit (0.6931)', alpha=0.7)
    
    ax.plot(epochs, temporal_losses, marker='o', color=DELOITTE_GREEN, linewidth=2.5, 
            markersize=8, label='Temporal Bitstream Loss')
            
    ax.plot(epochs, spatial_losses, marker='s', color=WARNING_RED, linewidth=2.5, 
            markersize=8, label='Spatial Bitstream Loss')

    ax.set_title('Adversarial AI Convergence (LSTM Loss Landscape)', pad=15)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Binary Cross-Entropy Loss')
    
    min_loss = min(min(temporal_losses), min(spatial_losses))
    max_loss = max(max(temporal_losses), max(spatial_losses))
    buffer = (max_loss - min_loss) * 0.5 if max_loss != min_loss else 0.005
    
    ax.set_ylim(min_loss - buffer, max_loss + buffer)
    ax.set_xticks(epochs)
    
    ax.legend(loc='best', frameon=True, edgecolor=DELOITTE_GREY)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    filename = 'plot_5_ai_convergence.png'
    plt.savefig(filename, dpi=300)
    print(f"-> Saved: {filename}")
    plt.close()

# ==========================================
# PLOT 2: CRYPTOGRAPHIC SAFETY MARGIN
# ==========================================
def plot_safety_margin(shannon_t, min_ent_t, ratio_t, shannon_s, min_ent_s, ratio_s):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['Raw Shannon Entropy', 'AI Certified Min-Entropy', 'Toeplitz Ratio (LHL)']
    
    temporal_vals = [shannon_t, min_ent_t, ratio_t]
    spatial_vals = [shannon_s, min_ent_s, ratio_s]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars_t = ax.bar(x - width/2, temporal_vals, width, label='Temporal System', color=DELOITTE_GREEN)
    bars_s = ax.bar(x + width/2, spatial_vals, width, label='Spatial System', color=DELOITTE_BLUE)
    
    ax.set_title('Cryptographic Yield vs. Adversarial Predictability', pad=20)
    ax.set_ylabel('Information (Bits / Raw Bit)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    y_min = min(min(temporal_vals), min(spatial_vals)) - 0.05
    ax.set_ylim(max(0, y_min), 1.05)
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
    add_labels(bars_t)
    add_labels(bars_s)
    
    # 1. Tracciamo le linee orizzontali di riferimento dalla Min-Entropia
    ax.axhline(y=min_ent_t, color=DELOITTE_GREEN, linestyle='--', alpha=0.3, linewidth=1.2)
    ax.axhline(y=min_ent_s, color=DELOITTE_BLUE, linestyle='--', alpha=0.3, linewidth=1.2)
    
    # 2. Creiamo le "Quote Verticali" eleganti e discrete per evidenziare i 128 bit decurtati
    # Usa linee sottili e trasparenti, testo molto piccolo e discreto
    
    # Quota Temporale (sopra la barra del Ratio Temporale) - posizionata a sinistra
    x_ratio_t = x[2] - width/2 - 0.15
    ax.plot([x_ratio_t, x_ratio_t], [min_ent_t, ratio_t], color=DELOITTE_GREEN, 
            linewidth=0.8, alpha=0.4, linestyle='-')
    # Piccoli segmenti orizzontali alle estremità
    ax.plot([x_ratio_t - 0.05, x_ratio_t + 0.05], [min_ent_t, min_ent_t], 
            color=DELOITTE_GREEN, linewidth=0.8, alpha=0.4)
    ax.plot([x_ratio_t - 0.05, x_ratio_t + 0.05], [ratio_t, ratio_t], 
            color=DELOITTE_GREEN, linewidth=0.8, alpha=0.4)
    ax.text(x_ratio_t - 0.12, (ratio_t + min_ent_t)/2, '−128b',
            ha='right', va='center', color=DELOITTE_GREEN, fontweight='normal', fontsize=7,
            alpha=0.6)

    # Quota Spaziale (sopra la barra del Ratio Spaziale) - posizionata a destra
    x_ratio_s = x[2] + width/2 + 0.15
    ax.plot([x_ratio_s, x_ratio_s], [min_ent_s, ratio_s], color=DELOITTE_BLUE, 
            linewidth=0.8, alpha=0.4, linestyle='-')
    # Piccoli segmenti orizzontali alle estremità
    ax.plot([x_ratio_s - 0.05, x_ratio_s + 0.05], [min_ent_s, min_ent_s], 
            color=DELOITTE_BLUE, linewidth=0.8, alpha=0.4)
    ax.plot([x_ratio_s - 0.05, x_ratio_s + 0.05], [ratio_s, ratio_s], 
            color=DELOITTE_BLUE, linewidth=0.8, alpha=0.4)
    ax.text(x_ratio_s + 0.12, (ratio_s + min_ent_s)/2, '−128b',
            ha='left', va='center', color=DELOITTE_BLUE, fontweight='normal', fontsize=7,
            alpha=0.6)

    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filename = 'plot_6_safety_margin.png'
    plt.savefig(filename, dpi=300)
    print(f"-> Saved: {filename}")
    plt.close()

# ==========================================
# MAIN EXECUTION (DATA INJECTION)
# ==========================================
if __name__ == "__main__":
    print("\n=== GENERATING DYNAMIC ML AUDIT VISUALS ===")
    
    t_losses = [0.6930, 0.6933, 0.6930, 0.6927, 0.6931]
    s_losses = [0.6908, 0.6962, 0.6843, 0.6878, 0.6960]
    
    t_shannon = 0.99999987  
    t_min_ent = 1.000000    
    t_ratio = 1920 / 2048   # Strict LHL: (2048 * 1.0 - 128) / 2048 = 0.9375
    
    s_shannon = 0.99634314  
    s_min_ent = 0.893860    
    s_ratio = 1702 / 2048   # Strict LHL: (2048 * 0.893860 - 128) / 2048 = 0.8310
    
    plot_learning_curves(t_losses, s_losses)
    plot_safety_margin(t_shannon, t_min_ent, t_ratio, s_shannon, s_min_ent, s_ratio)
    
    print("=== VISUALIZATION COMPLETE ===\n")