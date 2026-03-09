import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# DELOITTE CORPORATE AESTHETIC CONFIGURATION
# ==========================================
DELOITTE_GREEN = '#86BC25'
DELOITTE_BLACK = '#000000'
DELOITTE_GREY  = '#53565A'
DELOITTE_LIGHT = '#D0D0CE'
WARNING_RED    = '#DA291C' # Used sparingly for adversarial metrics

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
# CERTIFIED ML AUDIT METRICS
# (Extracted from 1,000,000-bit LSTM run)
# ==========================================
EPOCHS = [1, 2, 3, 4, 5]
# Loss values from your 5533-second run
LOSS_VALUES = [0.6931, 0.6927, 0.6930, 0.6930, 0.6945] 
THEORETICAL_LIMIT = 0.693147 # -ln(0.5)

SHANNON_ENTROPY = 1.000000
MIN_ENTROPY = 0.996930
# Toeplitz compression: m (2000) / n (2048)
TOEPLITZ_RATIO = 2000 / 2048 

def generate_ml_dashboard():
    print("\n=== GENERATING ML ENTROPY ESTIMATION REPORT ===")
    
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle('Adversarial AI Audit & Entropy Squeezing Ratio', 
                 fontsize=16, fontweight='bold', y=1.02)
                 
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

    # ---------------------------------------------------------
    # PLOT 1: AI LEARNING CURVE (LSTM LOSS)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    
    # Plot the theoretical limit of pure guessing
    ax1.axhline(THEORETICAL_LIMIT, color=DELOITTE_GREEN, linestyle='--', linewidth=2, 
                label='Theoretical Guessing Limit (0.6931)')
    
    # Plot the AI's actual performance
    ax1.plot(EPOCHS, LOSS_VALUES, marker='o', color=WARNING_RED, linewidth=2, 
             markersize=8, label='LSTM Training Loss')
    
    ax1.set_title('Neural Network Convergence Failure', fontsize=14, pad=15)
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    
    # Zoom in tightly to show the microscopic struggle of the AI
    ax1.set_ylim(0.6900, 0.6960)
    ax1.set_xticks(EPOCHS)
    
    ax1.legend(loc='lower right', frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---------------------------------------------------------
    # PLOT 2: THE CRYPTOGRAPHIC SQUEEZE (RATIO)
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    
    labels = ['Shannon (Raw)', 'Min-Entropy (AI)', 'Toeplitz Ratio']
    values = [SHANNON_ENTROPY, MIN_ENTROPY, TOEPLITZ_RATIO]
    colors = [DELOITTE_GREY, WARNING_RED, DELOITTE_GREEN]
    
    bars = ax2.bar(labels, values, color=colors, width=0.5)
    
    ax2.set_title('Entropy Ratio & Safety Margin', fontsize=14, pad=15)
    ax2.set_ylabel('Information per Bit (Bits)')
    
    # Extreme zoom to visualize the 0.0234 safety margin
    ax2.set_ylim(0.9700, 1.002) 
    
    # Add the text labels on top of the bars
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                 f'{v:.5f}', ha='center', fontweight='bold')
                 
    # Draw an arrow showing the safety margin
    ax2.annotate('Cryptographic\nSafety Margin', 
                 xy=(2, TOEPLITZ_RATIO), xytext=(1.5, 0.985),
                 arrowprops=dict(facecolor=DELOITTE_BLACK, shrink=0.05, width=1.5, headwidth=8),
                 horizontalalignment='center', fontsize=10, fontweight='bold', color=DELOITTE_BLACK)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ---------------------------------------------------------
    # EXPORT
    # ---------------------------------------------------------
    plt.tight_layout()
    report_filename = "5_ml_entropy_ratio_dashboard.png"
    plt.savefig(report_filename, dpi=300, bbox_inches='tight')
    print(f"-> Saved highly optimized visual to: {report_filename}")
    plt.show()

if __name__ == "__main__":
    generate_ml_dashboard()