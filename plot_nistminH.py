import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==========================================
# DELOITTE BRAND IDENTITY (Extended Palette)
# ==========================================
# We define these globally for easier maintenance.
D_GREEN = '#86BC25'  # Primary
D_BLUE  = '#0097A9'  # Secondary
D_BLACK = '#000000'
D_SLATE = '#2A2E33'
D_GREY  = '#EAEAEA'  # Backgrounds/Ideal
D_TEXT  = '#5A5A5A'  # Labels
D_RED   = '#DA291C'  # WARNING/Bias

def plot_advanced_nist_audit(h_spatial, h_temporal):
    """
    Generates a professional diagnostic dashboard for NIST SP 800-90B results.
    Refined layout to ensure H_min values are inside bars for maximum readability.
    """
    # 1. Hardware Initialization (Set default white background)
    fig, ax = plt.subplots(figsize=(13, 8), facecolor='white')
    
    # 2. Data Preparation
    labels = ['SPATIAL DIMENSION\n(Path Selection)', 'TEMPORAL DIMENSION\n(Inter-arrival Time)']
    values = [h_spatial, h_temporal]
    bottlenecks = ['Compression Test', 'T-Tuple Test']
    physics_labels = [
        'Detected: Subtle path memory in detector array.', 
        'Detected: Hardware clock jitter & TDC resolution limitations.'
    ]
    
    # 3. Layer 1: Plotting the Ideal Standard (The 1.0 Entropy Target)
    # This creates a grey 'gauge' background that komunikasi standard quality.
    ax.barh(labels, [1.0, 1.0], color=D_GREY, height=0.5, label='Theoretical Ideal (1.0)', alpha=0.5)
    
    # 4. Layer 2: Plotting the Validated H_min (The Certified Audit Result)
    colors = [D_BLUE, D_GREEN]
    bars = ax.barh(labels, values, color=colors, height=0.5, alpha=0.9, zorder=3)
    
    # 5. Non-Invasive Aesthetic Refinement
    ax.set_xlim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(D_GREY)
    
    # Add subtle vertical grid lines for information density
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, color=D_TEXT)
    ax.set_axisbelow(True)
    
    # 6. CRITICAL LAYER: Adding Labels (Safe and Elegant Placement)
    for i, (val, bottleneck, physics) in enumerate(zip(values, bottlenecks, physics_labels)):
        
        # --- SOLUTION: MAIN VALUE LABEL (Moved INSIDE bar) ---
        # We center the number within the colored bar using zorder=5 
        # to guarantee it appears over the ideal background. We use white for contrast.
        ax.text(val / 2, i, f'{val:.6f}', va='center', ha='center', 
                fontsize=18, fontweight='bold', color='white', zorder=5)
        
        # Entropy Gap Marker (The Hardware Bias Area)
        # We show how much entropy is physically present but unusable without whitening.
        gap = 1.0 - val
        ax.annotate('', xy=(1.0, i + 0.28), xytext=(val, i + 0.28),
                    arrowprops=dict(arrowstyle='<->', color=D_RED, lw=1.5, alpha=0.6))
        ax.text((val + 1.0)/2, i + 0.35, f'Entropy Gap (Raw Bias): {gap:.4f}', 
                ha='center', fontsize=9, color=D_RED, style='italic', weight='bold')
        
        # Diagnostic Box (Keep right aligned to structure the dashboard)
        # Now there is no risk of obscuring the main entropy numbers.
        box_text = f"CRITICAL BOTTLENECK: {bottleneck.upper()}\n{physics}"
        ax.text(1.14, i, box_text, va='center', ha='right', fontsize=10, 
                color=D_TEXT, bbox=dict(facecolor='white', edgecolor=D_GREY, boxstyle='round,pad=0.8'))

    # 7. Editorial Typography & Titles
    plt.title('QUANTUM RAW ENTROPY AUDIT: NIST SP 800-90B CERTIFICATION', 
              fontsize=20, weight='bold', color=D_BLACK, loc='left', pad=30)
    
    ax.set_xlabel('Certified Min-Entropy ($H_{min}$) per Raw Bit', fontsize=12, color=D_TEXT, labelpad=15)
    
    # Adjust tick appearance
    ax.tick_params(axis='x', colors=D_TEXT, labelcolor=D_TEXT)
    ax.tick_params(axis='y', left=False, colors=D_TEXT, labelcolor=D_TEXT)

    # 8. Integrated Strategic Footer Note
    footer_text = (
        "Methodology: NIST ea_non_iid suite executed via WSL bridge (Ubuntu 24.04 toolchain).\n"
        "Interpretation: Post-Toeplitz extraction ratios m/n must strictly adhere to these conservative mathematical bounds\n"
        "to guarantee cryptographic indistinguishability from a uniform distribution (Leftover Hash Lemma, -128b)."
    )
    fig.text(0.12, 0.02, footer_text, fontsize=9, color=D_TEXT, alpha=0.8, style='italic', linespacing=1.6)

    # Prevent title/footer collision
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    
    # High-resolution output for Master's thesis
    save_path = 'nist_dashboard_safe_readability.png'
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"-> Professional Visualization generated: {save_path}")
    plt.show()

# ==========================================
# MAIN EXECUTION (DATA INJECTION)
# ==========================================
if __name__ == "__main__":
    print("\n=== GENERATING ADVANCED CRYPTOGRAPHIC DASHBOARD ===")
    
    # Verified h' results from your WSL ea_non_iid executions
    h_spatial = 0.753286
    h_temporal = 0.939019
    
    plot_advanced_nist_audit(h_spatial, h_temporal)
    print("=== DASHBOARD STREAM COMPLETE ===\n")