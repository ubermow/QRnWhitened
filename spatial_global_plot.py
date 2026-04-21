import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SWABIAN HARDWARE INTEGRATION
# ==========================================
SWABIAN_PYTHON_PATH = r"C:\Program Files\Swabian Instruments\Time Tagger\driver\x64\python3.10"
SWABIAN_ROOT_PATH = r"C:\Program Files\Swabian Instruments\Time Tagger"

if SWABIAN_PYTHON_PATH not in sys.path:
    sys.path.append(SWABIAN_PYTHON_PATH)
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(SWABIAN_ROOT_PATH)
    os.add_dll_directory(SWABIAN_PYTHON_PATH)

try:
    from TimeTagger import FileReader
except ImportError:
    print("[CRITICAL ERROR] TimeTagger FileReader could not be loaded.")
    sys.exit(1)

# ==========================================
# 2. DELOITTE CORPORATE AESTHETIC CONFIGURATION
# ==========================================
DELOITTE_BLUE  = '#0097A9'  
DELOITTE_RED   = '#DA291C'  
DELOITTE_GREY  = '#A6A6A6'  
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
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.frameon': False,
})

def despine_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')

# ==========================================
# 3. UNIFIED ENGINE: READ, ANALYZE & PLOT
# ==========================================
def analyze_and_plot_deloitte(input_filepath, chunk_size=10_000_000):
    print(f"\n=== DELOITTE QRNG: RAW DATA TO PUBLICATION PLOT ===")
    print(f"-> Target: {input_filepath}")
    
    CHANNEL_T = 1  
    CHANNEL_A = 2  
    CHANNEL_B = 3  
    WINDOW_PS = 5000  
    DELAY_PS = -500   
    
    try:
        reader = FileReader(input_filepath)
    except Exception as e:
        print(f"[ERROR] Could not open {input_filepath}: {e}")
        return

    print("-> Crunching raw timestamps...")
    start_time = time.time()
    
    count_T, count_A, count_B, count_TA, count_TB = 0, 0, 0, 0, 0
    chunks_processed = 0
    first_ts, last_ts = None, None
    
    while reader.hasData():
        buffer = reader.getData(chunk_size)
        channels = buffer.getChannels()
        timestamps_ps = buffer.getTimestamps()
        
        if len(timestamps_ps) > 0:
            if first_ts is None: first_ts = timestamps_ps[0]
            last_ts = timestamps_ps[-1]
            
        t_T = timestamps_ps[channels == CHANNEL_T]
        t_A = timestamps_ps[channels == CHANNEL_A] + DELAY_PS
        t_B = timestamps_ps[channels == CHANNEL_B] + DELAY_PS
        
        count_T += len(t_T)
        count_A += len(t_A)
        count_B += len(t_B)
        
        if len(t_T) > 0:
            left_A = np.searchsorted(t_A, t_T)
            right_A = np.searchsorted(t_A, t_T + WINDOW_PS)
            count_TA += np.sum(right_A > left_A)
            
            left_B = np.searchsorted(t_B, t_T)
            right_B = np.searchsorted(t_B, t_T + WINDOW_PS)
            count_TB += np.sum(right_B > left_B)
            
        chunks_processed += 1
        if chunks_processed % 10 == 0:
            print(f"   ... Analyzed {chunks_processed * (chunk_size // 1_000_000)}M events", end='\r')

    if first_ts is None or last_ts is None:
        print("\n[ERROR] No valid timestamps found.")
        return

    acq_time_sec = (last_ts - first_ts) / 1e12
    rates = {
        'T': (count_T / acq_time_sec) / 1000,
        'A': (count_A / acq_time_sec) / 1000,
        'B': (count_B / acq_time_sec) / 1000,
        'TA': (count_TA / acq_time_sec) / 1000,
        'TB': (count_TB / acq_time_sec) / 1000
    }
    
    print(f"\n-> Analysis complete in {time.time() - start_time:.2f}s")
    print(f"-> Rendering Deloitte Plot...")

    # --- PLOTTING LOGIC ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 2]})
    fig.subplots_adjust(wspace=0.25)

    # Left Plot
    bars1 = ax1.bar(['Det T\n(Idler)', 'Det A\n(Signal 0)', 'Det B\n(Signal 1)'], 
                    [rates['T'], rates['A'], rates['B']], 
                    color=DELOITTE_GREY, alpha=0.7, edgecolor=WHITE, lw=1.5, width=0.45)
    ax1.set_ylabel('Detection Rate (kHz)', fontweight='bold', labelpad=10)
    ax1.set_title('Macroscopic Hardware Rates', pad=20, fontweight='bold', color=DELOITTE_DARK)
    for bar in bars1:
        y = bar.get_height()
        ax1.annotate(f'{y:.2f}', xy=(bar.get_x() + bar.get_width()/2, y), xytext=(0, 4), 
                     textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    despine_ax(ax1)
    ax1.grid(axis='y', alpha=0.15, linestyle='-')
    ax1.set_ylim(0, max([rates['T'], rates['A'], rates['B']]) * 1.2)

    # Right Plot
    bars2 = ax2.bar(['Coin\nT & A', 'Coin\nT & B'], [rates['TA'], rates['TB']], 
                    color=DELOITTE_BLUE, alpha=0.95, edgecolor=WHITE, lw=1.5, width=0.4)
    ax2.set_ylabel('Coincidence Rate (kHz)', fontweight='bold', labelpad=10)
    ax2.set_title('Heralded Spatial Yield ($W=5$ ns)', pad=20, fontweight='bold', color=DELOITTE_DARK)
    for bar in bars2:
        y = bar.get_height()
        ax2.annotate(f'{y:.2f}', xy=(bar.get_x() + bar.get_width()/2, y), xytext=(0, 4), 
                     textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    avg_coin = np.mean([rates['TA'], rates['TB']])
    ax2.axhline(y=avg_coin, color=DELOITTE_RED, linestyle='--', alpha=0.6, lw=1.5, 
                label=f'Balanced Mean: {avg_coin:.2f} kHz', zorder=0)
    
    despine_ax(ax2)
    ax2.grid(axis='y', alpha=0.15, linestyle='-')
    ax2.set_ylim(0, max([rates['TA'], rates['TB']]) * 1.2)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), frameon=True, facecolor=WHITE, edgecolor='#E0E0E0')

    fig.suptitle('Spatial Mode Balancing & Quantum Yield Distribution', fontweight='bold', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig('plot_3_spatial_balancing.png', dpi=400, bbox_inches='tight')
    print("[SUCCESS] Saved as plot_3_spatial_balancing.png")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Inserisci qui il tuo percorso raw assoluto o relativo
    RAW_FILE_PATH = r"data\\raw\\3hours_nopeople\\3h_raw_photons.ttbin"
    analyze_and_plot_deloitte(RAW_FILE_PATH)