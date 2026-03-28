import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import matplotlib.pyplot as plt
import time


# ==========================================
# 1. HARDWARE ACCELERATION & AESTHETICS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"-> Using device: {DEVICE}")

# Corporate Color Palette (Deloitte)
DELOITTE_GREEN = '#86BC25'
DELOITTE_BLUE  = '#0097A9' 
WARNING_RED    = '#DA291C' 
DARK_SLATE     = '#2A2E33'
LIGHT_GREY     = '#EAEAEA'
TEXT_GREY      = '#5A5A5A'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'text.color': DARK_SLATE,
})

# Toggles for Calibration Sanity Checks
DEBUG_MODE = False  # Set to True to inject 101010... sequence

# ==========================================
# 2. MEMORY-OPTIMIZED DATASET
# ==========================================
class BitstreamDataset(Dataset):
    """
    Pre-casts the entire bitstream into a PyTorch tensor. 
    Eliminates the massive CPU bottleneck of casting np.float32 per batch.
    """
    def __init__(self, bitstream, seq_length=64):
        self.seq_length = seq_length
        self.bitstream_tensor = torch.from_numpy(bitstream).float()

    def __len__(self):
        return len(self.bitstream_tensor) - self.seq_length

    def __getitem__(self, idx):
        x = self.bitstream_tensor[idx : idx + self.seq_length].unsqueeze(-1)
        y = self.bitstream_tensor[idx + self.seq_length].unsqueeze(0)
        return x, y

# ==========================================
# 3. SOTA NEURAL NETWORK ARCHITECTURE
# ==========================================
class TPALSTM(nn.Module):
    """
    Rigorous Temporal Pattern Attention (TPA) LSTM.
    Uses 1D Convolutions across the temporal dimension of hidden states to 
    isolate deterministic cyclic resonances in the physical hardware.
    """
    def __init__(self, sequence_length=256, hidden_size=64, num_layers=2, num_cnn_filters=32):
        super(TPALSTM, self).__init__()
        
        # Dropout = 0: Allow maximal overfitting to hardware flaws
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=0.0)
        
        # TPA specific components
        self.cnn = nn.Conv1d(in_channels=hidden_size, out_channels=num_cnn_filters, kernel_size=1)
        self.w_h = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(num_cnn_filters, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.attention_proj = nn.Linear(num_cnn_filters, hidden_size)
        
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # lstm_out: (batch, seq_len, hidden_size), h_n: (num_layers, batch, hidden_size)
        lstm_out, (h_n, _) = self.lstm(x)
        h_n_last = h_n[-1] # Target the final hidden state
        
        # CNN requires (batch, channels, length)
        cnn_in = lstm_out.transpose(1, 2)
        cnn_out = self.cnn(cnn_in) # (batch, num_filters, seq_len)
        
        # H matrix for attention scoring: (batch, seq_len, num_filters)
        H = cnn_out.transpose(1, 2)
        
        # Scoring function: f(H_i, h_t) = v^T tanh(W_h h_t + W_v H_i)
        h_n_proj = self.w_h(h_n_last).unsqueeze(1) # (batch, 1, hidden_size)
        H_proj = self.w_v(H)                       # (batch, seq_len, hidden_size)
        
        scores = self.v(torch.tanh(h_n_proj + H_proj)).squeeze(-1) # (batch, seq_len)
        
        # TPA traditionally uses sigmoid to allow multiple patterns to be attended to simultaneously
        alpha = torch.sigmoid(scores) # (batch, seq_len)
        
        # Context vector v_t
        v_t = torch.bmm(alpha.unsqueeze(1), H).squeeze(1) # (batch, num_filters)
        
        # Final prediction vector
        v_t_proj = self.attention_proj(v_t)
        h_star = h_n_last + v_t_proj
        
        return self.sigmoid(self.fc(h_star))

# ==========================================
# 4. TRAINING UTILITIES
# ==========================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==========================================
# 5. DATA PREPARATION & METROLOGY ENGINE
# ==========================================
def load_bitstream(filepath, max_bits=1_000_000):
    if DEBUG_MODE:
        print(f"\n[!] WARNING: DEBUG_MODE IS ACTIVE. Injecting deterministic alternating sequence.")
        # Generates [1, 0, 1, 0, ...] to mathematically prove gradient descent works
        return np.array([i % 2 for i in range(max_bits)], dtype=np.uint8)

    print(f"\n-> Loading data from {os.path.basename(filepath)}...")
    bytes_to_read = (max_bits // 8) + 1
    
    if not os.path.exists(filepath):
        print(f"[ERROR] Could not find {filepath}.")
        return None
        
    with open(filepath, 'rb') as f:
        raw_bytes = np.frombuffer(f.read(bytes_to_read), dtype=np.uint8)
        
    return np.unpackbits(raw_bytes)[:max_bits]


def compile_model_for_cpu(seq_length=256, batch_size=512):
    print("\n=== INITIATING TORCHSCRIPT JIT COMPILATION ===")
    
    # 1. Initialize the standard Python model
    model = TPALSTM(sequence_length=seq_length)
    model.eval() # Must be in eval mode for a clean trace
    
    # 2. Generate a dummy tensor with the exact dimensions of your training batches
    # Shape: (Batch Size, Sequence Length, Input Features)
    dummy_input = torch.randn(batch_size, seq_length, 1)
    
    print("-> Tracing computational graph...")
    start_time = time.time()
    
    # 3. Trace the model
    # The tracer runs the dummy input through the forward pass and records every C++ operation
    traced_model = torch.jit.trace(model, dummy_input)
    
    compile_time = time.time() - start_time
    print(f"-> Graph successfully compiled in {compile_time:.4f} seconds.")
    
    # 4. Save the compiled model for the audit loop
    compiled_filename = "tpalstm_jit_optimized.pt"
    traced_model.save(compiled_filename)
    print(f"-> Saved optimized C++ backend model to: {compiled_filename}")
    
    return traced_model


def run_ai_audit(filepath, epochs=50, batch_size=512, seq_length=256):
    bitstream = load_bitstream(filepath, max_bits=2_000_000)
    if bitstream is None: return None
    
    p1 = np.mean(bitstream)
    p0 = 1.0 - p1
    shannon_ent = -(p0 * np.log2(p0) + p1 * np.log2(p1)) if p0 > 0 and p1 > 0 else 0.0

    total_len = len(bitstream)
    train_split = int(total_len * 0.7)
    val_split = int(total_len * 0.85)

    train_data = bitstream[:train_split]
    val_data = bitstream[train_split:val_split]
    test_data = bitstream[val_split:]

    train_loader = DataLoader(BitstreamDataset(train_data, seq_length), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(BitstreamDataset(val_data, seq_length), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(BitstreamDataset(test_data, seq_length), batch_size=batch_size, shuffle=False)

    # Load the pre-compiled C++ model
    model = torch.jit.load("tpalstm_jit_optimized.pt").to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=25) 

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        epoch_start_time = time.time() # Start the stopwatch
        
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_end_time = time.time() # Stop the stopwatch
        epoch_duration = epoch_end_time - epoch_start_time # Calculate elapsed time
        
        print(f"   Epoch {epoch+1}/{epochs} | Time: {epoch_duration:.2f}s | LR: {current_lr:.8f} | Train Loss: {avg_train_loss:.8f} | Val Loss: {avg_val_loss:.8f}")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("   [!] Early stopping triggered. Absolute hardware floor reached.")
            break


    # Evaluation on Test Data
    model.eval()
    correct_guesses = 0
    total_guesses = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            predicted = (outputs >= 0.5).float()
            correct_guesses += (predicted == batch_y).sum().item()
            total_guesses += batch_x.size(0)

    ai_accuracy = correct_guesses / total_guesses
    naive_accuracy = max(p1, p0)
    
    # NIST SP 800-90B Min-Entropy calculation
    p_guess_effective = max(ai_accuracy, naive_accuracy, 0.5)
    min_ent = -np.log2(p_guess_effective)
    
    # Leftover Hash Lemma (LHL) Metrology
    n_input_bits = 131072 # Typical input block size (128KB)
    epsilon = 2**-64      # Target collision probability
    penalty = 2 * np.log2(1/epsilon) # Exactly 128 bits
    
    m_out = int(np.floor(n_input_bits * min_ent) - penalty)
    m_out = max(m_out, 0) # Zero out negative yields
    toeplitz_ratio = m_out / float(n_input_bits)
    
    return train_loss_history, val_loss_history, shannon_ent, min_ent, toeplitz_ratio

# ==========================================
# 6. EDITORIAL VISUALIZATION ENGINE
# ==========================================
def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.yaxis.grid(True, linestyle='-', alpha=0.4, color=LIGHT_GREY)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', bottom=True, color='#CCCCCC', labelcolor=TEXT_GREY)
    ax.tick_params(axis='y', left=False, labelcolor=TEXT_GREY)

def plot_learning_curves(temporal_val_losses, spatial_val_losses):
    epochs_t = np.arange(1, len(temporal_val_losses) + 1)
    epochs_s = np.arange(1, len(spatial_val_losses) + 1)
    theoretical_limit = 0.69314718 # -ln(0.5) expanded for scientific rigor

    fig, ax = plt.subplots(figsize=(10, 6))
    style_axes(ax)
    
    ax.axhline(theoretical_limit, color=DARK_SLATE, linestyle='--', linewidth=1.5, 
               label='Theoretical Guessing Limit (0.6931)', alpha=0.7, zorder=2)
    
    ax.plot(epochs_t, temporal_val_losses, marker='o', color=DELOITTE_GREEN, linewidth=2.5, 
            markersize=7, label='Temporal Validation Loss', alpha=0.9, zorder=3)
            
    ax.plot(epochs_s, spatial_val_losses, marker='s', color=WARNING_RED, linewidth=2.5, 
            markersize=7, label='Spatial Validation Loss', alpha=0.9, zorder=3)

    ax.set_title('Adversarial AI Convergence (TPA-LSTM Loss Landscape)', fontsize=16, color=DARK_SLATE, pad=20, loc='left')
    ax.set_xlabel('Training Epochs', fontsize=11, color=TEXT_GREY, labelpad=10)
    ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=11, color=TEXT_GREY, labelpad=10)
    
    all_losses = temporal_val_losses + spatial_val_losses
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    buffer = (max_loss - min_loss) * 0.5 if max_loss != min_loss else 0.005
    
    ax.set_ylim(min_loss - buffer, max_loss + buffer)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='white', fontsize=11)

    plt.tight_layout()
    filename = 'plot_ai_convergence.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"-> Saved Visualization: {filename}")
    plt.close()

def plot_safety_margin(shannon_t, min_ent_t, ratio_t, shannon_s, min_ent_s, ratio_s):
    fig, ax = plt.subplots(figsize=(12, 7))
    style_axes(ax)
    
    labels = ['Raw Shannon\nEntropy', 'AI Certified\nMin-Entropy', 'Toeplitz Ratio\n(LHL)']
    temporal_vals = [shannon_t, min_ent_t, ratio_t]
    spatial_vals = [shannon_s, min_ent_s, ratio_s]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars_t = ax.bar(x - width/2, temporal_vals, width, label='Temporal System', color=DELOITTE_GREEN, alpha=0.9)
    bars_s = ax.bar(x + width/2, spatial_vals, width, label='Spatial System', color=DELOITTE_BLUE, alpha=0.9)
    
    ax.set_title('Cryptographic Yield vs. Adversarial Predictability', fontsize=16, color=DARK_SLATE, pad=25, loc='left')
    ax.set_ylabel('Information (Bits / Raw Bit)', fontsize=11, color=TEXT_GREY, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    
    ax.set_ylim(0.0, 1.1)
    
    text_box_style = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 6), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color=DARK_SLATE, bbox=text_box_style)
                        
    add_labels(bars_t)
    add_labels(bars_s)
    
    ax.axhline(y=min_ent_t, color=DELOITTE_GREEN, linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axhline(y=min_ent_s, color=DELOITTE_BLUE, linestyle=':', alpha=0.5, linewidth=1.5)
    
    x_ratio_t = x[2] - width/2 - 0.12
    ax.plot([x_ratio_t, x_ratio_t], [min_ent_t, ratio_t], color=DELOITTE_GREEN, linewidth=1.2, alpha=0.6)
    ax.plot([x_ratio_t - 0.03, x_ratio_t + 0.03], [min_ent_t, min_ent_t], color=DELOITTE_GREEN, linewidth=1.2, alpha=0.6)
    ax.plot([x_ratio_t - 0.03, x_ratio_t + 0.03], [ratio_t, ratio_t], color=DELOITTE_GREEN, linewidth=1.2, alpha=0.6)
    ax.text(x_ratio_t - 0.06, (ratio_t + min_ent_t)/2, '−128b Penalty',
            ha='right', va='center', color=DELOITTE_GREEN, fontsize=9, alpha=0.8)

    x_ratio_s = x[2] + width/2 + 0.12
    ax.plot([x_ratio_s, x_ratio_s], [min_ent_s, ratio_s], color=DELOITTE_BLUE, linewidth=1.2, alpha=0.6)
    ax.plot([x_ratio_s - 0.03, x_ratio_s + 0.03], [min_ent_s, min_ent_s], color=DELOITTE_BLUE, linewidth=1.2, alpha=0.6)
    ax.plot([x_ratio_s - 0.03, x_ratio_s + 0.03], [ratio_s, ratio_s], color=DELOITTE_BLUE, linewidth=1.2, alpha=0.6)
    ax.text(x_ratio_s + 0.06, (ratio_s + min_ent_s)/2, '−128b Penalty',
            ha='left', va='center', color=DELOITTE_BLUE, fontsize=9, alpha=0.8)

    ax.legend(loc='lower right', frameon=False, fontsize=11)

    plt.tight_layout()
    filename = 'plot_safety_margin.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"-> Saved Visualization: {filename}")
    plt.close()

if __name__ == "__main__":
    print(f"=== INITIALIZING TPA-LSTM CRYPTO-AUDIT ON {DEVICE.type.upper()} ===")

    # [!] FIX: We must compile and save the JIT model before the audit loops try to load it
    compile_model_for_cpu(seq_length=256, batch_size=512)

    # Ensure these files exist or create dummy data for testing
    FILE_TEMPORAL = r"data\\raw\\3hours_nopeople\\temporal_3hraw_bitstream.bin"  
    FILE_SPATIAL  = r"data\\raw\\3hours_nopeople\\spatial_3hraw_bitstream.bin"  

    MAX_EPOCHS = 50
    BATCH_SIZE = 512
    
    print("\n[+] Analyzing Temporal Sequence...")
    t_results = run_ai_audit(FILE_TEMPORAL, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
    
    print("\n[+] Analyzing Spatial Sequence...")
    s_results = run_ai_audit(FILE_SPATIAL, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)

    # ... (Rest of your visualization code remains identical)

    if t_results and s_results:
        print("\n[+] Generating Audit Visualizations...")
        t_train, t_val, t_shan, t_min, t_ratio = t_results
        s_train, s_val, s_shan, s_min, s_ratio = s_results
        
        plot_learning_curves(t_val, s_val)
        plot_safety_margin(t_shan, t_min, t_ratio, s_shan, s_min, s_ratio)
        
        print("\n=== AUDIT COMPLETE ===")