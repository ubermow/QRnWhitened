import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

# ==========================================
# 1. HARDWARE ACCELERATION & AESTHETICS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deloitte Corporate Color Palette
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

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class QuantumPredictorLSTM(nn.Module):
    def __init__(self, sequence_length=64, hidden_size=32, num_layers=2):
        super(QuantumPredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        linear_out = self.fc(last_time_step_out)
        return self.sigmoid(linear_out)

# ==========================================
# 3. VECTORIZED DATA PREPARATION
# ==========================================
def load_and_prepare_data(filepath, max_bits=1_000_000, seq_length=64):
    print(f"\n-> Loading data from {os.path.basename(filepath)}...")
    bytes_to_read = (max_bits // 8) + 1
    
    if not os.path.exists(filepath):
        print(f"[ERROR] Could not find {filepath}.")
        return None
        
    with open(filepath, 'rb') as f:
        raw_bytes = np.frombuffer(f.read(bytes_to_read), dtype=np.uint8)
        
    bitstream = np.unpackbits(raw_bytes)[:max_bits]
    
    # Calculate Raw Shannon Entropy of the string
    p1 = np.sum(bitstream) / len(bitstream)
    p0 = 1.0 - p1
    shannon_entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1)) if p0 > 0 and p1 > 0 else 0.0
    
    windows = sliding_window_view(bitstream, seq_length + 1)
    X = windows[:, :-1].astype(np.float32).reshape(-1, seq_length, 1)
    y = windows[:, -1].astype(np.float32).reshape(-1, 1)
    
    split_index = int(len(X) * 0.8)
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]
    
    count_1 = np.sum(y_test)
    naive_accuracy = max(count_1 / len(y_test), 1.0 - (count_1 / len(y_test)))
    
    return (torch.tensor(X_train), torch.tensor(y_train)), (torch.tensor(X_test), torch.tensor(y_test)), naive_accuracy, shannon_entropy

# ==========================================
# 4. AI AUDIT ENGINE
# ==========================================
def run_ai_audit(filepath, epochs=5, batch_size=1024):
    data_payload = load_and_prepare_data(filepath, max_bits=1_000_000, seq_length=64)
    if data_payload is None: return None
    
    (X_train, y_train), (X_test, y_test), naive_acc, shannon_ent = data_payload
    
    model = QuantumPredictorLSTM(sequence_length=64).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    loss_history = []
    
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        total_loss = 0
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train[indices].to(DEVICE)
            batch_y = y_train[indices].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / (X_train.size()[0] / batch_size)
        loss_history.append(avg_loss)
        print(f"   Epoch {epoch+1}/{epochs} completed. Avg Loss: {avg_loss:.4f}")
        
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        predicted_bits = (test_predictions >= 0.5).float()
        correct_guesses = (predicted_bits == y_test).sum().item()
        ai_accuracy = correct_guesses / len(y_test)
        
    p_guess_effective = max(ai_accuracy, naive_acc, 0.5)
    min_ent = -np.log2(p_guess_effective)
    
    # Calculate Leftover Hash Lemma Ratio
    m_out = int(np.floor(2048 * min_ent) - 128)
    toeplitz_ratio = m_out / 2048
    
    return loss_history, shannon_ent, min_ent, toeplitz_ratio

# ==========================================
# 5. EDITORIAL VISUALIZATION ENGINE
# ==========================================
def style_axes(ax):
    """Applies the clean, non-invasive editorial aesthetic to an axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.yaxis.grid(True, linestyle='-', alpha=0.4, color=LIGHT_GREY)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', bottom=True, color='#CCCCCC', labelcolor=TEXT_GREY)
    ax.tick_params(axis='y', left=False, labelcolor=TEXT_GREY)

def plot_learning_curves(temporal_losses, spatial_losses):
    epochs = np.arange(1, len(temporal_losses) + 1)
    theoretical_limit = 0.693147 # -ln(0.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    style_axes(ax)
    
    ax.axhline(theoretical_limit, color=DARK_SLATE, linestyle='--', linewidth=1.5, 
               label='Theoretical Guessing Limit (0.6931)', alpha=0.7, zorder=2)
    
    ax.plot(epochs, temporal_losses, marker='o', color=DELOITTE_GREEN, linewidth=2.5, 
            markersize=7, label='Temporal Bitstream Loss', alpha=0.9, zorder=3)
            
    ax.plot(epochs, spatial_losses, marker='s', color=WARNING_RED, linewidth=2.5, 
            markersize=7, label='Spatial Bitstream Loss', alpha=0.9, zorder=3)

    ax.set_title('Adversarial AI Convergence (LSTM Loss Landscape)', fontsize=16, color=DARK_SLATE, pad=20, loc='left')
    ax.set_xlabel('Training Epochs', fontsize=11, color=TEXT_GREY, labelpad=10)
    ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=11, color=TEXT_GREY, labelpad=10)
    
    min_loss = min(min(temporal_losses), min(spatial_losses))
    max_loss = max(max(temporal_losses), max(spatial_losses))
    buffer = (max_loss - min_loss) * 0.5 if max_loss != min_loss else 0.005
    
    ax.set_ylim(min_loss - buffer, max_loss + buffer)
    ax.set_xticks(epochs)
    
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
    
    # 1. Subtle horizontal dashed lines for Min-Entropy reference
    ax.axhline(y=min_ent_t, color=DELOITTE_GREEN, linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axhline(y=min_ent_s, color=DELOITTE_BLUE, linestyle=':', alpha=0.5, linewidth=1.5)
    
    # 2. Elegant "-128b" vertical dimension lines
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

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"=== INITIALIZING AI CRYPTO-AUDIT ON {DEVICE.type.upper()} ===")
    
    # POINT THESE TO YOUR RAW FILES
    FILE_TEMPORAL = r"data\\raw\\3hours_nopeople\\temporal_3hraw_bitstream.bin"
    FILE_SPATIAL  = r"data\\raw\\3hours_nopeople\\spatial_3hraw_bitstream.bin"

    EPOCHS = 5
    
    print("\n[+] Analyzing Temporal Sequence...")
    t_results = run_ai_audit(FILE_TEMPORAL, epochs=EPOCHS)
    
    print("\n[+] Analyzing Spatial Sequence...")
    s_results = run_ai_audit(FILE_SPATIAL, epochs=EPOCHS)
    
    if t_results and s_results:
        print("\n=== GENERATING CORPORATE DASHBOARD VISUALS ===")
        t_losses, t_shannon, t_min_ent, t_ratio = t_results
        s_losses, s_shannon, s_min_ent, s_ratio = s_results
        
        plot_learning_curves(t_losses, s_losses)
        plot_safety_margin(t_shannon, t_min_ent, t_ratio, s_shannon, s_min_ent, s_ratio)
        
        print("\n=== AUDIT COMPLETE ===")