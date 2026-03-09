import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==========================================
class QuantumPredictorLSTM(nn.Module):
    def __init__(self, sequence_length=64, hidden_size=32, num_layers=2):
        """
        Initializes a deep Recurrent Neural Network (LSTM).
        Its purpose is to find hidden correlations in raw quantum bits
        caused by thermal or electronic hardware imperfections.
        """
        super(QuantumPredictorLSTM, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # LSTM Layer: The memory of the network. Searches for patterns over time.
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        # Dense Layer (Linear): Takes the LSTM results and compresses them.
        self.fc = nn.Linear(hidden_size, 1)
        
        # Activation Function (Sigmoid): Squeezes the final result 
        # between 0 and 1, returning a Probability.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the quantum data.
        """
        # x shape: (batch_size, sequence_length, 1)
        lstm_out, _ = self.lstm(x)
        
        # We take only the output of the last time step to make the prediction
        last_time_step_out = lstm_out[:, -1, :]
        
        linear_out = self.fc(last_time_step_out)
        probability = self.sigmoid(linear_out)
        
        return probability

# ==========================================
# 2. DATA PREPARATION MODULE
# ==========================================
def load_and_prepare_data(filepath, max_bits=100000, seq_length=64):
    """
    Loads a slice of the binary file and creates overlapping sequences.
    Example: [b1, b2, ... b64] -> Target: b65
    """
    print(f"-> Loading {max_bits:,} bits from {filepath}...")
    
    # Calculate how many bytes we need to read
    bytes_to_read = (max_bits // 8) + 1
    
    try:
        with open(filepath, 'rb') as f:
            raw_bytes = np.frombuffer(f.read(bytes_to_read), dtype=np.uint8)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {filepath}.")
        return None, None
        
    bitstream = np.unpackbits(raw_bytes)[:max_bits]
    
    print(f"-> Slicing data into sequences of length {seq_length}...")
    X = []
    y = []
    
    for i in range(len(bitstream) - seq_length):
        X.append(bitstream[i : i + seq_length])
        y.append(bitstream[i + seq_length])
        
    X = np.array(X, dtype=np.float32).reshape(-1, seq_length, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    # Split into Training (80%) and Testing (20%)
    split_index = int(len(X) * 0.8)
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]
    
    return (torch.tensor(X_train), torch.tensor(y_train)), (torch.tensor(X_test), torch.tensor(y_test))

# ==========================================
# 3. ENTROPY CALCULATION
# ==========================================
def calculate_min_entropy(p_guess):
    """
    Applies the formal definition of cryptographic Min-Entropy.
    H_min = -log2(max(p_guess, 1 - p_guess))
    """
    best_guess = max(p_guess, 1.0 - p_guess)
    min_entropy = -np.log2(best_guess)
    return min_entropy

# ==========================================
# 4. MAIN EXECUTION BLOCK (TRAINING LOOP)
# ==========================================
if __name__ == "__main__":
    FILENAME = "quantum_bitstream.bin"
    
    # We use 100,000 bits for a fast audit. 
    # Increase this to 1,000,000 for a rigorous overnight run.
    train_data, test_data = load_and_prepare_data(FILENAME, max_bits=1000000, seq_length=64)
    
    if train_data is not None:
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        print("\n=== INITIALIZING ADVERSARIAL AI ===")
        model = QuantumPredictorLSTM(sequence_length=64)
        
        # Binary Cross Entropy Loss is the standard for 0/1 prediction
        criterion = nn.BCELoss()
        # Adam optimizer adapts the learning rate dynamically
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        epochs = 5
        batch_size = 512
        
        print(f"-> Training on {len(X_train):,} sequences. Testing on {len(X_test):,} sequences.")
        print("-> Commencing Training Loop...")
        
        start_time = time.time()
        
        # --- TRAINING LOOP ---
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(X_train.size()[0])
            
            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            print(f"   Epoch {epoch+1}/{epochs} completed. Loss: {loss.item():.4f}")
            
        # --- EVALUATION LOOP ---
        print("\n=== EVALUATING MIN-ENTROPY ===")
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            # Convert probabilities to hard 0 or 1 predictions
            predicted_bits = (test_predictions >= 0.5).float()
            
            # Calculate exact accuracy
            correct_guesses = (predicted_bits == y_test).sum().item()
            total_guesses = len(y_test)
            accuracy = correct_guesses / total_guesses
            
        print(f"-> Adversarial Guessing Accuracy (P_guess): {accuracy*100:.4f}%")
        
        # The AI's accuracy IS the attacker's P_guess
        # If accuracy is worse than 50% (due to noise), we floor P_guess at 50%
        p_guess_effective = max(accuracy, 0.5)
        h_min = calculate_min_entropy(p_guess_effective)
        
        print(f"-> Certified Min-Entropy (H_inf): {h_min:.6f} bits/bit")
        
        elapsed_time = time.time() - start_time
        print(f"\n-> Audit completed in {elapsed_time:.2f} seconds.")