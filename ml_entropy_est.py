import torch
import torch.nn as nn
import numpy as np

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
        
        # Pass through the LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # We take only the output of the last time step to make the prediction
        last_time_step_out = lstm_out[:, -1, :]
        
        # Pass through the linear layer
        linear_out = self.fc(last_time_step_out)
        
        # Conversion to probability
        probability = self.sigmoid(linear_out)
        
        return probability

def calculate_min_entropy(p_guess):
    """
    Applies the formal definition of cryptographic Min-Entropy.
    H_min = -log2(max(p_guess, 1 - p_guess))
    """
    # The attacker always takes the safest bet
    best_guess = max(p_guess, 1.0 - p_guess)
    
    # If the probability is 0.5 (perfect), H_min will be 1.0 bit
    # If the probability is 1.0 (deterministic), H_min will be 0.0 bit
    min_entropy = -np.log2(best_guess)
    return min_entropy

# --- INTERNAL TEST ---
if __name__ == "__main__":
    print("-> Initializing Quantum Predictor LSTM...")
    model = QuantumPredictorLSTM()
    
    # Simulate an input: a batch of 10 sequences, each of 64 raw bits
    dummy_input = torch.randn(10, 64, 1) 
    
    # The network tries to predict the 65th bit
    predictions = model(dummy_input)
    
    print("-> Forward Pass test completed.")
    print(f"-> Example of predicted probability for the first sequence: {predictions[0].item():.4f}")
    
    # Calculate the entropy if the network guesses with 60% accuracy
    h_min = calculate_min_entropy(0.7)
    print(f"-> Min-Entropy with P_guess at 70%: {h_min:.4f} bits of real entropy per raw bit.")