# Import the necessary libraries for matrix operations and arrays
import numpy as np
from scipy.linalg import toeplitz

class ToeplitzExtractor:
    def __init__(self, n_input, m_output):
        """
        Initializes the Toeplitz Strong Extractor.
        n_input: length of the raw input bit sequence (n)
        m_output: length of the extracted output bit sequence (m)
        """
        self.n = n_input
        self.m = m_output
        
        # The Toeplitz matrix requires a specific seed length
        self.seed_length = self.m + self.n - 1
        self.matrix = None

    def generate_matrix(self, seed=None):
        """
        Generates the Toeplitz matrix based on the provided seed.
        seed: the initial bit sequence used to generate the matrix.
        """
        if seed is None:
            # Generate a random binary seed
            seed = np.random.randint(2, size=self.seed_length)
        elif len(seed) != self.seed_length:
            raise ValueError(f"Seed must be exactly {self.seed_length} bits long.")

        # A Toeplitz matrix is defined by its first column and first row.
        first_col = seed[:self.m]
        first_row = seed[self.m - 1:]
        
        # Build the matrix using scipy's optimized function
        self.matrix = toeplitz(c=first_col, r=first_row)
        print(f"[ToeplitzExtractor] Generated {self.m}x{self.n} matrix from seed.")
        
        return self.matrix

    def extract(self, raw_bits):
        """
        Performs the randomness extraction via matrix multiplication modulo 2.
        This mathematically simulates the concurrent XOR/AND pipeline of an FPGA.
        """
        if self.matrix is None:
            raise RuntimeError("Matrix not generated. Call generate_matrix() first.")
        if len(raw_bits) != self.n:
            raise ValueError(f"Input bits must be exactly {self.n} bits long.")

        # Matrix multiplication over Galois Field 2 (Modulo 2 arithmetic)
        # np.dot performs the mathematical equivalent of logical ANDs
        # % 2 performs the mathematical equivalent of logical XORs
        extracted_bits = np.dot(self.matrix, raw_bits) % 2
        
        return extracted_bits

# Quick internal test
if __name__ == "__main__":
    print("--- Running Toeplitz Extractor Test ---")
    # Using Zhang et al. parameters as an example: 1520 input bits, 1024 output bits
    extractor = ToeplitzExtractor(n_input=1520, m_output=1024)
    extractor.generate_matrix()
    
    # Create a dummy raw sequence
    dummy_raw = np.random.randint(2, size=1520)
    
    # Extract
    pure_bits = extractor.extract(dummy_raw)
    print(f"Extraction successful! Output length: {len(pure_bits)} bits.")
    print("--- Test Complete ---")