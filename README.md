# High-Speed Heralded QRNG with FFT-Accelerated Privacy Amplification #

**Author:** Simone Ferilli  
**Institution:** University of Pisa / Deloitte Portugal Consulting  
**Department:** Cyber and Telecom Networks — The Hoop Lab  
**Hardware Platform:** Swabian Time Tagger / EDU-QOP1(/M) Quantum Optics Kit / Lenovo ThinkPad (Deloitte IT)  
**Location:** Lisbon, Portugal  

---

## Project Context
This repository contains a complete, end-to-end pipeline for generating, validating, and purifying high-speed heralded quantum random numbers. The architecture extracts raw quantum entropy from two distinct physical dimensions (Spatial path-selection and Temporal inter-arrival times). 

To guarantee information-theoretic security, the pipeline features a novel **Hybrid FFT-Toeplitz Privacy Amplification** engine operating on massive 131,072-bit blocks. Cryptographic bounds are rigorously certified using both the official **NIST SP 800-90B** standard and an adversarial **Attention-LSTM** Deep Learning audit.

## Key Machine Learning Insight
Crucially, the AI-derived min-entropy ($H_{\infty}^{\text{AI}}$) was slightly *less conservative* than the rigorous NIST SP 800-90B bound. This indicates that the Attention-LSTM failed to identify exploitable structure beyond what classical estimators had already characterized. Consistent with findings by Gao *et al.* (2025) and Li *et al.* (2020), DL-based estimators converge near, but do not systematically exceed, classical statistical bounds for well-whitened QRNG outputs. This is a highly positive security indicator: even a non-linear universal function approximator finds no exploitable regularity in the whitened output beyond the classical baseline.

---

## Repository Organization
The repository is modularized into specific operational domains:

```text
├── data/
│   ├── raw/                 # Raw .bin files from Swabian bridge (not tracked)
│   └── whitened/            # Final purified cryptographic keys (not tracked)
├── src/
│   ├── raw_data.py          # Swabian data ingestion
│   ├── digitalization/      # temporal_digitalizer.py, spatial_digitalizer.py
│   ├── entropy_audit/       # ml_entropy_est.py, unpack_NISTminH.py
│   └── extraction/          # FFToeplitz.py
└── visuals/
    ├── photon_stat.py       # Physical photon profiling
    ├── visual_ratioML_est.py# AI epoch convergence visualization
    ├── compare_ntro.py      # Bias, autocorrelation (Lags 1-3), and throughput
    ├── bin2txt.py           # Binary to ASCII conversion for NIST STS
    ├── plot_nistminH.py     # Executive dashboard for NIST 90B results
    └── plot_nist_pval.py    # Visualization of NIST STS P-values
```

## Environment & Dependencies
This pipeline operates as a hybrid Windows/Linux system. Python processing is handled via standard environments (e.g., Conda), while cryptographic validation requires a native Linux environment.Bash# Python Environment Setup
conda create -n qrng_env python=3.9
conda activate qrng_env

pip install torch numpy matplotlib scipy

## Execution Flux:
Step-by-Step Guide,

Phase 1:
# Physical Ingestion & Digitalization

1. src/raw_data.py: Ingest the raw binary buffers from the Swabian Time Tagger. 

2. visuals/photon_stat.py: Profile heralded photon arrivals to verify hardware nominality.

3. src/digitalization/temporal_digitalizer.py: Extract raw temporal bitstreams from inter-arrival times ($\Delta t$).

4. src/digitalization/spatial_digitalizer.py: Extract raw spatial bitstreams from Channel A/B path selections.

Phase 2: Entropy Auditing (Classical & Adversarial)

5. src/entropy_audit/unpack_NISTminH.py: Unpack raw 8-bit binaries into 1-bit-per-symbol binaries required by the NIST SP 800-90B suite.

6. Execute NIST SP 800-90B (See Exploiting NIST Libraries section below) to establish the physical baseline.

7. src/entropy_audit/ml_entropy_est.py: Run the adversarial Attention-LSTM audit to verify the physical bound against deep-learning attacks.

8.visuals/visual_ratioML_est.py & visuals/plot_nistminH.py: Generate executive dashboards comparing physical bounds to the ideal theoretical limit.

Phase 3: Privacy Amplification

9. src/extraction/FFToeplitz.py: Execute the Hybrid FFT-Toeplitz extractor. This script processes 131,072-bit blocks using the dimension-specific extraction ratios derived from Phase 2, minus a strict 128-bit Leftover Hash Lemma (LHL) security penalty.

Phase 4: Final Cryptographic Validation

10. visuals/bin2txt.py: Convert the purified .bin keys into ASCII .txt strings (prevents Endianness errors in legacy C-suites).

11. Execute NIST SP 800-22 STS (See below) on the generated text files.

12. visuals/plot_nist_pval.py & visuals/compare_ntro.py: Plot final P-values, auto-correlation lags, absolute bias, and calculate final MB/s throughput.
  

## How to Exploit NIST Validation LibrariesTo keep this repository lightweight and strictly focused on novel research, the official NIST C/C++ libraries are not tracked here. They must be compiled locally using Windows Subsystem for Linux (WSL - Ubuntu 24.04).

1. NIST SP 800-90B (Min-Entropy Assessment)Used to calculate the $H_{min}$ of the raw quantum hardware.Bash# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential libbz2-dev libssl-dev libjsoncpp-dev libdivsufsort-dev libmpfr-dev libgmp-dev

# Clone and compile
git clone https://github.com/usnistgov/SP800-90B_EntropyAssessment.git
cd SP800-90B_EntropyAssessment/cpp
make

# Execute non-IID evaluation on unpacked binary (1 bit per symbol)
./ea_non_iid -v -c /path/to/your/unpacked_bitstream.bin 1
2. NIST SP 800-22 (Statistical Test Suite)Used to verify the cryptographic uniformity of the final FFT-whitened output.Download the suite from the Official NIST CSRC Website.Extract the folder and compile the Linux binary:Bashcd sts-2.1.2
make clean
make

# Execute the assessment
./assess 1000000
# Follow prompts: Select [0] for input file, [1] for all tests, and [1] for ASCII format.
 
 
 
 
### AcknowledgmentsThis research and engineering effort was conducted during my internship at Deloitte Portugal Consulting (Cyber and Telecom Networks). Special thanks to the leadership and technical mentors at The Hoop lab for providing the infrastructure, hardware bridging, and strategic guidance necessary to complete this experiment.
