import time
import os
import sys

print("=== INITIALIZING HARDWARE BRIDGE ===")
print(f"-> Using Python: {sys.version.split()[0]}")

# 1. Define the exact paths
SWABIAN_PYTHON_PATH = r"C:\Program Files\Swabian Instruments\Time Tagger\driver\x64\python3.10"
SWABIAN_ROOT_PATH = r"C:\Program Files\Swabian Instruments\Time Tagger"

# 2. Add the Python module to sys.path
if SWABIAN_PYTHON_PATH not in sys.path:
    sys.path.append(SWABIAN_PYTHON_PATH)

# 3. Add the DLL directories (Crucial for Python 3.8+)
if hasattr(os, 'add_dll_directory'):
    print("-> Adding strict DLL directory permissions...") 
    os.add_dll_directory(SWABIAN_ROOT_PATH)
    os.add_dll_directory(SWABIAN_PYTHON_PATH)

# 4. Unfiltered Import (If this crashes, we will see the exact C++ error)
print("-> Importing TimeTagger C++ Engine...")
from TimeTagger import createTimeTagger, FileWriter, freeTimeTagger
print("-> Engine Loaded Successfully!\n")



def acquire_raw_timestamps(duration_sec, filename, channels):
    """
    Connects exclusively to the FPGA, opens a direct binary stream 
    to disk (bypassing CPU bottlenecks), and closes gracefully.
    """
    print("\n=== QUANTUM RAW DATA ACQUISITION (RAW DUMP) ===")
    print("-> Attempting connection to the Time Tagger...")
    
    try:
        # Takes physical control of the hardware
        tagger = createTimeTagger()
    except Exception as e:
        print("\n[CONNECTION ERROR] Cannot access the FPGA.")
        print(f"Details: {e}")
        print("MANDATORY SOLUTION: You must completely close the Thorlabs EDU-QOP1")
        print("GUI software before running this script.")
        return

    print(f"-> Connection established. Initializing binary stream on channels {channels}...")
    print(f"-> Target file: {filename}")
    
    # FileWriter delegates writing directly to the C++/FPGA backend
    writer = FileWriter(tagger, filename, channels)
    
    print(f"-> Acquisition in progress for {duration_sec} seconds. DO NOT CLOSE THIS WINDOW...")
    
    try:
        # The loop merely provides visual feedback to the researcher;
        # actual writing is asynchronous in the background.
        for i in range(duration_sec):
            time.sleep(1)
            if (i + 1) % 10 == 0:
                print(f"   ... {i + 1} seconds elapsed ...")
                
    except KeyboardInterrupt:
        # This block intercepts the manual stop command (Ctrl+C)
        print("\n[WARNING] Manual interruption requested by the user (Ctrl+C).")
        print("-> Saving partial data...")

    finally:
        # The 'finally' block is ALWAYS executed, whether the time ends naturally,
        # or upon manual interruption/error. Vital to avoid leaving the 
        # USB port locked in a "zombie" state.
        print("-> Closing data stream and flushing hardware buffers...")
        writer.stop()
        freeTimeTagger(tagger)
        
        if os.path.exists(filename):
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            print("-> Acquisition completed successfully and hardware released.")
            print(f"-> Generated binary file size: {file_size_mb:.2f} MB\n")
        else:
            print("-> [ERROR] The binary file was not generated.\n")

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    # --- EXPERIMENTAL CONFIGURATION ---
    
    # Choose the physical (BNC) channels to listen to. 
    # Typically 1 = Detector T (Trigger), 2 = Detector A (Alice).
    TARGET_CHANNELS = [1,2] 
    
    # Output filename. The .ttbin extension stands for TimeTagger Binary.
    OUTPUT_FILE = "half_raw_photons.ttbin" 
    
    # Experiment duration in seconds.
    # Start with 60 seconds to test the pipeline and evaluate the MB footprint.
    ACQUISITION_TIME = 1800


    # Execute the acquisition
    acquire_raw_timestamps(ACQUISITION_TIME, OUTPUT_FILE, TARGET_CHANNELS)