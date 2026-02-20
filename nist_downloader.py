import requests
import pandas as pd
import time

def fetch_nist_beacon_data(filename="nist_random_data.csv", pulses=50):
    """
    Connects to the NIST Randomness Beacon API.
    Uses 'pulseIndex' to navigate the beacon chain backwards in time,
    collecting 512 bits of pure high-entropy data per pulse.
    """
    print(f"-> Connecting to NIST Randomness Beacon (Pulse Chain 1)...")
    
    current_url = "https://beacon.nist.gov/beacon/2.0/chain/1/pulse/last"
    all_bits = []
    successful_pulses = 0
    
    while successful_pulses < pulses:
        print(f"   Fetching pulse {successful_pulses + 1}/{pulses}...", end=" ", flush=True)
        
        try:
            response = requests.get(current_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pulse_data = data.get('pulse')
                
                if pulse_data:
                    hex_string = pulse_data.get('outputValue')
                    
                    if hex_string:
                        # Convert Hex to Binary and ensure 512 bits length
                        binary_string = bin(int(hex_string, 16))[2:].zfill(512)
                        all_bits.extend([int(b) for b in binary_string])
                        
                        # --- LA CORREZIONE CRITICA ---
                        current_index = pulse_data.get('pulseIndex')
                        
                        if current_index is not None:
                            # Forziamo il dato a diventare un numero intero matematico
                            current_index = int(current_index)
                            
                            # Ora la sottrazione è sicura
                            current_url = f"https://beacon.nist.gov/beacon/2.0/chain/1/pulse/{current_index - 1}"
                            successful_pulses += 1
                            print("SUCCESS!")
                        else:
                            print("FAILED (No pulseIndex found).")
                            break
                    else:
                        print("FAILED (No outputValue found).")
                        break
                else:
                    print("FAILED (No 'pulse' block).")
                    break
            else:
                print(f"FAILED (HTTP {response.status_code}).")
                break
                
        except Exception as e:
            # Abbiamo migliorato l'output dell'errore per vedere il VERO motivo del blocco
            print(f"CONNECTION ERROR: {type(e).__name__} - {e}")
            time.sleep(2)
            
    if all_bits:
        print(f"\n-> Download complete! Total bits extracted: {len(all_bits)}")
        df = pd.DataFrame(all_bits, columns=["Raw_Bits"])
        df.to_csv(filename, index=False)
        print(f"-> Saved data to: {filename}")
    else:
        print("\n-> Error: No data was collected.")

if __name__ == "__main__":
    fetch_nist_beacon_data()