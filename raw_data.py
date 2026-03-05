import time
import os
import sys

# Tentativo di importazione con fallback chiaro per evitare panico da terminale
try:
    from TimeTagger import createTimeTagger, FileWriter, freeTimeTagger
except ImportError:
    print("[ERRORE CRITICO] Modulo TimeTagger non trovato.")
    print("Assicurati di essere dentro il tuo ambiente virtuale (.venv) su Windows")
    print("e di aver installato le librerie Swabian Instruments.")
    sys.exit(1)

def acquire_raw_timestamps(duration_sec, filename, channels):
    """
    Si connette in modo esclusivo all'FPGA, apre un flusso binario diretto
    su disco (bypassando colli di bottiglia della CPU) e chiude con grazia.
    """
    print("\n=== ACQUISIZIONE DATI QUANTISTICI GREZZI (RAW DUMP) ===")
    print("-> Tentativo di connessione al Time Tagger...")
    
    try:
        # Prende il controllo fisico dell'hardware
        tagger = createTimeTagger()
    except Exception as e:
        print("\n[ERRORE DI CONNESSIONE] Impossibile accedere all'FPGA.")
        print(f"Dettaglio: {e}")
        print("SOLUZIONE OBBLIGATORIA: Devi chiudere completamente il software")
        print("interfaccia grafica Thorlabs EDU-QOP1 prima di lanciare questo script.")
        return

    print(f"-> Connessione stabilita. Inizializzazione stream binario sui canali {channels}...")
    print(f"-> File di destinazione: {filename}")
    
    # Il FileWriter delega la scrittura direttamente al backend C++/FPGA
    writer = FileWriter(tagger, filename, channels)
    
    print(f"-> Acquisizione in corso per {duration_sec} secondi. NON CHIUDERE LA FINESTRA...")
    
    try:
        # Il loop serve solo per dare un feedback visivo al ricercatore,
        # la scrittura vera e propria avviene in modo asincrono in background.
        for i in range(duration_sec):
            time.sleep(1)
            if (i + 1) % 10 == 0:
                print(f"   ... {i + 1} secondi trascorsi ...")
                
    except KeyboardInterrupt:
        # Questo blocco intercetta il comando di blocco manuale (Ctrl+C)
        print("\n[ATTENZIONE] Interruzione manuale richiesta dall'utente (Ctrl+C).")
        print("-> Salvataggio dei dati parziali in corso...")

    finally:
        # Il blocco 'finally' viene eseguito SEMPRE, sia alla fine del tempo,
        # sia in caso di interruzione manuale o errore imprevisto.
        # È vitale per non lasciare la porta USB bloccata in stato "zombie".
        print("-> Chiusura del flusso dati e svuotamento dei buffer hardware...")
        writer.stop()
        freeTimeTagger(tagger)
        
        if os.path.exists(filename):
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            print("-> Acquisizione terminata correttamente e hardware rilasciato.")
            print(f"-> Dimensione file binario generato: {file_size_mb:.2f} MB\n")
        else:
            print("-> [ERRORE] Il file binario non è stato generato.\n")

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    # --- CONFIGURAZIONE SPERIMENTALE ---
    
    # Scegli i canali fisici (BNC) da ascoltare. 
    # Tipicamente 1 = Rivelatore T (Trigger), 2 = Rivelatore A (Alice).
    TARGET_CHANNELS = [1, 2, 3] 
    
    # Nome del file di output. L'estensione .ttbin sta per TimeTagger Binary.
    OUTPUT_FILE = "raw_photons.ttbin" 
    
    # Durata dell'esperimento in secondi.
    # Inizia con 60 secondi per testare che tutto funzioni e valutare il peso in MB.
    ACQUISITION_TIME = 60 
    
    # Esecuzione
    acquire_raw_timestamps(ACQUISITION_TIME, OUTPUT_FILE, TARGET_CHANNELS)