import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Path Setup
data_path = os.path.join('..', 'data', 'processed', 'stgnn_ready_data.csv')
output_dir = os.path.join('..', 'src', 'visualizations')

# 2. Load the blinded data
df = pd.read_csv(data_path)

def analyze_fft_signal():
    print("--- Isolating Periodic C2 Heartbeats via FFT ---")
    
    # Filter for a sample malicious session
    malicious_sample = df[df['label'] == 1]['e6b_log'].values
    
    if len(malicious_sample) > 0:
        # Perform Fast Fourier Transform (FFT)
        # This converts time-based signals into frequency-based signals
        fft_values = np.fft.fft(malicious_sample)
        frequencies = np.fft.fftfreq(len(malicious_sample))
        
        # Plotting the "Power Spectrum"
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, np.abs(fft_values))
        plt.title('Signal Isolation: Frequency Analysis of C2 Traffic')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude (Signal Strength)')
        plt.grid(True)
        
        # Save to your existing visualizations folder
        save_path = os.path.join(output_dir, 'fft_heartbeat_analysis.png')
        plt.savefig(save_path)
        print(f"FFT Analysis complete. Saved to: {save_path}")
    else:
        print("No malicious samples found for FFT analysis.")

if __name__ == "__main__":
    analyze_fft_signal()