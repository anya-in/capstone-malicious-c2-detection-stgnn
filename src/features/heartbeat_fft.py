import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_fft_signal():
    print("--- Isolating Periodic C2 Heartbeats via FFT ---")

    # 1. Path Setup (Absolute for your system)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) 
    
    data_path = os.path.join(project_root, 'data', 'processed', 'stgnn_ready_data.csv')
    output_dir = os.path.join(project_root, 'src', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return

    # 2. Load Data
    df = pd.read_csv(data_path)

    # 3. Filter for ALL Malicious Traffic
    # We use the whole malicious stream to ensure we have enough data points for a rhythm
    malicious_df = df[df['label'] == 1].copy()

    if len(malicious_df) > 10:
        print(f"Analyzing {len(malicious_df)} malicious samples...")
        
        # 4. Extract the signal from 'e6b_log' (Timing/Sequence Feature)
        signal = malicious_df['e6b_log'].values
        
        # STEP A: Remove DC Offset (Center the signal around zero)
        # This is critical to see the "spike" instead of a huge bar at 0
        signal_centered = signal - np.mean(signal)

        # STEP B: Perform Fast Fourier Transform (FFT)
        n = len(signal_centered)
        fft_values = np.fft.fft(signal_centered)
        frequencies = np.fft.fftfreq(n)

        # Masking: We only care about positive frequencies (the real half)
        pos_mask = frequencies > 0
        plot_freq = frequencies[pos_mask]
        plot_mag = np.abs(fft_values)[pos_mask]

        # 5. Plotting the Power Spectrum
        plt.figure(figsize=(12, 6))
        plt.plot(plot_freq, plot_mag, color='#2980b9', linewidth=1.5, label='Signal Power')
        
        # Find and annotate the highest spike (The Heartbeat)
        peak_idx = np.argmax(plot_mag)
        peak_freq = plot_freq[peak_idx]
        
        plt.scatter(peak_freq, plot_mag[peak_idx], color='red', s=60, zorder=5)
        plt.annotate(f'C2 Heartbeat Detected\nFreq: {peak_freq:.4f}', 
                     xy=(peak_freq, plot_mag[peak_idx]), 
                     xytext=(peak_freq + 0.03, plot_mag[peak_idx] * 0.9),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1))

        plt.title('Signal Isolation: Frequency Analysis of Malicious C2 Beacons', fontsize=14)
        plt.xlabel('Frequency (Cycles per Packet)', fontsize=12)
        plt.ylabel('Magnitude (Signal Strength)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # 6. Save and Show
        save_path = os.path.join(output_dir, 'fft_heartbeat_analysis.png')
        plt.savefig(save_path)
        print(f"Success! Plot saved to: {save_path}")
        
        # This will open a window with the graph
        plt.show()

    else:
        print(f"Not enough malicious samples (found {len(malicious_df)}) to perform FFT.")

if __name__ == "__main__":
    analyze_fft_signal()