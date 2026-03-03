import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_time_rhythm():
    print("--- Visualizing Temporal Rhythm (Time Domain) ---")

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

    # 3. Prepare the Data
    # Filtering malicious vs benign to see the contrast
    malicious = df[df['label'] == 1].copy()
    benign = df[df['label'] == 0].copy()

    # 4. Create the Plot
    plt.figure(figsize=(14, 6))

    # Plot Benign Traffic as a gray background 'cloud'
    plt.scatter(benign.index, benign['e6b_log'], 
                color='gray', alpha=0.15, s=8, label='Benign Traffic (Chaotic)')

    # Plot Malicious Traffic as distinct red pulses
    plt.scatter(malicious.index, malicious['e6b_log'], 
                color='#e74c3c', alpha=0.8, s=12, label='C2 Malicious (Periodic Pulse)')

    # 5. Formatting
    plt.title('Time-Domain Signal Isolation: The C2 Heartbeat Pulse', fontsize=14)
    plt.xlabel('Packet Sequence (Arrival Order)', fontsize=12)
    plt.ylabel('Timing Feature Value (e6b_log)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Place legend in a spot that doesn't block data
    plt.legend(loc='upper right', frameon=True, shadow=True)

    # 6. Save and Show
    save_path = os.path.join(output_dir, 'time_domain_rhythm.png')
    plt.savefig(save_path, dpi=300) # Higher DPI for better report quality
    print(f"Success! Time-domain plot saved to: {save_path}")
    
    # Opens the plot window
    plt.show()

if __name__ == "__main__":
    plot_time_rhythm()