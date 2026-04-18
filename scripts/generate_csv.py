import pandas as pd
import numpy as np
import os

def generate_test_data(rows=1000):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, "data")
    filename = os.path.join(data_dir, "test_traffic.csv")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data = []
    for _ in range(rows):
        scenario = np.random.choice(['benign', 'malicious', 'suspicious'])
        
        if scenario == 'benign':
            # [Log Freq, Total Bytes, PSD Peak, Duration, Client Size, Entropy, Symmetry, Verdict]
            row = [0.1, 12735, 0.05, 46, 852, 3.5, 0.02, 'benign']
        elif scenario == 'malicious':
            row = [5.5, 400, 0.95, 1200, 32, 7.9, 1.0, 'malicious']
        else:
            row = [2.5, 2500, 0.68, 300, 128, 5.0, 0.85, 'suspicious']
        data.append(row)

    cols = ["Log Frequency", "Total Bytes", "PSD Peak", "Duration", "Client Size", "Entropy", "Packet Symmetry", "Verdict"]
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(filename, index=False)
    print(f"✅ Created CSV with 'Verdict' and 'Packet Symmetry' at: {filename}")

if __name__ == "__main__":
    generate_test_data()