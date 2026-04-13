import pandas as pd
import numpy as np
import os
import sys

# GUARANTEED MESSAGE
print("\n" + "="*40)
print("🚀 GNN DATA GENERATOR BOOTING UP...")
print(f"📍 SCRIPT LOCATION: {os.path.abspath(__file__)}")
print("="*40 + "\n")

def generate_test_data(rows=1000):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, "data")
    filename = os.path.join(data_dir, "test_traffic.csv")

    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"📁 Creating folder: {data_dir}")

        # Data creation logic...
        data = []
        for _ in range(rows):
            scenario = np.random.choice(['benign', 'malicious', 'suspicious', 'large_file'])
            if scenario == 'benign': row = [0.1, 12735, 0.0, 46, 852, 3.5]
            elif scenario == 'malicious': row = [5.5, 400, 0.95, 1200, 32, 7.9]
            elif scenario == 'suspicious': row = [2.5, 2500, 0.68, 300, 128, 5.0]
            else: row = [1.0, 50000, 0.85, 1500, 852, 4.0]
            data.append(row)

        df = pd.DataFrame(data, columns=["Log Frequency", "Total Bytes", "PSD Peak", "Duration", "Client Size", "Entropy"])
        df.to_csv(filename, index=False)
        
        print(f"✅ SUCCESS! Generated {len(df)} rows.")
        print(f"📄 PATH: {filename}\n")
    
    except Exception as e:
        print(f"❌ ERROR OCCURRED: {e}")

if __name__ == "__main__":
    generate_test_data()