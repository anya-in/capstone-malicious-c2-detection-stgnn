import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

"""
PROJECT: Resilient Detection of Malicious C2 Beacons (Team 11)
TOPIC: "Total Eclipse" Preprocessing Phase
PURPOSE: This script blinds the model to ECH-visible data and applies 
         cybersecurity-specific feature engineering to enhance ST-GNN signals.
"""

# 1. Path Management
# Path is relative to running the script from the 'src' folder
data_path = os.path.join('..', 'data', 'raw', 'ml_features_and_labels.csv')
output_dir = os.path.join('..', 'data', 'processed')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def run_extensive_preprocessing():
    print("--- Starting Project-Specific Preprocessing ---")
    
    # Load raw telemetry data
    df = pd.read_csv(data_path)
    
    # PHASE A: THE TOTAL ECLIPSE (Blinding)
    # We remove e4 because ECH (Encrypted Client Hello) makes this data 
    # invisible to network defenders. Our ST-GNN must learn without it.
    blinded_cols = ['e4_entropy_h']
    df = df.drop(columns=blinded_cols)
    print(f"[Topic Specific] Blinded ECH-visible features: {blinded_cols}")

    # PHASE B: FEATURE ENGINEERING (Cybersecurity Specific)
    # C2 Beacons are often defined by their timing (IAT) and payload consistency.
    # We ensure timing features (e6b) are treated as float for precision.
    df['e6b_flow_duration_ms'] = df['e6b_flow_duration_ms'].astype(float)
    
    # PHASE C: LOG TRANSFORMATION
    # Network traffic durations often vary by orders of magnitude (Skewed).
    # Applying log1p helps the GNN converge faster on "Heartbeat" patterns.
    df['e6b_log'] = np.log1p(df['e6b_flow_duration_ms'])
    print("[Technical] Applied Log Transformation to Flow Durations.")

    # PHASE D: ROBUST SCALING
    # We scale entropy (e5) and log-timing (e6b_log) so the GNN 
    # weights them equally during the spatial-temporal message passing.
    scaler = StandardScaler()
    cols_to_scale = ['e5_entropy_c', 'e6b_log']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"[Technical] Normalized {cols_to_scale} for GNN compatibility.")

    # PHASE E: FINAL CLEANING
    # Ensure no Infinity or NaN values remain from transformations
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Save to 'processed' folder for the GNN phase
    output_path = os.path.join(output_dir, 'stgnn_ready_data.csv')
    df.to_csv(output_path, index=False)
    print(f"--- Preprocessing Complete: {output_path} ---")

if __name__ == "__main__":
    run_extensive_preprocessing()