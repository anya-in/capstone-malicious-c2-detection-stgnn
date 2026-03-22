import pandas as pd
import numpy as np
import torch
import os
from torch_geometric.data import Data
from scipy.signal import welch

def get_dominant_frequency(series):
    x = series.to_numpy()
    if len(x) < 5: return 0.0
    try:
        f, Pxx = welch(x, fs=1.0, nperseg=min(len(x), 256))
        return f[np.argmax(Pxx)]
    except: return 0.0

def generate_graph_data():
    print(f"--- 🚀 Final Build: Mapping 40k Rows to ST-GNN ---")
    
    base_path = os.getcwd()
    data_path = os.path.join(base_path, 'data', 'processed', 'stgnn_ready_data.csv')
    output_path = os.path.join(base_path, 'data', 'processed', 'network_graph.pt')
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found!")
        return

    df = pd.read_csv(data_path)

    # 1. Temporal Analysis (PSD)
    print("Step 1: Calculating Heartbeat Frequencies (PSD)...")
    rhythms = df.groupby('ID')['e6b_log'].apply(lambda s: get_dominant_frequency(s)).to_dict()
    df['psd_peak'] = df['ID'].map(rhythms)

    # 2. Spatial Analysis (Linear Chain Optimization)
    print("Step 2: Building Sparse Adjacency Matrix...")
    edge_index = []
    taxonomy_groups = df.groupby('taxonomy').groups
    for indices in taxonomy_groups.values():
        idx_list = list(indices)
        if len(idx_list) > 1:
            for k in range(len(idx_list) - 1):
                edge_index.append([idx_list[k], idx_list[k+1]])
                edge_index.append([idx_list[k+1], idx_list[k]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 3. Node Features (Your Exact Column Names)
    print("Step 3: Preparing Node Features...")
    feature_cols = [
        'e6b_log', 'e2c_total_bytes', 'psd_peak', 
        'e6b_flow_duration_ms', 'e2_client_size', 'e5_entropy_c'
    ]
    
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(df['label'].values, dtype=torch.long)

    # 4. Save
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graph_data, output_path)
    print(f"✅ Success! Graph saved with {x.shape[1]} features and {edge_index.shape[1]} edges.")

if __name__ == "__main__":
    generate_graph_data()