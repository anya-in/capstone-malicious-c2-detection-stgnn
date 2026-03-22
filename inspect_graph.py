import torch

# Load the file with the security fix
data = torch.load('data/processed/network_graph.pt', weights_only=False)

print("--- 🔍 Graph Data Inspection ---")
print(f"Object Type: {type(data)}")
print("-" * 30)
# 1. Check the Nodes and Features
print(f"Number of Nodes (Flows): {data.num_nodes}")
print(f"Number of Features per Node: {data.num_node_features}")
print(f"Shape of Feature Matrix (x): {data.x.shape}")

# 2. Check the Edges (Connections)
print(f"Number of Edges (Connections): {data.num_edges}")
print(f"Shape of Edge Index: {data.edge_index.shape}")

# 3. Check the Labels
unique_labels, counts = torch.unique(data.y, return_counts=True)
print("-" * 30)
for label, count in zip(unique_labels, counts):
    label_name = "Malicious (C2)" if label == 1 else "Benign"
    print(f"Label {label} [{label_name}]: {count.item()} instances")

# 4. Peek at the first node's data
# Columns: e6b_log, e2c_total_bytes, psd_peak, e6b_flow_duration_ms, e2_client_size, e5_entropy_c
print("-" * 30)
print("Example Node Feature Vector (First Flow):")
print(data.x[0])