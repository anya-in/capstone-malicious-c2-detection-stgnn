import torch
import os

# Load your actual graph data
graph_path = os.path.join('data', 'processed', 'network_graph.pt')
data = torch.load(graph_path, weights_only=False)

# Calculate the exact means and stds the model was trained on
means = data.x.mean(dim=0)
stds = data.x.std(dim=0) + 1e-6

print("--- 📋 COPY THESE INTO YOUR APP.PY ---")
print(f"MEANS: {means.tolist()}")
print(f"STDS: {stds.tolist()}")