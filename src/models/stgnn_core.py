import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import GRU, Linear

class STGNN_C2_Detector(torch.nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super(STGNN_C2_Detector, self).__init__()
        # Spatial: Learns behavioral clusters
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Temporal: Analyzes the 'Pulse' of the traffic
        self.gru = GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Output: Binary Classification (Benign vs Malicious)
        self.fc = Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Spatial Phase
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        # Temporal Phase
        x, _ = self.gru(x.unsqueeze(1))
        x = x.squeeze(1)
        
        return F.log_softmax(self.fc(x), dim=1)