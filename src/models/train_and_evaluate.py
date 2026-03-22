import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from stgnn_core import STGNN_C2_Detector

def run_pipeline():
    print("--- 🧠 ST-GNN: Final Balanced Training (The IEEE Edition) ---")
    
    # 1. Load Data
    graph_path = os.path.join('data', 'processed', 'network_graph.pt')
    data = torch.load(graph_path, weights_only=False)
    
    # Feature Normalization
    means = data.x.mean(dim=0)
    stds = data.x.std(dim=0) + 1e-6
    data.x = (data.x - means) / stds

    # IMPROVEMENT: Balanced Weights (1.0 vs 3.5)
    # This prevents Precision from tanking while keeping Recall high.
    weights = torch.tensor([1.0, 3.5]) 
    
    model = STGNN_C2_Detector(input_dim=data.num_node_features)
    
    # Regularization (weight_decay) stops overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 2. Training Loop
    model.train()
    for epoch in range(1, 201): # Increased to 200 for better stability
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            preds = out.argmax(dim=1)
            c2_mask = (data.y == 1)
            c2_recall = (preds[c2_mask] == 1).sum().item() / c2_mask.sum().item()
            print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | C2 Recall: {c2_recall:.2%}')

    # 3. Final Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data)
        preds = out.argmax(dim=1)
    
    y_true, y_pred = data.y.numpy(), preds.numpy()
    
    # 4. Save Final Visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'C2'], yticklabels=['Benign', 'C2'])
    plt.title('Final Balanced Detection Matrix')
    plt.savefig('src/visualizations/final_confusion_matrix.png')

    print("\n--- Final Results for Paper ---")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'C2']))
    
    # Save the 'Brain' for the Dashboard
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/stgnn_final_balanced.pth')
    print("✅ Model saved as 'models/stgnn_final_balanced.pth'")

if __name__ == "__main__":
    run_pipeline()