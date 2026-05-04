import os
import sys
import torch
from torchview import draw_graph

# ---  FIX: ADD MODELS PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
models_path = os.path.join(root_path, 'src', 'models')
if models_path not in sys.path:
    sys.path.append(models_path)

from stgnn_core import STGNN_C2_Detector

# ---  GENERATE DIAGRAM ---
model = STGNN_C2_Detector(input_dim=6)
batch_size = 1

# This creates the visual representation
try:
    model_graph = draw_graph(
        model, 
        input_size=(batch_size, 6), 
        device='cpu', 
        save_graph=True, 
        filename="stgnn_architecture"
    )
    print(" Success! Check your project folder for 'stgnn_architecture.gv.pdf' or '.png'")
except Exception as e:
    print(f" Error: {e}")
