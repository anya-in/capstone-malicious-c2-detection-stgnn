import numpy as np

try:
    import streamlit as st
except Exception:  # Fallback stub for environments where Streamlit isn't installed (e.g., linters)
    from types import SimpleNamespace

    def _noop(*args, **kwargs):
        return None

    def _false(*args, **kwargs):
        return False

    class _Progress:
        def progress(self, *a, **k):
            return None

    def _cache_resource(fn):
        return fn

    # Minimal stub exposing only the attributes used in this file
    st = SimpleNamespace(
        set_page_config=_noop,
        cache_resource=_cache_resource,
        title=_noop,
        markdown=_noop,
        file_uploader=lambda *a, **k: None,
        write=_noop,
        error=_noop,
        button=_false,
        progress=lambda *a, **k: _Progress(),
        columns=lambda n: [SimpleNamespace(metric=_noop) for _ in range(n)],
        metric=_noop,
        dataframe=_noop,
        download_button=_noop,
    )
import torch
import pandas as pd
import os
import numpy as np
import sys
from torch_geometric.data import Data

# --- CONFIG & PATHS ---
st.set_page_config(page_title="ST-GNN Batch Engine", page_icon="🚀", layout="wide")

# Ensure the model core is accessible
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src', 'models')))
from stgnn_core import STGNN_C2_Detector

@st.cache_resource
def load_stgnn():
    # Use absolute pathing to find the model weights
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_weights = os.path.join(base_path, 'models', 'stgnn_final_balanced.pth')
    
    if not os.path.exists(model_weights):
        st.error(f"⚠️ Model file NOT found at: {model_weights}")
        return None

    model = STGNN_C2_Detector(input_dim=6)
    model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    return model

# Calibration constants
MEANS = torch.tensor([6.197e-09, 12734.86, 0.0, 46.03, 852.60, 3.383e-08])
STDS = torch.tensor([1.00, 6276.81, 0.1, 337.83, 653.97, 0.96])

st.title("🛡️ ST-GNN Batch C2 Detector")
st.markdown("Upload bulk network telemetry to identify hidden C2 beacons in encrypted traffic.")

uploaded_file = st.file_uploader("Upload Network Logs (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"📊 Loaded {len(df)} traffic flows.")
    
    if st.button("🚀 Start Deep Analysis"):
        model = load_stgnn()
        results = []
        progress_bar = st.progress(0)
        
        # Batch processing loop
        for i, row in df.iterrows():
            # Normalize and prepare graph object
            #x = torch.tensor([row.values[:6]], dtype=torch.float)
            x_np = np.array(row.values[:6], dtype=np.float32).reshape(1, -1)
            x = torch.from_numpy(x_np)
            x_norm = (x - MEANS) / (STDS + 1e-6)
            data = Data(x=x_norm, edge_index=torch.tensor([[0],[0]]))
            
            with torch.no_grad():
                logits = model(data)
                # Temperature Scale T=2.0 for Gray Area sensitivity
                prob = torch.softmax(logits / 2.0, dim=1)[0][1].item()
            
            # 3-Tier Classification
            status = "MALICIOUS" if prob > 0.50 else ("SUSPICIOUS" if prob > 0.05 else "BENIGN")
            results.append({"Risk Score": f"{prob*100:.1f}%", "Verdict": status})
            progress_bar.progress((i + 1) / len(df))

        # Merge results and display
        res_df = pd.concat([df, pd.DataFrame(results)], axis=1)
        
        # Visual Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Malicious Detected", len(res_df[res_df['Verdict'] == 'MALICIOUS']))
        c2.metric("Suspicious Flagged", len(res_df[res_df['Verdict'] == 'SUSPICIOUS']))
        c3.metric("Benign Flows", len(res_df[res_df['Verdict'] == 'BENIGN']))

        st.dataframe(res_df.style.applymap(
            lambda x: 'background-color: #ff4b4b' if x == 'MALICIOUS' else 
                      ('background-color: #ffa500' if x == 'SUSPICIOUS' else ''),
            subset=['Verdict']
        ))

        st.download_button("📥 Download Results", res_df.to_csv(index=False), "c2_analysis_results.csv")