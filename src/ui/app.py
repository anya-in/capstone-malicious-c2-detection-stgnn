import streamlit as st
import torch
import os
import sys
import numpy as np
import pandas as pd
from torch_geometric.data import Data

# --- 🛠️ PROJECT PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
models_path = os.path.join(root_path, 'src', 'models')
if models_path not in sys.path:
    sys.path.append(models_path)

from stgnn_core import STGNN_C2_Detector

# --- 🎨 UI CONFIG ---
st.set_page_config(page_title="ST-GNN C2 Analysis", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_trained_model():
    model_weights = os.path.join(root_path, 'models', 'stgnn_final_balanced.pth')
    if not os.path.exists(model_weights):
        st.error("⚠️ Model file not found. Ensure .pth is in /models/ folder.")
        return None
    model = STGNN_C2_Detector(input_dim=6)
    model.load_state_dict(torch.load(model_weights, weights_only=False))
    model.eval()
    return model

model = load_trained_model()

# --- 📊 CALIBRATION PARAMETERS ---
MY_MEANS = torch.tensor([6.197e-09, 12734.86, 0.0, 46.03, 852.60, 3.383e-08], dtype=torch.float)
MY_STDS = torch.tensor([1.00, 6276.81, 0.1, 337.83, 653.97, 0.96], dtype=torch.float)

# --- 🛰️ NAVIGATION ---
tab1, tab2 = st.tabs(["Single Traffic Flow", "Batch Log Processing"])

# --- TAB 1: INDIVIDUAL ANALYSIS ---
with tab1:
    st.header("Individual Flow Inspection")
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        f1 = st.number_input("Log Frequency", value=0.0)
        f2 = st.number_input("Total Bytes", value=12735)
        f3_psd = st.slider("PSD Peak (Heartbeat)", 0.0, 1.0, 0.0, format="%.4f")
        f4 = st.number_input("Duration (ms)", value=46)
        f5 = st.number_input("Client Hello Size", value=852)
        f6_entropy = st.slider("Payload Entropy", 0.0, 8.0, 3.5)
        analyze = st.button("🔍 Run GNN Analysis")

    with col_result:
        if analyze and model is not None:
            raw = torch.tensor([[f1, f2, f3_psd, f4, f5, f6_entropy]], dtype=torch.float)
            norm = (raw - MY_MEANS) / (MY_STDS + 1e-6)
            
            edge_idx = torch.tensor([[0], [0]], dtype=torch.long)
            data_obj = Data(x=norm, edge_index=edge_idx)

            with torch.no_grad():
                logits = model(data_obj)
                
                # --- 🔥 TEMPERATURE FIX ---
                # Increase this number (e.g., to 3.0 or 5.0) to make the model even "shier" 
                # and more prone to showing middle-range percentages.
                temp = 2.0 
                prob = torch.softmax(logits / temp, dim=1)[0][1].item()

            # Updated display logic for the 3-tier warning system
            if prob > 0.50: 
                st.error(f"### 🚨 MALICIOUS BEACON: {prob*100:.1f}% Risk")
                st.write("**Verdict:** High confidence C2 detection.")
            elif prob > 0.15:
                st.warning(f"### ⚠️ SUSPICIOUS ACTIVITY: {prob*100:.1f}% Risk")
                st.write("**Verdict:** Anomalous behavior detected. Possible beaconing.")
            else:
                st.success(f"### ✅ BENIGN TRAFFIC: {prob*100:.1f}% Risk")
                st.write("**Verdict:** Normal traffic pattern.")
            
            st.progress(min(prob, 1.0))
            st.bar_chart(norm.numpy().flatten())

# --- TAB 2: BATCH PROCESSING ---
with tab2:
    st.header("Bulk Log Analysis")
    file = st.file_uploader("Upload CSV", type="csv")
    
    if file is not None and model is not None:
        batch_df = pd.read_csv(file)
        if st.button("🚀 Process Batch Data"):
            labels = []
            for index, row in batch_df.iterrows():
                row_tensor = torch.tensor([row.values[:6]], dtype=torch.float)
                norm_row = (row_tensor - MY_MEANS) / (MY_STDS + 1e-6)
                d = Data(x=norm_row, edge_index=torch.tensor([[0], [0]]))
                with torch.no_grad():
                    out = model(d)
                    p = torch.softmax(out / 2.0, dim=1)[0][1].item()
                    labels.append("MALICIOUS" if p > 0.50 else ("SUSPICIOUS" if p > 0.15 else "BENIGN"))
            
            batch_df['ST-GNN Prediction'] = labels
            st.dataframe(batch_df)

st.divider()
st.caption("Capstone Project 2026 | Spatio-Temporal GNN Malicious C2 Detection")