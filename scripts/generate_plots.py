import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Create the folder for your poster assets
os.makedirs('src/visualizations', exist_ok=True)

# Load the data you generated earlier
df = pd.read_csv('data/test_traffic.csv')

# --- PLOT 1: PSD Peak (Heartbeat) Separability ---
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df, x='PSD Peak', hue='Verdict', fill=True, palette='viridis')
plt.title('Spectral Separability: C2 Heartbeats vs. Benign Noise')
plt.savefig('src/visualizations/psd_separability.png')
print("✅ Saved: psd_separability.png")

# --- PLOT 2: Feature Correlation (The Heatmap) ---
plt.figure(figsize=(10, 8))
# Drop non-numeric for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdYlGn', fmt='.2f')
plt.title('Feature Interaction Matrix (ST-GNN Inputs)')
plt.savefig('src/visualizations/feature_correlation.png')
print("✅ Saved: feature_correlation.png")

# --- PLOT 3: Packet Symmetry (The 1:1 Evidence) ---
plt.figure(figsize=(8, 6))
sns.boxplot(x='Verdict', y='Packet Symmetry', data=df, palette='magma')
plt.title('Forensic Attribution: Packet Symmetry (1:1 Ratio)')
plt.savefig('src/visualizations/symmetry_boxplot.png')
print("✅ Saved: symmetry_boxplot.png")

print("\n🚀 All plots ready in src/visualizations/ for your poster!")