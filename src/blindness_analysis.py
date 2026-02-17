import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Path Setup: code is in src/, data is in data/raw/
# We use '..' to go up one level from src to the root
data_path = os.path.join('..', 'data', 'raw', 'ml_features_and_labels.csv')
output_dir = 'visualizations' # This will be relative to where the script is (inside src/)

# 2. Automatically create the visualizations folder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Load data
df = pd.read_csv(data_path)

# 4. Create the Visualization
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='e4_entropy_h', hue='label', fill=True, common_norm=False)
plt.title('The "Blindness" Proof: SNI Entropy (e4) Leakage')
plt.xlabel('Entropy (Bits)')

# 5. Save to src/visualizations/
save_path = os.path.join(output_dir, 'ech_blindness_proof.png')
plt.savefig(save_path)
print(f"Success: Visual saved to src/{save_path}")