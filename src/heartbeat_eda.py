import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Path Setup
data_path = os.path.join('..', 'data', 'raw', 'ml_features_and_labels.csv')
output_dir = 'visualizations'

# 2. Automatically create the visualizations folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Load data
df = pd.read_csv(data_path)

# 4. Filter for Malicious only (assuming label 1 is malicious)
malicious_df = df[df['label'] == 1]

# 5. Create Timing Profile
plt.figure(figsize=(10, 6))
# Using e6b for flow duration as identified in your report
sns.histplot(malicious_df['e6b_flow_duration_ms'], bins=50, color='red', log_scale=True)
plt.title('C2 Signature: Malicious Flow Duration (e6b) Heartbeats')
plt.xlabel('Duration (ms) - Log Scale')

# 6. Save to src/visualizations/
save_path = os.path.join(output_dir, 'malicious_heartbeat_profile.png')
plt.savefig(save_path)
print(f"Success: Visual saved to src/{save_path}")