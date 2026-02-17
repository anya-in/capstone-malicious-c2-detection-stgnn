import pandas as pd
from ydata_profiling import ProfileReport
import os

# 1. Ensure the output directory exists
os.makedirs('reports', exist_ok=True)

# 2. Load your verified dataset
# Make sure the path matches where you put the file!
data_path = 'data/raw/ml_features_and_labels.csv'
df = pd.read_csv(data_path)

print(f"Dataset loaded: {df.shape[0]} rows found. Generating report...")

# 3. Generate the Extensive Report (Minimal Mode for speed/stability)
profile = ProfileReport(
    df, 
    title="Team 11: CIC-PQC-OAV (2025) Extensive EDA Report",
    minimal=True 
)

# 4. Save to HTML
profile.to_file("reports/Team11_Full_EDA_Report.html")
print("Success! Report saved in the 'reports' folder.")