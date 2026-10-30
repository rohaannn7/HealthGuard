from pathlib import Path
from ucimlrepo import fetch_ucirepo

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading Chronic Kidney Disease dataset...")

# Fetch the official UCI dataset
kidney_data = fetch_ucirepo(id=336)

# Combine input features and target
X = kidney_data.data.features
y = kidney_data.data.targets

import pandas as pd

df = pd.concat([X, y], axis=1)

# Save as CSV
output_file = RAW_DIR / "kidney.csv"
df.to_csv(output_file, index=False)

print("\nDownload completed successfully!")
print("Saved to:", output_file)
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())