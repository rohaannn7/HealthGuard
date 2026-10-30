from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "liver.csv"


# Download the Indian Liver Patient Dataset from UCI
print("Downloading the Indian Liver Patient Dataset...")

dataset = fetch_ucirepo(id=225)

# Extract input features and target
X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# Combine features and target into one DataFrame
df = pd.concat(
    [X.reset_index(drop=True), y.reset_index(drop=True)],
    axis=1,
)

# Save the raw dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset downloaded successfully!")
print("Saved to:", OUTPUT_FILE)
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nTarget distribution:")
print(df.iloc[:, -1].value_counts(dropna=False))