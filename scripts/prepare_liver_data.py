from pathlib import Path

import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "liver.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "liver_cleaned.csv"


# Load raw dataset
df = pd.read_csv(RAW_FILE)

# Clean column names
df.columns = df.columns.str.strip()

# Remove duplicate records
df = df.drop_duplicates().copy()

# Convert numeric columns to numeric values
numeric_columns = [
    "Age", "TB", "DB", "Alkphos", "Sgpt",
    "Sgot", "TP", "ALB", "A/G Ratio"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Clean gender values
df["Gender"] = df["Gender"].astype("string").str.strip().str.title()

# Convert target to numeric
df["Selector"] = pd.to_numeric(df["Selector"], errors="coerce")

# Remove rows with missing target or gender
df = df.dropna(subset=["Selector", "Gender"]).copy()

# Keep only valid target labels
df = df[df["Selector"].isin([1, 2])].copy()

# Convert target to binary:
# 1 = Liver disease category
# 0 = No liver disease category
df["Selector"] = df["Selector"].map({1: 1, 2: 0}).astype(int)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Liver dataset preparation completed!")
print("Saved to:", OUTPUT_FILE)
print("Dataset shape:", df.shape)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTarget distribution (1 = liver disease category, 0 = no liver disease category):")
print(df["Selector"].value_counts())

print("\nFirst five rows:")
print(df.head())