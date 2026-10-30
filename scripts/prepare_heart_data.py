from pathlib import Path
import pandas as pd

# Project paths
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "raw" / "heart.csv"
output_file = project_root / "data" / "processed" / "heart_cleaned.csv"

# Column names for the UCI Cleveland Heart Disease dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num"
]

# Check that the downloaded file exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Dataset not found at {input_file}. "
        "Check that heart.csv is inside data/raw."
    )

# Load data: the original file has no headers
df = pd.read_csv(
    input_file,
    header=None,
    names=columns,
    na_values="?"
)

# Convert columns to numeric values
df = df.apply(pd.to_numeric, errors="coerce")

print("Original dataset shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())

# Convert target to binary classification
df["target"] = (df["num"] > 0).astype(int)

# Remove the original target column
df = df.drop(columns=["num"])

# Create processed-data folder
output_file.parent.mkdir(parents=True, exist_ok=True)

# Save prepared dataset
df.to_csv(output_file, index=False)

print("\nPrepared dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["target"].value_counts())

print("\nSaved prepared dataset to:", output_file)
print("\nFirst five rows:")
print(df.head())