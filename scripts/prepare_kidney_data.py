from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "kidney.csv"

df = pd.read_csv(INPUT_FILE)

print("KIDNEY DISEASE DATASET")
print("=" * 40)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df["class"].value_counts(dropna=False))

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nFirst five rows:")
print(df.head())
