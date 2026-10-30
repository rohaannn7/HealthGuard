from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "liver_cleaned.csv"
MODEL_DIR = BASE_DIR / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Separate features and target
X = df.drop(columns=["Selector"])
y = df["Selector"]

# Feature types
numeric_features = [
    "Age", "TB", "DB", "Alkphos", "Sgpt",
    "Sgot", "TP", "ALB", "A/G Ratio"
]
categorical_features = ["Gender"]

# Numeric preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# Models to train
models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    ),
}

# Train and save each complete pipeline
for model_name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X, y)

    output_file = MODEL_DIR / f"liver_{model_name}.joblib"
    joblib.dump(pipeline, output_file)

    print(f"Saved {model_name} model to: {output_file}")

print("\nLiver disease model training completed!")
print("Features used:", X.columns.tolist())
print("Training records:", len(X))
print("Target distribution:")
print(y.value_counts().sort_index())