from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "raw" / "kidney.csv"
MODEL_DIR = BASE_DIR / "app" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and clean dataset
# --------------------------------------------------

print("Loading Kidney Disease dataset...")

df = pd.read_csv(DATA_FILE)

# Normalize column names.
df.columns = df.columns.str.strip().str.lower()

# Normalize alternate column names used by this dataset.
df = df.rename(
    columns={
        "wbcc": "wc",
        "rbcc": "rc",
    }
)

# Convert whitespace and '?' markers to missing values.
df = df.replace(r"^\s*\?\s*$", pd.NA, regex=True)

# Remove extra spaces from text values.
for column in df.select_dtypes(include=["object", "str"]).columns:
    df[column] = df[column].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )

# Remove rows without a target label.
df = df.dropna(subset=["class"]).copy()

# Separate features and target.
X = df.drop(columns=["class"]).copy()

y = (
    df["class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"ckd": 1, "notckd": 0})
)

# Remove rows with unrecognized target labels.
valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)


# --------------------------------------------------
# 3. Identify feature types
# --------------------------------------------------

numeric_features = [
    "age",
    "bp",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
]

categorical_features = [
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]

# Verify that the expected columns exist.
expected_features = numeric_features + categorical_features
missing_columns = sorted(set(expected_features) - set(X.columns))

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}\n"
        f"Available columns: {X.columns.tolist()}"
    )

# Convert numerical columns to numbers.
# Unreadable values become missing and are imputed below.
for column in numeric_features:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Treat nominal features as categorical, including numeric-looking
# categories such as specific gravity, albumin and sugar.
for column in categorical_features:
    X[column] = X[column].astype("object")

print("Dataset shape:", df.shape)
print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))
print("Total input features:", len(expected_features))

print("\nTarget distribution:")
print(y.value_counts().rename(index={1: "CKD", 0: "Not CKD"}))


# --------------------------------------------------
# 4. Preprocessing pipelines
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# --------------------------------------------------
# 5. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 6. Define models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ),
}


# --------------------------------------------------
# 7. Train, evaluate and save models
# --------------------------------------------------

results = []

for name, estimator in models.items():
    print(f"\nTraining {name}...")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(metrics)

    if name == "Logistic Regression":
        filename = "kidney_logistic_regression.joblib"
    else:
        filename = "kidney_random_forest.joblib"

    model_path = MODEL_DIR / filename
    joblib.dump(model, model_path)

    print("Saved model:", model_path)


# --------------------------------------------------
# 8. Save model comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

comparison_file = MODEL_DIR / "kidney_model_comparison.csv"
results_df.to_csv(comparison_file, index=False)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

print("\nComparison saved to:", comparison_file)
print("\nKidney Disease model training completed!")
