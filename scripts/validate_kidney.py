from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "raw" / "kidney.csv"
OUTPUT_FILE = BASE_DIR / "app" / "models" / "kidney_cross_validation.csv"

print("Loading Kidney Disease dataset...")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

df = df.rename(columns={"wbcc": "wc", "rbcc": "rc"})
df = df.replace(r"^\s*\?\s*$", pd.NA, regex=True)

for column in df.select_dtypes(include=["object"]).columns:
    df[column] = df[column].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )

df = df.dropna(subset=["class"]).copy()

X = df.drop(columns=["class"]).copy()

y = (
    df["class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"ckd": 1, "notckd": 0})
)

valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

numeric_features = [
    "age", "bp", "bgr", "bu", "sc", "sod",
    "pot", "hemo", "pcv", "wc", "rc"
]

categorical_features = [
    "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet", "pe", "ane"
]

for column in numeric_features:
    X[column] = pd.to_numeric(X[column], errors="coerce")

for column in categorical_features:
    X[column] = X[column].astype("object")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

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

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

results = []

for name, estimator in models.items():
    print(f"\nValidating {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    row = {"Model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric.upper()} Mean"] = values.mean()
        row[f"{metric.upper()} Std"] = values.std()

    results.append(row)

results_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n5-FOLD CROSS-VALIDATION RESULTS")
print(results_df.round(4).to_string(index=False))

print("\nResults saved to:", OUTPUT_FILE)
print("\nCross-validation completed!")