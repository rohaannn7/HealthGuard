from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "liver_cleaned.csv"
OUTPUT_FILE = BASE_DIR / "app" / "models" / "liver_model_comparison.csv"

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Selector"])
y = df["Selector"]

numeric_features = [
    "Age", "TB", "DB", "Alkphos", "Sgpt",
    "Sgot", "TP", "ALB", "A/G Ratio"
]
categorical_features = ["Gender"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    ),
}

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = []

for model_name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier),
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        error_score="raise",
    )

    result = {
        "Model": model_name,
    }

    for metric in scoring:
        values = scores[f"test_{metric}"]
        result[f"{metric.capitalize()} Mean"] = values.mean()
        result[f"{metric.capitalize()} Std"] = values.std()

    results.append(result)

    print(f"\n{model_name}")
    for metric in scoring:
        values = scores[f"test_{metric}"]
        print(
            f"{metric.capitalize()}: "
            f"{values.mean():.4f} +/- {values.std():.4f}"
        )

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nValidation completed!")
print("Results saved to:", OUTPUT_FILE)
print("\nModel comparison:")
print(results_df.to_string(index=False))