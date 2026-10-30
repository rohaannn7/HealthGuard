from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"

RESULTS_DIR = BASE_DIR / "app" / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate dataset
# --------------------------------------------------

print("\nLoading diabetes dataset...")

df = pd.read_csv(DATA_PATH)

if "Outcome" not in df.columns:
    raise ValueError("The dataset must contain an Outcome column.")

X = df.drop(columns=["Outcome"]).copy()
y = df["Outcome"].astype(int)

if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
    raise ValueError("Outcome must contain both classes: 0 and 1.")

print("Dataset shape:", df.shape)
print("Records:", len(df))


# --------------------------------------------------
# 3. Define candidate missing measurements
# --------------------------------------------------

zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Replace selected zero measurements with missing values.
# This happens before cross-validation, but no statistics
# are calculated here. Imputation itself remains inside
# each pipeline and is fitted separately within each fold.

X[zero_as_missing] = X[zero_as_missing].replace(
    0, np.nan
)


# --------------------------------------------------
# 4. Define preprocessing pipelines
# --------------------------------------------------

feature_names = list(X.columns)

logistic_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            feature_names,
        )
    ]
)

forest_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            feature_names,
        )
    ]
)


# --------------------------------------------------
# 5. Define candidate models
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessing", logistic_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessing", forest_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]
    ),
}


# --------------------------------------------------
# 6. Configure stratified cross-validation
# --------------------------------------------------

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


# --------------------------------------------------
# 7. Evaluate models
# --------------------------------------------------

all_results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Cross-validating:", model_name)
    print("=" * 60)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
        error_score="raise",
    )

    result = {
        "Model": model_name,
    }

    for metric in scoring:
        values = scores[f"test_{metric}"]

        mean_score = values.mean()
        std_score = values.std(ddof=1)

        result[f"{metric}_mean"] = mean_score
        result[f"{metric}_std"] = std_score

        print(
            f"{metric.upper():10s}: "
            f"{mean_score:.4f} "
            f"+/- {std_score:.4f}"
        )

    all_results.append(result)


# --------------------------------------------------
# 8. Save cross-validation results
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

output_path = (
    RESULTS_DIR / "diabetes_cross_validation.csv"
)

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

print("\nResults saved to:")
print(output_path)

print("\nCross-validation completed.")