from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"

MODEL_DIR = BASE_DIR / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("\nLoading diabetes dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Validate dataset
# --------------------------------------------------

required_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("The dataset is empty.")

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Unexpected explicit missing values found in the raw dataset."
    )

if not set(df["Outcome"].unique()).issubset({0, 1}):
    raise ValueError("Outcome must contain only 0 and 1.")

if df["Outcome"].nunique() != 2:
    raise ValueError("Both outcome classes must be present.")

if not np.isfinite(
    df[required_columns].to_numpy(dtype=float)
).all():
    raise ValueError("Dataset contains non-finite numeric values.")

print("Dataset validation successful.")


# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["Outcome"])
y = df["Outcome"].astype(int)

print("\nClass distribution:")
print(y.value_counts().sort_index())


# --------------------------------------------------
# 5. Split data before fitting preprocessing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 6. Define candidate missing measurements
# --------------------------------------------------

zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Preserve the raw data.
# Convert only the selected zero measurements to NaN.
X_train = X_train.copy()
X_test = X_test.copy()

X_train[zero_as_missing] = X_train[zero_as_missing].replace(
    0, np.nan
)

X_test[zero_as_missing] = X_test[zero_as_missing].replace(
    0, np.nan
)


# --------------------------------------------------
# 7. Build preprocessing pipelines
# --------------------------------------------------

# Logistic Regression needs scaling.
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
            list(X.columns),
        )
    ]
)

# Random Forest does not require feature scaling.
forest_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            list(X.columns),
        )
    ]
)


# --------------------------------------------------
# 8. Define models
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
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


# --------------------------------------------------
# 9. Train and evaluate
# --------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ROC-AUC": roc_auc,
        }
    )

    # Save each complete pipeline, including preprocessing.
    filename = (
        "diabetes_logistic_regression.joblib"
        if model_name == "Logistic Regression"
        else "diabetes_random_forest.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "feature_names": list(X.columns),
            "model_name": model_name,
            "dataset": "Pima Indians Diabetes Database",
        },
        MODEL_DIR / filename,
    )

    print(f"\nSaved model: {MODEL_DIR / filename}")


# --------------------------------------------------
# 10. Compare models
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

results_path = MODEL_DIR / "diabetes_model_comparison.csv"

results_df.to_csv(results_path, index=False)

print("\nComparison saved to:", results_path)

print("\nTraining and evaluation completed.")
print(
    "These results are experimental and do not establish "
    "clinical validity."
)