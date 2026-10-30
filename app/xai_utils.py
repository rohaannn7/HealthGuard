from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "app" / "models"
DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"

DIABETES_MODEL_PATH = MODEL_DIR / "diabetes_random_forest.joblib"

DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

_diabetes_model = None
_diabetes_explainer = None


def get_diabetes_explainer():
    """Load the model and prepare a reusable SHAP explainer."""
    global _diabetes_model, _diabetes_explainer

    if _diabetes_explainer is not None:
        return _diabetes_model, _diabetes_explainer

    bundle = joblib.load(DIABETES_MODEL_PATH)

    _diabetes_model = (
        bundle["model"] if isinstance(bundle, dict) else bundle
    )

    features = DIABETES_FEATURES

    if isinstance(bundle, dict) and "feature_names" in bundle:
        features = list(bundle["feature_names"])

    if features != DIABETES_FEATURES:
        raise ValueError(
            "The model's feature order differs from the configured "
            "diabetes feature order. Update DIABETES_FEATURES first."
        )

    data = pd.read_csv(DATA_PATH)

    missing = [name for name in features if name not in data.columns]
    if missing:
        raise ValueError(f"Diabetes dataset is missing: {missing}")

    background = data[features].sample(
        n=min(50, len(data)),
        random_state=42,
    )

    def predict_positive_class(rows):
        frame = pd.DataFrame(rows, columns=features)
        return _diabetes_model.predict_proba(frame)[:, 1]

    _diabetes_explainer = shap.Explainer(
        predict_positive_class,
        background,
        algorithm="permutation",
    )

    return _diabetes_model, _diabetes_explainer


def explain_diabetes(input_values):
    """Return a probability and SHAP contributions for one assessment."""
    model, explainer = get_diabetes_explainer()

    sample = pd.DataFrame([input_values], columns=DIABETES_FEATURES)

    probability = float(model.predict_proba(sample)[0, 1])

    explanation = explainer(
        sample,
        max_evals=2 * len(DIABETES_FEATURES) + 1,
    )

    contributions = np.asarray(explanation.values).reshape(-1)

    results = []

    for feature, contribution in zip(
        DIABETES_FEATURES,
        contributions,
    ):
        value = input_values[feature]

        results.append({
            "feature": feature,
            "value": float(value),
            "contribution": float(contribution),
            "absolute_contribution": float(abs(contribution)),
            "direction": (
                "increases model output"
                if contribution > 0
                else "decreases model output"
                if contribution < 0
                else "no directional contribution"
            ),
        })

    results.sort(
        key=lambda item: item["absolute_contribution"],
        reverse=True,
    )

    return {
        "probability": probability,
        "features": results,
        "explainer_method": "SHAP permutation explainer",
    }

# ---------------- HEART DISEASE SHAP EXPLANATION ----------------

HEART_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "heart_cleaned.csv"
)
HEART_MODEL_PATH = MODEL_DIR / "heart_random_forest.joblib"

HEART_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

_heart_model = None
_heart_explainer = None


def get_heart_explainer():
    """Load the heart model and create a reusable SHAP explainer."""
    global _heart_model, _heart_explainer

    if _heart_explainer is not None:
        return _heart_model, _heart_explainer

    _heart_model = joblib.load(HEART_MODEL_PATH)

    data = pd.read_csv(HEART_DATA_PATH)

    missing = [
        feature for feature in HEART_FEATURES
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Heart dataset is missing expected features: {missing}"
        )

    background = data[HEART_FEATURES].sample(
        n=min(50, len(data)),
        random_state=42,
    )

    def predict_positive_class(rows):
        frame = pd.DataFrame(rows, columns=HEART_FEATURES)
        return _heart_model.predict_proba(frame)[:, 1]

    _heart_explainer = shap.Explainer(
        predict_positive_class,
        background,
        algorithm="permutation",
    )

    return _heart_model, _heart_explainer


def explain_heart(input_values):
    """Return the positive-class probability and feature contributions."""
    model, explainer = get_heart_explainer()

    sample = pd.DataFrame(
        [[input_values[name] for name in HEART_FEATURES]],
        columns=HEART_FEATURES,
    )

    probability = float(model.predict_proba(sample)[0, 1])

    explanation = explainer(
        sample,
        max_evals=2 * len(HEART_FEATURES) + 1,
    )

    contributions = np.asarray(explanation.values).reshape(-1)

    results = []

    for feature, contribution in zip(
        HEART_FEATURES,
        contributions,
    ):
        results.append({
            "feature": feature,
            "value": float(input_values[feature]),
            "contribution": float(contribution),
            "absolute_contribution": float(abs(contribution)),
            "direction": (
                "increases model output"
                if contribution > 0
                else "decreases model output"
                if contribution < 0
                else "no directional contribution"
            ),
        })

    results.sort(
        key=lambda item: item["absolute_contribution"],
        reverse=True,
    )

    return {
        "probability": probability,
        "features": results,
        "explainer_method": "SHAP permutation explainer",
    }