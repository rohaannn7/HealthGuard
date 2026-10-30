
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# --------------------------------------------------
# 1. Paths and feature configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "app" / "models" / "liver_random_forest.joblib"
DATA_PATH = BASE_DIR / "data" / "raw" / "liver.csv"

LIVER_FEATURES = [
    "Age",
    "Gender",
    "TB",
    "DB",
    "Alkphos",
    "Sgpt",
    "Sgot",
    "TP",
    "ALB",
    "A/G Ratio",
]

NUMERIC_FEATURES = [
    "Age",
    "TB",
    "DB",
    "Alkphos",
    "Sgpt",
    "Sgot",
    "TP",
    "ALB",
    "A/G Ratio",
]

_model = None
_explainer = None
_background = None


# --------------------------------------------------
# 2. Normalize input values
# --------------------------------------------------

def normalize_gender(value):
    """Normalize gender values without changing the model's input format."""

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in {"male", "m"}:
        return "Male"

    if value in {"female", "f"}:
        return "Female"

    return str(value)


def prepare_liver_data(data):
    """Prepare input data using the expected feature order and types."""

    frame = pd.DataFrame(data, columns=LIVER_FEATURES).copy()

    for feature in NUMERIC_FEATURES:
        frame[feature] = pd.to_numeric(
            frame[feature],
            errors="coerce",
        )

    frame["Gender"] = frame["Gender"].map(normalize_gender)

    return frame


# --------------------------------------------------
# 3. Load model and initialize SHAP
# --------------------------------------------------

def get_liver_explainer():
    """Load the model and create a reusable SHAP explainer."""

    global _model, _explainer, _background

    if _explainer is not None:
        return _model, _explainer, _background

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Liver model not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Liver dataset not found: {DATA_PATH}"
        )

    _model = joblib.load(MODEL_PATH)

    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    missing = [
        feature
        for feature in LIVER_FEATURES
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Liver dataset is missing features: {missing}"
        )

    # Prepare the original, model-compatible background.
    background = prepare_liver_data(
        data[LIVER_FEATURES]
    )

    # SHAP must perform its masking and distance calculations
    # on numeric values. Encode Gender for SHAP only.
    gender_mapping = {
        "Male": 0.0,
        "Female": 1.0,
    }

    background_encoded = background.copy()

    background_encoded["Gender"] = (
        background_encoded["Gender"]
        .map(gender_mapping)
    )

    # Use complete rows for a stable, numeric background sample.
    background_encoded = background_encoded.dropna()

    if background_encoded.empty:
        raise ValueError(
            "No complete background rows are available "
            "for liver SHAP."
        )

    _background = background_encoded.sample(
        n=min(20, len(background_encoded)),
        random_state=42,
    ).copy()

    def predict_disease_probability(rows):
        """
        Convert SHAP's numeric rows back into model-compatible
        input and return the liver-disease probability.
        """

        frame = pd.DataFrame(
            rows,
            columns=LIVER_FEATURES,
        )

        for feature in NUMERIC_FEATURES:
            frame[feature] = pd.to_numeric(
                frame[feature],
                errors="coerce",
            )

        # Decode the numeric Gender representation.
        frame["Gender"] = frame["Gender"].map(
            lambda value: (
                "Female"
                if float(value) >= 0.5
                else "Male"
            )
        )

        probabilities = _model.predict_proba(frame)
        classes = list(_model.classes_)

        # The trained model uses encoded classes [0, 1].
        # In this project, 1 means liver disease.
        positive_index = next(
            (
                index
                for index, label in enumerate(classes)
                if str(label).strip().lower()
                in {"1", "true", "disease", "liver disease"}
            ),
            None,
        )

        if positive_index is None:
            raise ValueError(
                "Could not identify the liver-disease class. "
                f"Model classes: {classes}"
            )

        return probabilities[:, positive_index]

    _explainer = shap.Explainer(
        predict_disease_probability,
        _background,
        algorithm="permutation",
    )

    return _model, _explainer, _background


# --------------------------------------------------
# 4. Explain an individual liver assessment
# --------------------------------------------------

def explain_liver(input_values):
    """
    Return the liver-disease model probability and SHAP
    contributions for an individual assessment.
    """

    model, explainer, _ = get_liver_explainer()

    # Prepare original values for the trained model.
    sample = prepare_liver_data(
        [[input_values[name] for name in LIVER_FEATURES]]
    )

    # Determine the disease-positive class.
    classes = list(model.classes_)

    positive_index = next(
        (
            index
            for index, label in enumerate(classes)
            if str(label).strip().lower()
            in {"1", "true", "disease", "liver disease"}
        ),
        None,
    )

    if positive_index is None:
        raise ValueError(
            "Could not identify the liver-disease class. "
            f"Model classes: {classes}"
        )

    # Calculate the model's prediction probability.
    probability = float(
        model.predict_proba(sample)[0, positive_index]
    )

    # Encode Gender numerically for SHAP calculations.
    encoded_sample = sample.copy()

    encoded_sample["Gender"] = (
        encoded_sample["Gender"]
        .map({
            "Male": 0.0,
            "Female": 1.0,
        })
    )

    if encoded_sample.isna().any().any():
        raise ValueError(
            "Invalid or missing input values. "
            "Check the liver assessment form, especially Gender "
            "and the numerical measurements."
        )

    # Compute SHAP contributions.
    explanation = explainer(
        encoded_sample[LIVER_FEATURES],
        max_evals=2 * len(LIVER_FEATURES) + 1,
    )

    contributions = np.asarray(
        explanation.values
    ).reshape(-1)

    results = []

    for feature, contribution in zip(
        LIVER_FEATURES,
        contributions,
    ):
        results.append({
            "feature": feature,
            "value": str(input_values[feature]),
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

    # Show the features with the largest absolute contributions first.
    results.sort(
        key=lambda item: item["absolute_contribution"],
        reverse=True,
    )

    return {
        "probability": probability,
        "features": results,
        "explainer_method": "SHAP permutation explainer",
    }
