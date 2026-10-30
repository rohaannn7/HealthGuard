from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# --------------------------------------------------
# 1. Paths and feature configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "app" / "models" / "kidney_random_forest.joblib"
DATA_PATH = BASE_DIR / "data" / "raw" / "kidney.csv"

NUMERIC_FEATURES = [
    "age", "bp", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc",
]

CATEGORICAL_FEATURES = [
    "sg", "al", "su", "rbc", "pc", "pcc",
    "ba", "htn", "dm", "cad", "appet", "pe", "ane",
]

KIDNEY_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_model = None
_explainer = None
_background = None
_category_maps = None


# --------------------------------------------------
# 2. Normalize categorical values
# --------------------------------------------------

def normalize_category(value):
    """Normalize categorical values consistently."""
    if pd.isna(value):
        return "__missing__"

    value = str(value).strip().lower()

    # Normalize numeric-looking categories such as 1.020 to 1.02.
    if value not in {"", "__missing__"}:
        try:
            number = float(value)
            value = format(number, ".15g")
        except (ValueError, TypeError):
            pass

    return value


# --------------------------------------------------
# 3. Load model and initialize SHAP
# --------------------------------------------------

def get_kidney_explainer():
    """Load the trained pipeline and prepare a reusable SHAP explainer."""

    global _model, _explainer, _background, _category_maps

    if _explainer is not None:
        return _model, _explainer, _background

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Kidney model not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Kidney dataset not found: {DATA_PATH}"
        )

    _model = joblib.load(MODEL_PATH)

    data = pd.read_csv(DATA_PATH)

    # Normalize dataset column names.
    data.columns = data.columns.str.strip().str.lower()

    # Match the column names used during model training.
    data = data.rename(
        columns={
            "wbcc": "wc",
            "rbcc": "rc",
        }
    )

    missing = [
        feature
        for feature in KIDNEY_FEATURES
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Kidney dataset is missing features: {missing}"
        )

    # Ensure numerical features are numeric.
    for feature in NUMERIC_FEATURES:
        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

    # Create mappings for categorical features.
    _category_maps = {}

    for feature in CATEGORICAL_FEATURES:
        categories = {
            normalize_category(value)
            for value in data[feature].dropna().unique()
        }

        # Preserve missing values as a separate category.
        if data[feature].isna().any():
            categories.add("__missing__")

        ordered_categories = sorted(categories)

        _category_maps[feature] = {
            category: index
            for index, category in enumerate(ordered_categories)
        }

    # Select representative reference rows.
    _background = data[KIDNEY_FEATURES].sample(
        n=min(20, len(data)),
        random_state=42,
    ).copy()

    # Encode categorical columns for SHAP calculations.
    for feature in CATEGORICAL_FEATURES:
        mapping = _category_maps[feature]

        _background[feature] = (
            _background[feature]
            .map(normalize_category)
            .map(mapping)
            .astype(float)
        )

    def predict_positive_class(rows):
        """
        Convert SHAP's numeric arrays back to model-compatible inputs
        and return the probability of the CKD class.
        """

        frame = pd.DataFrame(
            rows,
            columns=KIDNEY_FEATURES,
        )

        for feature in NUMERIC_FEATURES:
            frame[feature] = pd.to_numeric(
                frame[feature],
                errors="coerce",
            )

        for feature in CATEGORICAL_FEATURES:
            mapping = _category_maps[feature]

            reverse_mapping = {
                float(index): category
                for category, index in mapping.items()
            }

            # SHAP perturbs numeric representations. Map each
            # perturbed value to its nearest valid category.
            def decode_category(value):
                if pd.isna(value):
                    return np.nan

                nearest_index = int(round(float(value)))

                if nearest_index not in range(len(reverse_mapping)):
                    nearest_index = min(
                        max(nearest_index, 0),
                        len(reverse_mapping) - 1,
                    )

                category = reverse_mapping[float(nearest_index)]

                if category == "__missing__":
                    return np.nan

                return category

            frame[feature] = frame[feature].map(
                decode_category
            )

        # Restore the exact feature order used by the trained pipeline.
        frame = frame[KIDNEY_FEATURES]

        probabilities = _model.predict_proba(frame)

        classes = list(_model.classes_)

        positive_index = next(
            (
                index
                for index, label in enumerate(classes)
                if str(label).strip().lower()
                in {"ckd", "1", "true"}
            ),
            None,
        )

        if positive_index is None:
            raise ValueError(
                f"Could not identify the CKD class in {classes}"
            )

        return probabilities[:, positive_index]

    _explainer = shap.Explainer(
        predict_positive_class,
        _background,
        algorithm="permutation",
    )

    return _model, _explainer, _background


# --------------------------------------------------
# 4. Explain an individual kidney assessment
# --------------------------------------------------

def explain_kidney(input_values):
    """
    Return the model's CKD probability and per-feature SHAP
    contributions for a single assessment.
    """

    model, explainer, _ = get_kidney_explainer()

    # Build the original input for the trained model.
    sample = pd.DataFrame(
        [[input_values[name] for name in KIDNEY_FEATURES]],
        columns=KIDNEY_FEATURES,
    )

    for feature in NUMERIC_FEATURES:
        sample[feature] = pd.to_numeric(
            sample[feature],
            errors="coerce",
        )

    for feature in CATEGORICAL_FEATURES:
        sample[feature] = sample[feature].map(
            lambda value: (
                np.nan
                if pd.isna(value) or str(value).strip() == ""
                else str(value).strip().lower()
            )
        )

    # Determine the CKD class index.
    classes = list(model.classes_)

    positive_index = next(
        (
            index
            for index, label in enumerate(classes)
            if str(label).strip().lower()
            in {"ckd", "1", "true"}
        ),
        None,
    )

    if positive_index is None:
        raise ValueError(
            f"Could not identify the CKD class in {classes}"
        )

    probability = float(
        model.predict_proba(sample)[0, positive_index]
    )

    # Encode categorical values exactly as the explainer expects.
    encoded_sample = sample.copy()

    for feature in CATEGORICAL_FEATURES:
        mapping = _category_maps[feature]

        def encode_category(value):
            if pd.isna(value):
                value = "__missing__"
            else:
                value = normalize_category(value)

            if value not in mapping:
                raise ValueError(
                    f"Unknown category {value!r} for {feature}. "
                    f"Valid categories: {list(mapping.keys())}"
                )

            return float(mapping[value])

        encoded_sample[feature] = (
            encoded_sample[feature].map(encode_category)
        )

    # Calculate SHAP contributions.
    explanation = explainer(
        encoded_sample[KIDNEY_FEATURES],
        max_evals=2 * len(KIDNEY_FEATURES) + 1,
    )

    contributions = np.asarray(
        explanation.values
    ).reshape(-1)

    results = []

    for feature, contribution in zip(
        KIDNEY_FEATURES,
        contributions,
    ):
        value = input_values[feature]

        results.append({
            "feature": feature,
            "value": str(value),
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