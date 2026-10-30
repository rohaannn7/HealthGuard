from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "app" / "models"
DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"

MODEL_PATH = MODEL_DIR / "diabetes_random_forest.joblib"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"] if isinstance(bundle, dict) else bundle

    if isinstance(bundle, dict) and "feature_names" in bundle:
        FEATURES = list(bundle["feature_names"])

    data = pd.read_csv(DATA_PATH)

    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing features: {missing}. "
            f"Available columns: {data.columns.tolist()}"
        )

    background = data[FEATURES].sample(
        n=min(50, len(data)),
        random_state=42,
    ).copy()

    # Explain the probability of class 1 while preserving the pipeline.
    def predict_positive_class(rows):
        frame = pd.DataFrame(rows, columns=FEATURES)
        return model.predict_proba(frame)[:, 1]

    explainer = shap.Explainer(
        predict_positive_class,
        background,
        algorithm="permutation",
    )

    # Explain one representative example from the dataset.
    sample = background.iloc[[0]]
    explanation = explainer(sample, max_evals=2 * len(FEATURES) + 1)

    print("\nSHAP explanation generated successfully.")
    print("Model type:", type(model).__name__)
    print("Explained input:")
    print(sample.to_string(index=False))

    print("\nPredicted probability for class 1:")
    print(f"{predict_positive_class(sample.to_numpy())[0]:.4f}")

    print("\nFeature contributions:")
    contributions = explanation.values[0]

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP Contribution": contributions,
        "Absolute Contribution": np.abs(contributions),
    })

    importance = importance.sort_values(
        "Absolute Contribution",
        ascending=False,
    )

    print(importance[["Feature", "SHAP Contribution"]].to_string(index=False))

    output_path = BASE_DIR / "data" / "processed" / "diabetes_shap_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)

    print(f"\nFeature contributions saved to: {output_path}")


if __name__ == "__main__":
    main()