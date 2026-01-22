import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path

from decision_engine import (
    determine_action,
    price_band,
    reviewer_notes,
    build_decision_output
)

DATA_PATH = Path("data/processed/featured_used_cars.csv")
MODEL_PATH = Path("models/xgboost_price_model.pkl")

def load_artifacts():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["selling_price"])
    y = df["selling_price"]

    model = joblib.load(MODEL_PATH)

    return X, y, model

def compute_confidence(y_true, y_pred):
    rel_error = abs(y_pred - y_true) / y_true
    confidence = 1 - rel_error
    return np.clip(confidence, 0, 1)

def setup_shap(model, X_background):
    explainer = shap.TreeExplainer(model)
    return explainer

def run_single_decision():
    X, y, model = load_artifacts()

    # pick a realistic sample
    idx = 42
    row = X.iloc[idx]
    actual_price = y.iloc[idx]

    predicted_price = model.predict(row.values.reshape(1, -1))[0]

    confidence = compute_confidence(
        actual_price,
        predicted_price
    )

    # SHAP explanation
    explainer = shap.TreeExplainer(model)

    row_df = X.iloc[[idx]].astype(float)
    shap_vals = explainer.shap_values(row_df)

    shap_vals = shap_vals[0]

    shap_df = pd.DataFrame({
        "feature": X.columns,
        "impact": shap_vals
    })

    top_features = (
        shap_df
        .assign(abs_impact=lambda x: x.impact.abs())
        .sort_values("abs_impact", ascending=False)
        .head(3)["feature"]
        .tolist()
    )


    decision = build_decision_output(
        row=row,
        predicted_price=predicted_price,
        confidence=confidence,
        top_shap_features=top_features
    )

    return decision, actual_price

def main():
    decision, actual_price = run_single_decision()

    print("\n=== FINAL DECISION OUTPUT ===")
    for k, v in decision.items():
        print(f"{k}: {v}")

    print(f"\nActual Price: {int(actual_price)}")

if __name__ == "__main__":
    main()
