import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

DATA_PATH = Path("data/processed/featured_used_cars.csv")
MODEL_PATH = Path("models/xgboost_price_model.pkl")

def load_data_and_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["selling_price"])
    y = df["selling_price"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    return X_val, y_val, model

def overall_performance(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"Overall Validation MAPE: {mape:.4f}")

def error_analysis(y_true, y_pred):
    errors = (y_pred - y_true) / y_true
    return errors

import matplotlib.pyplot as plt

def plot_error_distribution(errors):
    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=50)
    plt.title("Prediction Error Distribution (Relative)")
    plt.xlabel("Relative Error")
    plt.ylabel("Count")
    plt.show()

def price_bucket_analysis(y_true, y_pred):
    df_eval = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred
    })

    df_eval["price_bucket"] = pd.cut(
        df_eval["actual"],
        bins=[0, 300000, 600000, 1000000, np.inf],
        labels=["low", "mid", "high", "premium"]
    )

    bucket_mape = (
        df_eval
        .groupby("price_bucket")
        .apply(lambda x: mean_absolute_percentage_error(x["actual"], x["predicted"]))
    )

    print("\nMAPE by Price Bucket:")
    print(bucket_mape)

def flag_high_risk_cases(X_val, y_true, y_pred, threshold=0.25):
    df_risk = X_val.copy()
    df_risk["actual_price"] = y_true
    df_risk["predicted_price"] = y_pred
    df_risk["relative_error"] = abs(y_pred - y_true) / y_true

    risky = df_risk[df_risk["relative_error"] > threshold]
    print(f"High-risk predictions: {len(risky)} / {len(df_risk)}")

    return risky

def compute_confidence(y_true, y_pred):
    relative_error = abs(y_pred - y_true) / y_true
    confidence = 1 - relative_error
    confidence = confidence.clip(0, 1)
    return confidence

def main():
    X_val, y_val, model = load_data_and_model()

    preds = model.predict(X_val)

    overall_performance(y_val, preds)

    errors = error_analysis(y_val, preds)
    plot_error_distribution(errors)

    price_bucket_analysis(y_val, preds)

    risky_cases = flag_high_risk_cases(X_val, y_val, preds)

    confidence = compute_confidence(y_val, preds)
    print(f"Average confidence: {confidence.mean():.3f}")

if __name__ == "__main__":
    main()
