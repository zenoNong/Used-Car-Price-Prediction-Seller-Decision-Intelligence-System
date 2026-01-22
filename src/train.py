import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib

DATA_PATH = Path("data/processed/featured_used_cars.csv")
MODEL_DIR = Path("models")

def load_and_split():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["selling_price"])
    y = df["selling_price"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val

def evaluate_model(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape

def train_linear_regression(X_train, y_train, X_val, y_val):
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse, mape = evaluate_model(y_val, preds)

    print(f"Linear Regression → RMSE: {rmse:.2f}, MAPE: {mape:.4f}")
    return model

def train_random_forest(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse, mape = evaluate_model(y_val, preds)
    print(f"Random Forest → RMSE: {rmse:.2f}, MAPE: {mape:.4f}")

    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse, mape = evaluate_model(y_val, preds)
    print(f"XGBoost → RMSE: {rmse:.2f}, MAPE: {mape:.4f}")

    return model

def save_model(model, name):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{name}.pkl")

def main():
    X_train, X_val, y_train, y_val = load_and_split()

    lr = train_linear_regression(X_train, y_train, X_val, y_val)
    rf = train_random_forest(X_train, y_train, X_val, y_val)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    save_model(xgb_model, "xgboost_price_model")


if __name__ == "__main__":
    main()
