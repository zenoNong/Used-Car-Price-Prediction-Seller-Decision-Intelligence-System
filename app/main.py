from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
import json

with open("models/feature_schema.json", "r") as f:
    FEATURE_COLUMNS = list(pd.read_json(f, typ="series").values)

from src.decision_engine import (
    determine_action,
    price_band,
    reviewer_notes
)

app = FastAPI(title="Used Car Pricing Decision API")

MODEL_PATH = "models/xgboost_price_model.pkl"
DATA_PATH = "data/processed/featured_used_cars.csv"

model = joblib.load(MODEL_PATH)

# background data for SHAP
background = pd.read_csv(DATA_PATH).drop(columns=["selling_price"]).sample(500, random_state=42)
explainer = shap.TreeExplainer(model)

class CarFeatures(BaseModel):
    # Core numeric features
    vehicle_age: int
    km_driven: int
    mileage: float
    engine: int
    max_power: float
    seats: int
    km_per_year: float
    power_per_cc: float
    brand_popularity: float
    model_rarity: float

    # Fuel type (one-hot)
    fuel_type_petrol: int = 0
    fuel_type_diesel: int = 0
    fuel_type_lpg: int = 0
    fuel_type_electric: int = 0

    # Transmission
    transmission_type_manual: int = 0

    # Seller type
    seller_type_individual: int = 0
    seller_type_trustmark_dealer: int = 0

    # Mileage buckets
    mileage_bucket_medium: int = 0
    mileage_bucket_high: int = 0
    mileage_bucket_very_high: int = 0


def validate_input(car: CarFeatures):
    errors = []

    if car.vehicle_age < 0 or car.vehicle_age > 30:
        errors.append("Invalid vehicle_age")

    if car.engine <= 0:
        errors.append("Engine capacity must be > 0")

    if car.max_power <= 0:
        errors.append("max_power must be > 0")

    if car.seats <= 0:
        errors.append("seats must be > 0")

    if car.km_driven < 0:
        errors.append("km_driven cannot be negative")

    if errors:
        return False, errors

    return True, None



@app.post("/predict_price")
def predict_price(car: CarFeatures):
    is_valid, errors = validate_input(car)
    if not is_valid:
        return {
            "error": "Invalid input",
            "details": errors
        }
    
    row = pd.DataFrame(
        0.0,
        index=[0],
        columns=FEATURE_COLUMNS,
        dtype=float
    )


    # fill provided values
    for key, value in car.dict().items():
        if key in row.columns:
            row.at[0, key] = value

    row = row.astype(float)


    predicted_price = model.predict(row)[0]

    # confidence (proxy since true price unknown)
    confidence = 0.90 if predicted_price < 1_000_000 else 0.82

    shap_vals = explainer.shap_values(row)[0]

    shap_df = pd.DataFrame({
        "feature": row.columns,
        "impact": shap_vals
    })

    top_features = (
        shap_df
        .assign(abs_impact=lambda x: x.impact.abs())
        .sort_values("abs_impact", ascending=False)
        .head(3)["feature"]
        .tolist()
    )

    decision = determine_action(confidence, predicted_price)
    band = price_band(predicted_price, confidence)
    notes = reviewer_notes(row.iloc[0])

    return {
        "predicted_price": int(predicted_price),
        "confidence_score": round(confidence, 3),
        "decision": decision,
        "price_band": band,
        "key_price_drivers": top_features,
        "reviewer_notes": notes
    }
