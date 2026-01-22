import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/processed/cleaned_used_cars.csv")
OUTPUT_PATH = Path("data/processed/featured_used_cars.csv")


def load_data():
    return pd.read_csv(INPUT_PATH)

def add_usage_intensity(df):
    df["km_per_year"] = df["km_driven"] / (df["vehicle_age"] + 1)
    return df

def add_power_efficiency(df):
    df["power_per_cc"] = df["max_power"] / df["engine"]
    return df

def add_mileage_bucket(df):
    df["mileage_bucket"] = pd.cut(
        df["mileage"],
        bins=[0, 15, 20, 25, 40],
        labels=["low", "medium", "high", "very_high"]
    )
    return df

def add_brand_popularity(df):
    brand_counts = df["brand"].value_counts(normalize=True)
    df["brand_popularity"] = df["brand"].map(brand_counts)
    return df

def add_model_rarity(df):
    model_freq = df["model"].value_counts(normalize=True)
    df["model_rarity"] = df["model"].map(model_freq)
    return df

def encode_low_cardinality(df):
    low_card_cols = [
        "fuel_type",
        "transmission_type",
        "seller_type",
        "mileage_bucket"
    ]
    df = pd.get_dummies(df, columns=low_card_cols, drop_first=True)
    return df

def drop_text_columns(df):
    return df.drop(columns=["brand", "model"])

def final_validation(df):
    assert df.isnull().sum().sum() == 0, "Nulls introduced in feature engineering"
    return df

def save_data(df):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

def run_pipeline():
    df = load_data()

    df = add_usage_intensity(df)
    df = add_power_efficiency(df)
    df = add_mileage_bucket(df)

    df = add_brand_popularity(df)
    df = add_model_rarity(df)

    df = encode_low_cardinality(df)
    df = drop_text_columns(df)

    df = final_validation(df)
    save_data(df)


if __name__ == "__main__":
    run_pipeline()
