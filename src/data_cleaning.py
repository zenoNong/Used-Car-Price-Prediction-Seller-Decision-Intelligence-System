import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/used_cars.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_used_cars.csv")


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df

def drop_irrelevant_columns(df):
    columns_to_drop = ["Unnamed: 0", "car_name"]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    return df

def validate_target(df):
    assert "selling_price" in df.columns, "Target column missing"

    # Remove rows with non-positive prices (defensive programming)
    df = df[df["selling_price"] > 0]

    return df

def cap_price_outliers(df, lower_quantile=0.01, upper_quantile=0.99):
    lower = df["selling_price"].quantile(lower_quantile)
    upper = df["selling_price"].quantile(upper_quantile)

    df["selling_price"] = df["selling_price"].clip(lower, upper)
    return df

def normalize_categoricals(df):
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        df[col] = (
            df[col]
            .str.lower()
            .str.strip()
        )
    return df

def validate_numeric_ranges(df):
    # Vehicle age should be reasonable
    df = df[df["vehicle_age"].between(0, 30)]

    # Mileage sanity
    df = df[df["mileage"].between(5, 40)]

    # Engine CC sanity
    df = df[df["engine"].between(600, 5000)]

    return df

def final_validation(df):
    # No missing values allowed
    assert df.isnull().sum().sum() == 0, "Missing values detected"

    # No duplicate rows
    df = df.drop_duplicates()

    return df

def save_data(df):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

def run_pipeline():
    df = load_data()
    df = drop_irrelevant_columns(df)
    df = validate_target(df)
    df = cap_price_outliers(df)
    df = normalize_categoricals(df)
    df = validate_numeric_ranges(df)
    df = final_validation(df)
    save_data(df)


if __name__ == "__main__":
    run_pipeline()
