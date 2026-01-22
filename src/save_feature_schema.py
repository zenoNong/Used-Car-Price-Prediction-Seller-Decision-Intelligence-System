import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/featured_used_cars.csv")
SCHEMA_PATH = Path("models/feature_schema.json")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["selling_price"])

SCHEMA_PATH.parent.mkdir(exist_ok=True)
X.columns.to_series().to_json(SCHEMA_PATH)

print("Feature schema saved.")
