import pandas as pd
import shap
import joblib
from pathlib import Path

DATA_PATH = Path("data/processed/featured_used_cars.csv")
MODEL_PATH = Path("models/xgboost_price_model.pkl")

def load_data_and_model(sample_size=1000):
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["selling_price"])
    y = df["selling_price"]

    # SHAP is expensive → sample safely
    X_sample = X.sample(sample_size, random_state=42)

    model = joblib.load(MODEL_PATH)
    return X, X_sample, y, model

def create_explainer(model):
    explainer = shap.TreeExplainer(model)
    return explainer

def compute_shap_values(explainer, X_sample):
    shap_values = explainer.shap_values(X_sample)
    return shap_values

def global_feature_importance(shap_values, X_sample):
    shap.summary_plot(shap_values, X_sample, show=False)

def explain_single_prediction(explainer, model, X):
    instance = X.sample(1, random_state=7)
    shap_values = explainer.shap_values(instance)

    shap.force_plot(
        explainer.expected_value,
        shap_values,
        instance,
        matplotlib=True
    )
def price_justification(shap_values, feature_names, top_k=3):
    impact = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    }).sort_values(by="shap_value", key=abs, ascending=False)

    return impact.head(top_k)

def main():
    X, X_sample, y, model = load_data_and_model()

    explainer = create_explainer(model)
    shap_values = compute_shap_values(explainer, X_sample)

    global_feature_importance(shap_values, X_sample)
    explain_single_prediction(explainer, model, X)


if __name__ == "__main__":
    main()
