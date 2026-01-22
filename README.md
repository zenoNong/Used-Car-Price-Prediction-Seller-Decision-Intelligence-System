#  Used-Car-Price-Prediction-Seller-Decision-Intelligence-System

An **end-to-end, industry-grade machine learning system** that predicts used car prices and converts raw ML outputs into **business-ready pricing decisions** using confidence scoring, explainability, and human-in-the-loop routing.

This project is inspired by real-world automotive marketplaces (e.g., C2B used-car platforms) and is designed to reflect **production ML engineering practices**, not just model training.

---

## 📌 Key Highlights

* End-to-end ML pipeline: **data → model → decisions → API**
* Business-aware **feature engineering** (depreciation, usage, market demand)
* Robust **model evaluation beyond RMSE** (MAPE, bucket-wise errors)
* **Explainable AI (SHAP)** for global & per-car justification
* **Decision Intelligence Layer** (auto-quote vs manual review)
* Fully deployed as a **FastAPI service**
* Production concerns handled: schema alignment, validation, confidence routing

---

## 🧠 Problem Statement

Used car pricing is inherently noisy due to:

* Non-linear depreciation
* Varying usage patterns
* Brand & market perception
* Sparse data for premium / rare models

A pure ML prediction is **not sufficient**. A real system must:

1. Predict a fair price
2. Know **when to trust itself**
3. Explain *why* a price was assigned
4. Route risky cases to human reviewers

This project solves all four.

---

## 🏗️ System Architecture

**High-level flow:**

```
Raw Data
   ↓
Data Cleaning & Validation
   ↓
EDA & Business Insights
   ↓
Feature Engineering
   ↓
Model Training (XGBoost)
   ↓
Evaluation & Error Analysis
   ↓
Explainability (SHAP)
   ↓
Decision Intelligence Layer
   ↓
FastAPI Deployment
```

📌 *Add architecture diagram image here*

---

## 📂 Project Structure

```
Used-Car-Price-Prediction-Seller-Decision-Intelligence-System/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── cleaned_used_cars.csv
│       └── featured_used_cars.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── save_feature_scheme.py
│   ├── data_cleaning.py
│   ├── explainability.py
│   ├── decision_engine.py
│   └── run_decision_engine.py
│
├── app/
│   └── main.py          # FastAPI app
│
├── models/
│   ├── xgboost_price_model.pkl
│   └── feature_schema.json
│
├── README.md
└── requirements.txt
```

---

## 🔍 Phase-wise Breakdown

### Phase 1 — Data Cleaning & Validation

* Removed missing / invalid entries
* Normalized numeric fields (engine, power, mileage)
* Standardized categorical values
* Ensured no target leakage

Output:

* `cleaned_used_cars.csv`

---

### Phase 2 — Exploratory Data Analysis (EDA)

EDA was driven by **business questions**, not visuals.

Key insights:

* Sharp depreciation in first 5–7 years
* Usage intensity (km/year) matters more than raw km
* Automatic transmission & popular brands command premiums
* Premium cars show higher pricing variance

📌 *Add EDA plots here: price vs age, km vs price, brand boxplots*

---

### Phase 3 — Feature Engineering

Engineered features to encode real-world pricing logic:

**Usage & Depreciation**

* `vehicle_age`
* `km_per_year`

**Performance Signals**

* `max_power`
* `power_per_cc`

**Market Demand Signals**

* `brand_popularity` (frequency-based)
* `model_rarity`

**Categorical Encoding**

* One-hot encoding for fuel, transmission, seller type
* Stable schema saved for inference

Output:

* `featured_used_cars.csv`

---

### Phase 4 — Modeling

Models trained & compared:

| Model             | Purpose                |
| ----------------- | ---------------------- |
| Linear Regression | Baseline sanity check  |
| Random Forest     | Non-linear baseline    |
| **XGBoost**       | Final production model |

Primary metric:

* **MAPE** (business-aligned for pricing systems)

Final choice:

* **XGBoost** due to lowest MAPE and stable behavior

---

### Phase 5 — Evaluation & Error Analysis

Beyond global metrics:

* Bucket-wise MAPE (low / mid / high / premium)
* Relative error distribution
* High-risk prediction identification

Sample results:

| Price Segment | MAPE |
| ------------- | ---- |
| Low           | ~18% |
| Mid           | ~13% |
| High          | ~10% |
| Premium       | ~11% |

📌 *Add error distribution & bucket MAPE plots here*

---

### Phase 6 — Explainability (SHAP)

Used **SHAP (TreeExplainer)** to ensure transparency.

**Global explainability:**

* Top drivers: vehicle age, power, mileage, brand popularity

**Local explainability:**

* Per-car force plots explaining price push & pull

This enables:

* Seller-facing justification
* Internal audit & trust

📌 *Add SHAP summary plot & force plot images here*

---

### Phase 7 — Seller Decision Intelligence Layer

Converted ML predictions into **business decisions**.

For each car:

* Predicted price
* Confidence score
* Price band
* Auto-quote vs manual review
* Reviewer notes

Decision rules:

| Confidence | Action                 |
| ---------- | ---------------------- |
| ≥ 0.85     | Auto-Quote             |
| 0.70–0.85  | Auto-Quote (Wide Band) |
| < 0.70     | Manual Review          |

This mirrors real marketplace workflows.

---

### Phase 8 — FastAPI Deployment

Exposed the full system via API:

```
POST /predict_price
```

Features:

* Strict input schema
* Feature schema alignment
* Domain validation (reject invalid cars)
* Explainable response

Sample response:

```json
{
  "predicted_price": 450000,
  "confidence_score": 0.9,
  "decision": "AUTO_QUOTE",
  "price_band": [427500, 472500],
  "key_price_drivers": ["vehicle_age", "km_driven", "max_power"],
  "reviewer_notes": ["Manual transmission"]
}
```

---

## 🧪 Testing

* Swagger UI (`/docs`) used for manual testing
* Tested normal, edge, and invalid inputs
* Domain validation prevents non-physical cars

---

## 🧠 Key Engineering Learnings

* Feature engineering > model choice
* Schema alignment is critical in production ML
* Tree models fallback to learned baselines
* ML systems must handle invalid input explicitly
* Explainability is mandatory for pricing systems

---

## 📈 Results Summary

* Stable MAPE across segments
* ~85–90% predictions auto-quotable
* ~10–15% routed to manual review
* Explainable, auditable decisions

📌 *Add final results summary plots here*

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 📌 Future Improvements

* Confidence calibration using residual modeling
* Time-aware pricing trends
* Seller-side UX integration
* Online learning / retraining pipeline

---

## 🏁 Conclusion

This project demonstrates how to build a **real-world ML pricing system**, not just a predictive model. It integrates data science, ML engineering, explainability, and business decision logic - closely mirroring production systems used in large-scale marketplaces.

---

**Author**: Zeno Nongmaithem 
>**Focus**: Data Science, Machine Learning, Decision Systems

