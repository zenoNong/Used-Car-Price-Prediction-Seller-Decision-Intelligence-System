def determine_action(confidence, predicted_price):
    if confidence >= 0.85:
        return "AUTO_QUOTE"
    elif confidence >= 0.70:
        return "AUTO_QUOTE_WITH_CAUTION"
    else:
        return "MANUAL_REVIEW"

def price_band(predicted_price, confidence):
    if confidence >= 0.85:
        delta = 0.05
    elif confidence >= 0.70:
        delta = 0.10
    else:
        return None

    lower = int(predicted_price * (1 - delta))
    upper = int(predicted_price * (1 + delta))

    return lower, upper

def reviewer_notes(row):
    notes = []

    if row["vehicle_age"] > 7:
        notes.append("High vehicle age")

    if row["km_driven"] > 100000:
        notes.append("High mileage")

    if row["brand_popularity"] < 0.05:
        notes.append("Low brand demand")

    if row["transmission_type_manual"] == 1:
        notes.append("Manual transmission")

    return notes

def build_decision_output(row, predicted_price, confidence, top_shap_features):
    action = determine_action(confidence, predicted_price)
    band = price_band(predicted_price, confidence)

    decision = {
        "predicted_price": int(predicted_price),
        "confidence_score": round(confidence, 3),
        "decision": action,
        "price_band": band,
        "key_price_drivers": top_shap_features,
        "reviewer_notes": reviewer_notes(row)
    }

    return decision
