feature_types = {
    "age": "number",
    "height": "number",
    "weight": "number",
    "first_period": "number",
    "regular_period": "select",
    "cycle_length": "number",
    "periods_skipped": "number",
    "blood_loss": "select",
    "weight_gain": "select",
    "excercise": "select",
    "dark_patches": "select",
    "sleep": "number",
    "step_count": "number",
    "balanced_meal": "select",
    "junk_food": "number",
    "acne": "select",
    "facial_hair": "select",
    "hair_thinning": "select",
    "history_pcos": "select",
    "history_diabetes": "select",
    "thyroid": "select",
    "anxiety_depression": "select",
    "mood_swings": "select",
    "difficulty_sleeping": "select",
    "low_energy": "select",
    "cravings": "select",
    
}

features = [{"name": feat, "type": feature_types.get(feat, "number")} for feat in feature_types]

import json
with open("features.json", "w") as fh:
    json.dump(features, fh, indent=2)
