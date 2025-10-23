# utils.py
import os
import json
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_pipeline.joblib')
FEATURES_PATH = os.path.join(BASE_DIR, 'features.json')

# load model if exists; app will handle errors if missing
MODEL = None
if os.path.exists(MODEL_PATH):
    MODEL = joblib.load(MODEL_PATH)

EXPECTED_FEATURES = []
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, 'r') as fh:
        EXPECTED_FEATURES = json.load(fh)

def prepare_input_from_form(form):
    """
    Map request.form to a single-row DataFrame matching EXPECTED_FEATURES.
    Accepts booleans as 'yes/no' or '1/0', numeric strings, or missing -> NaN.
    """
    if not EXPECTED_FEATURES:
        raise RuntimeError("features.json not found in project root.")
    row = {}
    for col in EXPECTED_FEATURES:
        val = form.get(col)
        if val is None:
            # try with spaces substitution
            val = form.get(col.replace('_',' '))
        if val is None or val == '':
            row[col] = np.nan
            continue
        s = str(val).strip()
        low = s.lower()
        if low in ('yes','y','true','t','1','on'):
            row[col] = 1
            continue
        if low in ('no','n','false','f','0','off'):
            row[col] = 0
            continue
        try:
            if '.' in s:
                row[col] = float(s)
            else:
                row[col] = int(s)
        except:
            row[col] = s
    return pd.DataFrame([row], columns=EXPECTED_FEATURES)

def predict_from_df(df):
    """Return probability (0..1) using loaded MODEL"""
    if MODEL is None:
        raise RuntimeError("Model not found. Train the model or place model_pipeline.joblib in project root.")
    prob = MODEL.predict_proba(df)[:,1][0]
    return float(prob)

def get_recommendations(prob):
    p = float(prob)
    if p >= 0.7:
        level = "High risk"
        advice = [
            "See a gynecologist/endocrinologist for hormonal tests (AMH, LH/FSH, insulin).",
            "Adopt a low-glycemic diet and reduce refined carbs and sugary drinks.",
            "Aim for regular physical activity (150 min/week) and strength training.",
            "If BMI is high, aim for 5–10% weight loss — it often improves symptoms.",
        ]
    elif p >= 0.4:
        level = "At risk"
        advice = [
            "Improve sleep hygiene (7–8 hours).",
            "Increase daily steps and moderate exercise.",
            "Prefer balanced meals with protein and fiber.",
            "Monitor cycles and consult a doctor if symptoms persist.",
        ]
    else:
        level = "Low risk"
        advice = [
            "Maintain a healthy balanced lifestyle: exercise, sleep, wholesome food.",
            "Track menstrual cycles and report irregularities to provider."
        ]
    return {"level": level, "advice": advice}
