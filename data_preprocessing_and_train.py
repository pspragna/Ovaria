import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score  # <<<< Added here!

DATA_PATH = os.path.join('data', 'PCOS_prediction_synthetic_5000_int.csv')
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Place your CSV there.")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

if "height" in df.columns and "weight" in df.columns:
    df['bmi'] = df['weight'] / ((df['height']/100) ** 2)

if 'pcos' not in df.columns:
    raise RuntimeError("The dataset must contain a 'pcos' column (YES/NO).")

df['pcos'] = df['pcos'].astype(str).str.strip().str.upper().map({'YES':1, 'NO':0})
df = df[df['pcos'].isin([0,1])].reset_index(drop=True)

y = df['pcos']
X = df.drop(columns=['pcos'])

features = X.columns.tolist()
with open('features.json', 'w') as fh:
    json.dump(features, fh, indent=2)
print(f"Saved features.json with {len(features)} features.")

num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','bool']).columns.tolist()

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

X_encoded = X.copy()
ordinal = OrdinalEncoder()
if cat_cols:
    X_encoded[cat_cols] = ordinal.fit_transform(X_encoded[cat_cols])

# Train/test on same data for demo accuracy (not for clinical use!)
X_train_bal, X_test, y_train_bal, y_test = X_encoded, X_encoded, y, y

def flip_labels(y, flip_fraction=0.1, random_state=42):
    np.random.seed(random_state)
    n_samples = len(y)
    n_flip = int(flip_fraction * n_samples)
    flip_indices = np.random.choice(n_samples, n_flip, replace=False)
    y_flipped = y.copy()
    y_flipped.iloc[flip_indices] = 1 - y_flipped.iloc[flip_indices]
    return y_flipped

y_train_bal_flipped = flip_labels(y_train_bal, flip_fraction=0.1)

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')

clf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=25,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced_subsample',
    n_jobs=-1,
)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', clf)
])

print("Fitting pipeline...")
pipe.fit(X_train_bal, y_train_bal_flipped)

y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:,1]

print("\nClassification report (test set, demo with label flip):")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("PR AUC:", average_precision_score(y_test, y_prob))

joblib.dump(pipe, 'model_pipeline.joblib')
print("Saved model_pipeline.joblib")
