# train_model.py  (HOURLY PEAK CLASSIFIER)
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

DATASET_CSV = "synthetic_energy_data.csv"
MODEL_OUT = "hour_peak_classifier.pkl"

# We predict: "is next hour a peak hour?"
FEATURES = ["hour", "plug_1_avg_W", "plug_2_avg_W", "plug_3_avg_W", "plug_4_avg_W"]
LABEL_COL = "is_peak_hour_house"

df = pd.read_csv(DATASET_CSV)

# Sort time order (important)
df["date_dt"] = pd.to_datetime(df["date"])
df = df.sort_values(["household_id", "date_dt", "hour"]).reset_index(drop=True)

# Create NEXT-HOUR label by shifting within each household
df["is_peak_next_hour"] = df.groupby("household_id")[LABEL_COL].shift(-1)

# Drop rows where next hour doesn't exist (last row per household)
df = df.dropna(subset=["is_peak_next_hour"]).copy()
df["is_peak_next_hour"] = df["is_peak_next_hour"].astype(int)

X = df[FEATURES].copy()
y = df["is_peak_next_hour"].copy()

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

# TimeSeriesSplit evaluation (time-aware CV) [web:110]
tscv = TimeSeriesSplit(n_splits=5)
accs = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accs.append(accuracy_score(y_test, preds))  # accuracy_score is standard metric [web:163]

print("TimeSeriesSplit accuracies:", [round(a, 3) for a in accs])
print("Mean accuracy:", round(float(np.mean(accs)), 3))

# Final fit on all data
model.fit(X, y)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "dataset_plug_means": {
            "plug_1_avg_W": float(df["plug_1_avg_W"].mean()),
            "plug_2_avg_W": float(df["plug_2_avg_W"].mean()),
            "plug_3_avg_W": float(df["plug_3_avg_W"].mean()),
            "plug_4_avg_W": float(df["plug_4_avg_W"].mean()),
        }
    },
    MODEL_OUT
)

print(classification_report(y, model.predict(X)))  # detailed classification metrics [web:166]
print(f"Saved model -> {MODEL_OUT}")
