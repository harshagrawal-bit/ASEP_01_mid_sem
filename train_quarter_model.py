import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

DATASET_15M = "synthetic_energy_data_15min.csv"
MODEL_OUT = "quarter_peak_classifier.pkl"

FEATURES = [
    "hour",
    "quarter",
    "plug_1_15m_W",
    "plug_2_15m_W",
    "plug_3_15m_W",
    "plug_4_15m_W",
    "total_power_15m_W",
    "is_peak_hour_house"
]
TARGET = "is_peak_quarter_next"

df = pd.read_csv(DATASET_15M)

# Drop last rows where "next" label is NaN
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

# Sort time order (important for time-series split)
df["date_dt"] = pd.to_datetime(df["date"])
df = df.sort_values(["household_id", "date_dt", "hour", "quarter"]).reset_index(drop=True)

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Time-series split evaluation (no leakage)
tscv = TimeSeriesSplit(n_splits=5)  # time-aware CV [web:110]
accs = []

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accs.append(accuracy_score(y_test, preds))

print("TimeSeriesSplit accuracies:", [round(a, 3) for a in accs])
print("Mean accuracy:", round(float(np.mean(accs)), 3))

# Train final model on full dataset and save
model.fit(X, y)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "plug_means_15m": {
            "plug_1_15m_W": float(df["plug_1_15m_W"].mean()),
            "plug_2_15m_W": float(df["plug_2_15m_W"].mean()),
            "plug_3_15m_W": float(df["plug_3_15m_W"].mean()),
            "plug_4_15m_W": float(df["plug_4_15m_W"].mean()),
            "total_power_15m_W": float(df["total_power_15m_W"].mean()),
        }
    },
    MODEL_OUT
)

print(f"Saved -> {MODEL_OUT}")
