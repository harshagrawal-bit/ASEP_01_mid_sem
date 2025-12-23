import asyncio
from datetime import datetime, timedelta
from collections import deque

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from tapo import ApiClient

# -----------------------
# CONFIG
# -----------------------
EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
IP = "192.168.0.108"

READ_INTERVAL_SEC = 5            # keep 5s for demo
PEAK_THRESHOLD_W = 300
PAUSE_TIME_SEC = 300             # 5 minutes

# gating thresholds (probabilities)
HOUR_GATE_PROBA = 0.60
QUARTER_GATE_PROBA = 0.60

# models
HOUR_MODEL_PATH = "hour_peak_classifier.pkl"
QUARTER_MODEL_PATH = "quarter_peak_classifier.pkl"

# 5-min ahead predictor settings
FORECAST_HORIZON_SEC = 300       # 5 minutes
MIN_TRAIN_SAMPLES = 80           # need some history (80 samples @ 5s ≈ 6.6 min)
LAG_COUNT = 12                   # last 12 samples = last 1 minute (12*5s)

# -----------------------
# HELPERS
# -----------------------
def normalize_to_watts(power_value):
    if power_value is None:
        return None
    power = float(power_value)
    if power > 1000:
        power /= 1000.0
    return power

def quarter_from_minute(minute: int) -> int:
    return min(3, max(0, minute // 15))

async def countdown(seconds):
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        print(f"⏳ Plug resumes in {mins:02d}:{secs:02d}", end="\r")
        await asyncio.sleep(1)
    print(" " * 60, end="\r")

async def connect_device():
    print(f"Connecting to Tapo P110 at {IP} ...")
    client = ApiClient(EMAIL, PASSWORD)
    try:
        device = await client.p110(IP)
        print("✅ Connected to plug.")
        return device
    except Exception as e:
        print("❌ Could not connect to plug.")
        print("Reason:", repr(e))
        return None

def build_lag_features(series, idx, lag_count):
    # series: list of floats
    # idx: current index in series for which we build features using past values
    # returns vector length lag_count + 2 (rolling mean + slope)
    lags = [series[idx - k] for k in range(1, lag_count + 1)]  # last lag_count values
    lags = lags[::-1]  # oldest -> newest

    rolling_mean = float(np.mean(lags))
    slope = float(lags[-1] - lags[0])  # simple trend over the window

    return lags + [rolling_mean, slope]

# -----------------------
# MAIN
# -----------------------
async def main():
    # Load gate models
    hour_bundle = joblib.load(HOUR_MODEL_PATH)
    hour_model = hour_bundle["model"]
    hour_features = hour_bundle["features"]
    hour_means = hour_bundle.get("dataset_plug_means", {})

    quarter_bundle = joblib.load(QUARTER_MODEL_PATH)
    quarter_model = quarter_bundle["model"]
    quarter_features = quarter_bundle["features"]
    quarter_means = quarter_bundle.get("plug_means_15m", {})

    device = await connect_device()
    if device is None:
        return

    # For inactive plugs, use averages from 15-min dataset means
    q_p2 = float(quarter_means.get("plug_2_15m_W", 0.0))
    q_p3 = float(quarter_means.get("plug_3_15m_W", 0.0))
    q_p4 = float(quarter_means.get("plug_4_15m_W", 0.0))
    q_total_mean = float(quarter_means.get("total_power_15m_W", 0.0))

    # Online training buffers
    times = []
    p1_series = []  # plug1 watts

    # Online regressor (retrained occasionally)
    reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    reg_ready = False
    last_train_size = 0

    while True:
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        # ---- read live plug power
        try:
            data = await device.get_current_power()
        except Exception as e:
            print(f"{ts} - ❌ Read power failed:", repr(e))
            await asyncio.sleep(READ_INTERVAL_SEC)
            continue

        plug1_raw = getattr(data, "current_power", None)
        plug1_w = normalize_to_watts(plug1_raw)

        if plug1_w is None:
            print(f"{ts} - Current power not available")
            await asyncio.sleep(READ_INTERVAL_SEC)
            continue

        print(f"{ts} - Plug_1 current: {plug1_w:.3f} W")

        # store for online learning
        times.append(now)
        p1_series.append(float(plug1_w))

        # ---- Gate A: next hour peak?
        next_hour = (now + timedelta(hours=1)).hour

        p2 = float(hour_means.get("plug_2_avg_W", 0.0))
        p3 = float(hour_means.get("plug_3_avg_W", 0.0))
        p4 = float(hour_means.get("plug_4_avg_W", 0.0))

        hour_row = {
            "hour": next_hour,
            "plug_1_avg_W": plug1_w,
            "plug_2_avg_W": p2,
            "plug_3_avg_W": p3,
            "plug_4_avg_W": p4
        }
        X_hour = [[hour_row[f] for f in hour_features]]
        p_peak_hour = float(hour_model.predict_proba(X_hour)[0][1])
        print(f"🧠 P(next hour peak) = {p_peak_hour:.2f}")

        if p_peak_hour < HOUR_GATE_PROBA:
            await asyncio.sleep(READ_INTERVAL_SEC)
            continue

        # ---- Gate B: next quarter peak?
        next_q_dt = now + timedelta(minutes=15)
        next_q = quarter_from_minute(next_q_dt.minute)
        hour_for_quarter = next_q_dt.hour

        quarter_row = {
            "hour": hour_for_quarter,
            "quarter": next_q,
            "plug_1_15m_W": plug1_w,
            "plug_2_15m_W": q_p2,
            "plug_3_15m_W": q_p3,
            "plug_4_15m_W": q_p4,
            "total_power_15m_W": q_total_mean,
            "is_peak_hour_house": 1
        }
        X_q = [[quarter_row[f] for f in quarter_features]]
        p_peak_quarter = float(quarter_model.predict_proba(X_q)[0][1])
        print(f"🧠 P(next quarter peak) = {p_peak_quarter:.2f}")

        if p_peak_quarter < QUARTER_GATE_PROBA:
            await asyncio.sleep(READ_INTERVAL_SEC)
            continue

        # ---- Online 5-min ahead prediction (plug_1)
        # Build training set from what we've collected so far:
        # X(t) = lags up to 1 minute, y(t) = power at t + horizon
        horizon_steps = max(1, int(FORECAST_HORIZON_SEC / READ_INTERVAL_SEC))

        # Need enough points so that: idx - LAG_COUNT >= 0 and idx + horizon_steps < len(series)
        max_idx_for_train = len(p1_series) - horizon_steps - 1

        if max_idx_for_train >= LAG_COUNT and len(p1_series) >= MIN_TRAIN_SAMPLES:
            # retrain only when dataset has grown a bit (avoids retrain every loop)
            if len(p1_series) - last_train_size >= 20:
                X_train = []
                y_train = []
                for idx in range(LAG_COUNT, max_idx_for_train):
                    X_train.append(build_lag_features(p1_series, idx, LAG_COUNT))
                    y_train.append(p1_series[idx + horizon_steps])

                X_train = np.array(X_train, dtype=float)
                y_train = np.array(y_train, dtype=float)

                reg.fit(X_train, y_train)
                reg_ready = True
                last_train_size = len(p1_series)
                print(f"✅ 5-min predictor trained on {len(X_train)} samples")

        if not reg_ready:
            print(f"⌛ Collecting data for 5-min predictor... ({len(p1_series)}/{MIN_TRAIN_SAMPLES})")
            await asyncio.sleep(READ_INTERVAL_SEC)
            continue

        # Predict 5 minutes ahead from latest lag window
        idx_now = len(p1_series) - 1
        if idx_now >= LAG_COUNT:
            X_now = np.array([build_lag_features(p1_series, idx_now, LAG_COUNT)], dtype=float)
            plug1_5min_pred = float(reg.predict(X_now)[0])
            print(f"🔮 Predicted plug_1 in +5 min: {plug1_5min_pred:.2f} W")

            # Convert to predicted total by adding avg of other plugs
            total_5min_pred = plug1_5min_pred + q_p2 + q_p3 + q_p4
            print(f"🔮 Predicted TOTAL in +5 min: {total_5min_pred:.2f} W (threshold {PEAK_THRESHOLD_W} W)")

            if total_5min_pred > PEAK_THRESHOLD_W:
                print("⚠ Forecast says threshold will be exceeded in ~5 minutes.")
                choice = input("Override? Keep plug ON (y/n): ").strip().lower()

                if choice == "y":
                    print("✅ Override accepted: plug stays ON.")
                else:
                    print("🔌 Turning plug OFF now (preventive action)...")
                    try:
                        await device.off()
                    except Exception as e:
                        print("❌ Failed to turn OFF:", repr(e))
                        await asyncio.sleep(READ_INTERVAL_SEC)
                        continue

                    await countdown(PAUSE_TIME_SEC)

                    print("🔌 Turning plug ON...")
                    try:
                        await device.on()
                    except Exception as e:
                        print("❌ Failed to turn ON:", repr(e))
                    else:
                        print("🔁 Plug resumed.")

        await asyncio.sleep(READ_INTERVAL_SEC)

if __name__ == "__main__":
    asyncio.run(main())
