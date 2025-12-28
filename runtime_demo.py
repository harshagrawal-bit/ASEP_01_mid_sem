import asyncio
from datetime import datetime, timedelta
from collections import deque

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from tapo import ApiClient

# -----------------------
# CONFIG - DEMO VERSION
# -----------------------
EMAIL = "harshagrawal.6996@gmail.com"
PASSWORD = "10Harsh2006"
IP = "192.168.0.104"

READ_INTERVAL_SEC = 5
PEAK_THRESHOLD_W = 30            # Low threshold for demo (triggers easily)
PAUSE_TIME_SEC = 30              # 30 seconds for demo

# gating thresholds (probabilities)
HOUR_GATE_PROBA = 0.01           # Low so it passes gate
QUARTER_GATE_PROBA = 0.01        # Low so it passes gate

# models
HOUR_MODEL_PATH = "hour_peak_classifier.pkl"
QUARTER_MODEL_PATH = "quarter_peak_classifier.pkl"

# Predictor settings - FAST for demo
FORECAST_HORIZON_SEC = 30        # Predict 30 seconds ahead
MIN_TRAIN_SAMPLES = 12           
LAG_COUNT = 6                    # Last 6 samples (30 seconds of history)

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
    """Build lag features for time series forecasting"""
    if idx < lag_count:
        return None
    
    lags = [series[idx - k] for k in range(1, lag_count + 1)]
    lags = lags[::-1]  # oldest -> newest

    rolling_mean = float(np.mean(lags))
    slope = float(lags[-1] - lags[0])

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

    # For inactive plugs, use averages
    q_p2 = float(quarter_means.get("plug_2_15m_W", 0.0))
    q_p3 = float(quarter_means.get("plug_3_15m_W", 0.0))
    q_p4 = float(quarter_means.get("plug_4_15m_W", 0.0))
    q_total_mean = float(quarter_means.get("total_power_15m_W", 0.0))

    # Online training buffers
    times = []
    p1_series = []

    # Online regressor
    reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    reg_ready = False
    last_train_size = 0

    print("\n🎬 DEMO MODE: Gates disabled, fast predictor enabled")
    print(f"   Threshold: {PEAK_THRESHOLD_W}W | Horizon: {FORECAST_HORIZON_SEC}s | Pause: {PAUSE_TIME_SEC}s\n")

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

        # ---- Gate A: next hour peak? (shown but not enforced)
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
        print(f"🧠 P(next hour peak) = {p_peak_hour:.2f} [gate disabled for demo]")

        # ---- Gate B: next quarter peak? (shown but not enforced)
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
        print(f"🧠 P(next quarter peak) = {p_peak_quarter:.2f} [gate disabled for demo]")

        # ---- Online predictor training
        horizon_steps = max(1, int(FORECAST_HORIZON_SEC / READ_INTERVAL_SEC))
        max_idx_for_train = len(p1_series) - horizon_steps - 1

        if max_idx_for_train >= LAG_COUNT and len(p1_series) >= MIN_TRAIN_SAMPLES:
            # retrain when dataset grows
            if len(p1_series) - last_train_size >= 5 or not reg_ready:
                X_train = []
                y_train = []
                
                for idx in range(LAG_COUNT, max_idx_for_train + 1):
                    features = build_lag_features(p1_series, idx, LAG_COUNT)
                    if features is not None:
                        X_train.append(features)
                        y_train.append(p1_series[idx + horizon_steps])

                if len(X_train) > 0:
                    X_train = np.array(X_train, dtype=float)
                    y_train = np.array(y_train, dtype=float)

                    reg.fit(X_train, y_train)
                    reg_ready = True
                    last_train_size = len(p1_series)
                    print(f"✅ Predictor trained on {len(X_train)} samples")

        if not reg_ready:
            need = MIN_TRAIN_SAMPLES + LAG_COUNT + horizon_steps
            print(f"⌛ Collecting data for predictor... ({len(p1_series)}/{need})")
            await asyncio.sleep(READ_INTERVAL_SEC)
            continue

        # ---- Predict ahead
        idx_now = len(p1_series) - 1
        if idx_now >= LAG_COUNT:
            features_now = build_lag_features(p1_series, idx_now, LAG_COUNT)
            if features_now is not None:
                X_now = np.array([features_now], dtype=float)
                plug1_pred = float(reg.predict(X_now)[0])
                print(f"🔮 Predicted plug_1 in +{FORECAST_HORIZON_SEC}s: {plug1_pred:.2f} W")

                # Convert to predicted total
                total_pred = plug1_pred + q_p2 + q_p3 + q_p4
                print(f"🔮 Predicted TOTAL in +{FORECAST_HORIZON_SEC}s: {total_pred:.2f} W (threshold {PEAK_THRESHOLD_W} W)")

                if total_pred > PEAK_THRESHOLD_W:
                    print(f"⚠ PEAK ALERT: Threshold will be exceeded in ~{FORECAST_HORIZON_SEC} seconds!")
                    print(f"🔌 Plug will turn OFF automatically to prevent peak...")
                    
                    # Start countdown for user to respond (5 seconds)
                    for i in range(5, 0, -1):
                        print(f"   Press 'y' within {i} seconds to keep plug ON... ", end="\r")
                        await asyncio.sleep(1)
                    
                    print("\n")  # Clear countdown line
                    
                    # Check if user wants to override (non-blocking check)
                    # For demo purposes, ask directly
                    choice = input("❓ Keep plug ON? (y=YES, any other key=turn OFF): ").strip().lower()

                    if choice == "y":
                        print("✅ User override: plug stays ON.")
                    else:
                        print("🔌 Turning plug OFF now (preventive peak shaving)...")
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
                            print("🔁 Plug resumed. System back to monitoring mode.")

        await asyncio.sleep(READ_INTERVAL_SEC)

if __name__ == "__main__":
    asyncio.run(main())
