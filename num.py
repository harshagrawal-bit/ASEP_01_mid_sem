import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_energy_data(num_days=30, num_households=5, num_plugs=4, peak_threshold_percentile=80):
    start_date = datetime(2025, 11, 1)
    
    records = []
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        for household in range(1, num_households + 1):
            # Generate daily base pattern for each plug (simulate daily usage fluctuations)
            base_pattern = np.random.uniform(low=20, high=100, size=num_plugs)
            for hour in range(24):
                # Simulate hourly variation and peak hour spikes
                hour_factor = 1 + 0.5 * np.sin(2 * np.pi * (hour - 7) / 24)  # Peak around 7-8 AM
                plug_powers = base_pattern * hour_factor
                # Add random noise
                plug_powers = plug_powers + np.random.normal(0, 5, size=num_plugs)
                plug_powers = np.clip(plug_powers, a_min=0, a_max=None)  # No negative power
                
                total_power = plug_powers.sum()
                
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "household_id": f"H{household}",
                    **{f"plug_{i+1}_avg_W": plug_powers[i] for i in range(num_plugs)},
                    "total_power_W": total_power
                })
                
    df = pd.DataFrame(records)
    
    # Define peaks - label hours with total_power_W above the percentile threshold as peaks per household per day
    def label_peaks(sub_df):
        threshold = np.percentile(sub_df["total_power_W"], peak_threshold_percentile)
        sub_df["is_peak_hour_house"] = (sub_df["total_power_W"] >= threshold).astype(int)
        return sub_df
    
    df = df.groupby(["household_id", "date"]).apply(label_peaks).reset_index(drop=True)
    
    # Save to CSV
    csv_filename = "synthetic_energy_data.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Synthetic dataset saved as {csv_filename} with {len(df)} rows.")
    
    return df

# Generate dataset
generate_synthetic_energy_data()