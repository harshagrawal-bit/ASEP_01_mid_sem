import asyncio
import csv
from datetime import datetime
from tapo import ApiClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


EMAIL = "harshagrawal.6996@gmail.com"
PASSWORD = "10Harsh2006"
IP = "192.168.0.108"
CSV_FILE = "energy_data.csv"

# async def live_power():
#     client = ApiClient(EMAIL, PASSWORD)
#     device = await client.p110(IP)

#     while True:
#          data = await device.get_current_power()  # Returns CurrentPowerResult object
#          power = getattr(data, 'current_power', None)  # Access current_power attribute
#          timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#          if power is not None:
#              # Convert milliwatts to watts if needed
#              if power > 1000:
#                  power = power / 1000
#             #  print(f"{timestamp} - Current Power: {power:.3f} W")  
#              if power>=20:
#                     print(f"Power exceeded 20W, turning OFF the device.")
#                     await device.off()
#                     break
                
                
#          else:
#              print(f"{timestamp} - Current Power data not available")  
#          await asyncio.sleep(3)

        
synthetic_dataset = pd.read_csv("synthetic_energy_data.csv")
# print(synthetic_dataset['is_peak_hour_house'].value_counts())
X= synthetic_dataset[[
    "date",
    "hour",
    "household_id",
    "plug_1_avg_W",
    "plug_2_avg_W",
    "plug_3_avg_W",
    "plug_4_avg_W",
    "total_power_W"
]]

Y = synthetic_dataset["is_peak_hour_house"]
# print(X.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=4)
model = LogisticRegression()
model.fit(X_train,Y_train)
training_data_prediction = model.predict(X_train)
test_data_prediction = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_train, training_data_prediction))
#print(classification_report(y_test, y_pred))





# asyncio.run(live_power())
