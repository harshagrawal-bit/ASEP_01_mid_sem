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
print(synthetic_dataset.tail());




# asyncio.run(live_power())
