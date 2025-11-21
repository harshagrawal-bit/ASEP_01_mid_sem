import asyncio
from tapo import ApiClient

async def main():
    client = ApiClient("harshagrawal.6996@gmail.com", "10Harsh2006")
    
    # Connect to plug
    device = await client.p110("192.168.0.108")  # update IP
    
    print("Connected to device")
    while True:
        choice = input("\n Turn plug ON? (y/n) or q to quit :").strip().lower()
        if choice == "y":
                # TURN ONN
                await device.on()
                print("Turned ON")
            # await asyncio.sleep(2)

        elif choice =="n":
            # Turn OFF
            await device.off()
            print("Turned OFF")

        elif choice == "q":
             print("Exiting..")
             break
        
        else:
             print("Invalid choice, enter y/n/q.")
             
asyncio.run(main())
