import asyncio
from bleak import BleakScanner

async def check_ble():
    print("🔍 Scanning for BLE devices... Please wait.")
    try:
        devices = await BleakScanner.discover(timeout=1.0)
        if not devices:
            print("❌ No BLE devices found. BLE may not be working or no devices are nearby.")
        else:
            print(f"✅ Found {len(devices)} BLE device(s):")
            for d in devices:
                print(f"📱 Name: {d.name or 'Unknown'}, Address: {d.address}")
    except Exception as e:
        print("❌ Error during BLE scan:", e)

if __name__ == "__main__":
    asyncio.run(check_ble())
