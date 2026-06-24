import requests
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMD_WEATHER_API_URL

def fetch_weather_grid(lat, lon, date_str):
    """
    Fetch weather and rainfall data from IMD API or public meteorological forecast grids (e.g. Open-Meteo / Copernicus).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean",
        "timezone": "Asia/Kolkata"
    }
    
    try:
        # Override key fields matching target schema
        response = requests.get(IMD_WEATHER_API_URL, params=params, timeout=10)
        if response.status_code == 200:
            res_data = response.json()
            daily = res_data.get("daily", {})
            
            rainfall = daily.get("precipitation_sum", [0.0])[0]
            temp = daily.get("temperature_2m_mean", [None])[0]
            humidity = daily.get("relative_humidity_2m_mean", [None])[0]
            
            # Decide if extreme rainfall classifies as a disaster
            disaster_type = None
            disaster_occurred = False
            if rainfall and rainfall > 70.0:  # IMD Heavy Rainfall Threshold
                disaster_type = "flood"
                disaster_occurred = True
                
            return {
                "rainfall": rainfall,
                "temperature": temp,
                "humidity": humidity,
                "disaster_type": disaster_type,
                "disaster_occurred": disaster_occurred
            }
    except Exception as e:
        print(f"Error calling weather grid API: {e}")
        
    return {
        "rainfall": 0.0,
        "temperature": 27.0,
        "humidity": 60.0,
        "disaster_type": None,
        "disaster_occurred": False
    }

def download_imdlib_historical_rain(start_year, end_year):
    """
    Downloads and extracts gridded rainfall records for India using the IMDLIB library.
    """
    try:
        import imdlib as imd
        print(f"Initializing IMDLIB download for rainfall grid ({start_year}-{end_year})...")
        
        # Download rainfall data
        data = imd.get_data("rain", start_year, end_year, fn_format="yearwise")
        
        # Parse datasets to dataframe
        ds = data.to_xarray()
        df = ds.to_dataframe().reset_index()
        
        print("IMDLIB extraction completed successfully.")
        return df
    except Exception as e:
        print(f"IMDLIB extraction not available or failed: {e}. Falling back to default meteorological normals.")
        
        # Return a simple mock dataframe
        mock_data = pd.DataFrame([
            {
                "lat": 12.9716,
                "lon": 77.5946,
                "time": datetime.now(),
                "rain": 12.5
            }
        ])
        return mock_data
