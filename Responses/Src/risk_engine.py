import os
import json
import datetime
import requests

# Load path settings from config if available, else use default paths
try:
    from config import PROJECT_ROOT
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
except ImportError:
    # Fallback if config is not imported correctly
    _SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SRC_DIR))
    DATA_DIR = os.path.join(_REPO_ROOT, "Voice_Assistant", "Data")

DATABASE_PATH = os.path.join(DATA_DIR, "disaster_data.json")

# Coordinates of key districts in Karnataka, Kerala, and Tamil Nadu
LOCATION_COORDINATES = {
    "bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "karnataka"},
    "mysuru": {"lat": 12.2958, "lon": 76.6394, "state": "karnataka"},
    "kodagu": {"lat": 12.4244, "lon": 75.7382, "state": "karnataka"},
    "wayanad": {"lat": 11.6854, "lon": 76.1320, "state": "kerala"},
    "alappuzha": {"lat": 9.4981, "lon": 76.3388, "state": "kerala"},
    "chennai": {"lat": 13.0827, "lon": 80.2707, "state": "tamil nadu"},
    "nilgiris": {"lat": 11.4102, "lon": 76.6950, "state": "tamil nadu"},
    
    # State-level fallbacks (using capitals / central points)
    "karnataka": {"lat": 15.3173, "lon": 75.7139, "state": "karnataka"},
    "kerala": {"lat": 10.8505, "lon": 76.2711, "state": "kerala"},
    "tamil nadu": {"lat": 11.1271, "lon": 78.6569, "state": "tamil nadu"}
}

SUPPORTED_DISASTERS = ["earthquake", "flood", "cyclone", "landslide", "drought", "heatwave"]

# Global cache for weather forecast data to optimize query time
# Key: (round(lat, 2), round(lon, 2)), Value: (timestamp, weather_data_dict)
_weather_cache = {}

class RiskAssessmentEngine:
    def __init__(self):
        self.data = self._load_database()

    def _load_database(self):
        if os.path.exists(DATABASE_PATH):
            try:
                with open(DATABASE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error reading disaster database: {e}")
        return {}

    def fetch_weather_data(self, lat, lon):
        """Fetch current weather data from Open-Meteo API. Returns None if offline or failed."""
        global _weather_cache
        cache_key = (round(lat, 2), round(lon, 2))
        now = datetime.datetime.now()

        if cache_key in _weather_cache:
            timestamp, cached_data = _weather_cache[cache_key]
            # 1 hour cache validity
            if (now - timestamp).total_seconds() < 3600:
                return cached_data

        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max&timezone=Asia/Kolkata&forecast_days=7"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                daily = data.get("daily", {})
                result = {
                    "online": True,
                    "max_temp": max(daily.get("temperature_2m_max", [25.0])),
                    "total_rain": sum(daily.get("precipitation_sum", [0.0])),
                    "max_wind": max(daily.get("wind_speed_10m_max", [0.0]))
                }
                _weather_cache[cache_key] = (now, result)
                return result
        except Exception:
            pass
        return {"online": False}


    def get_climatology_fallback(self, state, month):
        """Get seasonal baseline normals for temperature and precipitation if offline."""
        climatology = self.data.get("climatology", {})
        state_clim = climatology.get(state.lower(), {})
        
        # Months are 1-indexed (Jan=1)
        rain_normals = state_clim.get("rainfall", [50]*12)
        temp_normals = state_clim.get("temperature", [27]*12)
        
        # Month index is 0-11
        idx = max(0, min(11, month - 1))
        return {
            "avg_temp": temp_normals[idx],
            "avg_rain": rain_normals[idx]
        }

    def resolve_location(self, location_name):
        """Map text location to standard name and state."""
        loc_lower = location_name.lower().strip()
        
        # Spelling variants / alias mapping
        aliases = {
            "waynad": "wayanad",
            "alleppey": "alappuzha",
            "ooty": "nilgiris",
            "bangalore": "bengaluru",
            "mysore": "mysuru",
            "coorg": "kodagu"
        }
        if loc_lower in aliases:
            loc_lower = aliases[loc_lower]

        # Check coordinates map first
        if loc_lower in LOCATION_COORDINATES:
            coord = LOCATION_COORDINATES[loc_lower]
            return loc_lower, coord["state"], coord["lat"], coord["lon"]
            
        # Fallback search inside disaster_data districts
        for state, state_data in self.data.items():
            if state == "climatology":
                continue
            districts = state_data.get("districts", {})
            for dist in districts.keys():
                if dist in loc_lower or loc_lower in dist:
                    coord = LOCATION_COORDINATES.get(dist, LOCATION_COORDINATES[state])
                    return dist, state, coord["lat"], coord["lon"]
                    
        # State-wide match fallback
        for state in ["karnataka", "kerala", "tamil nadu"]:
            if state in loc_lower or loc_lower in state:
                coord = LOCATION_COORDINATES[state]
                return state, state, coord["lat"], coord["lon"]

        # Default fallback to Karnataka central point
        coord = LOCATION_COORDINATES["karnataka"]
        return "karnataka", "karnataka", coord["lat"], coord["lon"]

    def calculate_risk(self, location, disaster_type=None):
        """Calculate risk score (0-100) and qualitative risk level for given location and disaster."""
        loc_resolved, state, lat, lon = self.resolve_location(location)
        current_month = datetime.datetime.now().month
        
        # Fetch weather data (either online forecast or offline normals fallback)
        weather = self.fetch_weather_data(lat, lon)
        if not weather.get("online"):
            fallback = self.get_climatology_fallback(state, current_month)
            weather["max_temp"] = fallback["avg_temp"]
            weather["total_rain"] = fallback["avg_rain"]
            weather["max_wind"] = 15.0 # Average default offline wind
            weather["online"] = False

        results = []
        target_disasters = [disaster_type.lower()] if (disaster_type and disaster_type.lower() in SUPPORTED_DISASTERS) else SUPPORTED_DISASTERS

        # Load vulnerability data
        state_data = self.data.get(state, {})
        district_data = state_data.get("districts", {}).get(loc_resolved, {})
        zone_data = state_data.get("zones", {})

        for dist_name in target_disasters:
            # Base historical score
            base_score = 20  # default
            reason = "General region vulnerability baseline."
            
            if district_data and dist_name in district_data:
                base_score = district_data[dist_name]["score"]
                reason = district_data[dist_name]["reason"]
            elif dist_name in zone_data:
                # If district specific score not available, use state/zone level score
                base_score = zone_data[dist_name].get("score", base_score)
                # Check if district is in high-risk list
                high_risk_list = zone_data[dist_name].get("high_risk_districts", [])
                if loc_resolved in high_risk_list:
                    base_score = min(90, base_score + 25)
                    reason = f"Identified as high-risk district for {dist_name} in {state.capitalize()} hazard map."
                else:
                    reason = zone_data[dist_name].get("reason", f"General state-level {dist_name} risk profile.")

            # Dynamic weather adjustments
            score = base_score
            details = []
            
            if dist_name == "earthquake":
                # Weather has no impact on earthquakes
                score = base_score
                details.append("Seismic zone risk is purely tectonic and unaffected by weather.")
                
            elif dist_name == "flood":
                if weather["online"]:
                    rain = weather["total_rain"]
                    details.append(f"7-day precipitation forecast: {rain:.1f} mm.")
                    if rain > 150:
                        score = min(100, score + 40)
                        details.append("Extreme rainfall forecast triggers high risk of flooding.")
                    elif rain > 70:
                        score = min(100, score + 25)
                        details.append("Heavy rainfall forecast. Watch for localized waterlogging.")
                    elif rain < 10:
                        score = max(0, score - 15)
                        details.append("Dry weather forecast reduces short-term flood risk.")
                else:
                    rain = weather["total_rain"]
                    details.append(f"Climatological average rainfall for month: {rain:.1f} mm.")
                    if rain > 200:
                        score = min(100, score + 20)
                        details.append("Active monsoon month; typical season for elevated water levels.")
                    elif rain < 20:
                        score = max(0, score - 15)
                        details.append("Dry season climatology reduces risk.")

            elif dist_name == "cyclone":
                # Cyclone risk is mostly coastal + wind speed
                if weather["online"]:
                    wind = weather["max_wind"]
                    details.append(f"Forecast max wind speed: {wind:.1f} km/h.")
                    if wind > 60:
                        score = min(100, score + 35)
                        details.append("High wind speed forecast indicates possible cyclonic system or depression.")
                    elif wind > 35:
                        score = min(100, score + 15)
                        details.append("Moderate winds forecasted.")
                else:
                    # Check if coastal state and month is prime cyclone season (Oct-Dec or Apr-May)
                    details.append("Offline mode: utilizing seasonal cyclone patterns.")
                    if state in ["tamil nadu", "kerala"] and current_month in [4, 5, 10, 11, 12]:
                        score = min(100, score + 15)
                        details.append("Currently in active post-monsoon cyclone season for East/West Coast.")

            elif dist_name == "landslide":
                # Relies heavily on slope + current rainfall
                if weather["online"]:
                    rain = weather["total_rain"]
                    details.append(f"Accumulated 7-day rainfall: {rain:.1f} mm.")
                    if score > 30: # Only if hilly/vulnerable topography
                        if rain > 120:
                            score = min(100, score + 40)
                            details.append("High rainfall on susceptible terrain creates extreme landslide conditions.")
                        elif rain > 50:
                            score = min(100, score + 20)
                            details.append("Moderate rainfall increases landslide likelihood.")
                    else:
                        score = max(0, score - 5)
                        details.append("Flat topography lowers risk regardless of rain.")
                else:
                    rain = weather["total_rain"]
                    if score > 30:
                        if rain > 250:
                            score = min(100, score + 20)
                            details.append("Heavy seasonal monsoon rains match historical landslide periods.")
                    else:
                        details.append("Topography not prone to landslides.")

            elif dist_name == "drought":
                # Relies on lack of rainfall
                if weather["online"]:
                    rain = weather["total_rain"]
                    details.append(f"Precipitation forecast: {rain:.1f} mm.")
                    if rain < 5 and score > 40:
                        score = min(100, score + 15)
                        details.append("Persistent lack of rainfall exacerbates drought conditions.")
                else:
                    rain = weather["total_rain"]
                    if rain < 20 and score > 40:
                        score = min(100, score + 10)
                        details.append("Historical dry season increases moisture deficit.")

            elif dist_name == "heatwave":
                temp = weather["max_temp"]
                details.append(f"Forecast max temperature: {temp:.1f}°C.")
                if temp > 43:
                    score = min(100, score + 45)
                    details.append("Critical temperature threshold crossed! Severe heatwave warning.")
                elif temp > 40:
                    score = min(100, score + 30)
                    details.append("Temperatures exceeding 40°C indicate active heatwave conditions.")
                elif temp < 30:
                    score = max(0, score - 25)
                    details.append("Cooler temperatures mitigate heatwave risk.")

            # Map score to risk level
            if score <= 33:
                level = "Low"
            elif score <= 66:
                level = "Moderate"
            else:
                level = "High"

            results.append({
                "disaster": dist_name.capitalize(),
                "score": int(score),
                "level": level,
                "reason": reason,
                "details": " ".join(details),
                "location": loc_resolved.capitalize(),
                "state": state.capitalize(),
                "online": weather.get("online", False)
            })

        return results
