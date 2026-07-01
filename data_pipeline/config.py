import os

# Database configurations (override with env variables if running in production/Docker)
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "crisis_ai_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

# Groq API Settings (primary — fast cloud inference)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_API_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


# API Keys & URLs
NCS_EARTHQUAKE_API_URL = os.environ.get("NCS_EARTHQUAKE_API_URL", "https://earthquake.usgs.gov/fdsnws/event/1/query")
IMD_WEATHER_API_URL = os.environ.get("IMD_WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast")

# Spatial bounding box for India (approximate coordinates for filtering)
INDIA_BOUNDS = {
    "min_lat": 8.0,
    "max_lat": 38.0,
    "min_lon": 68.0,
    "max_lon": 98.0
}
