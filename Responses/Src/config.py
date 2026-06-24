import os
import sys

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeErrors when printing emojis
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

# Paths: repository Voice_Assistant/ (next to Responses/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SRC_DIR))

# Try loading environment variables from a .env file in project or repository root
for root in [_REPO_ROOT, _SRC_DIR]:
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip("'\"")
        except Exception as e:
            print(f"Error loading .env file: {e}")

PROJECT_ROOT = os.path.join(_REPO_ROOT, "Voice_Assistant")
MODELS_PATH = os.path.join(PROJECT_ROOT, "Models")
DATA_PATH = os.path.join(PROJECT_ROOT, "Data")

# Vosk Model
VOSK_MODEL_PATH = os.path.join(MODELS_PATH, "vosk-model-small-en-us-0.15")

# RAG Files
FAISS_INDEX_PATH = os.path.join(DATA_PATH, "rag_index.faiss")
METADATA_PATH = os.path.join(DATA_PATH, "rag_metadata.json")
FAQ_PATH = os.path.join(DATA_PATH, "emergency_faq.json")
IMAGE_CATALOG_PATH = os.path.join(DATA_PATH, "emergency_images.json")

# Audio Settings
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000
# None = system default microphone; set to an index from sd.query_devices()
AUDIO_INPUT_DEVICE = None
# End utterance after this many seconds of silence (partial speech)
SILENCE_END_SECONDS = 1.2

# Groq API Settings (primary — fast cloud inference)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# TTS Settings
TTS_RATE = 150
TTS_VOLUME = 0.9

# Crisis Detection
SOS_KEYWORDS = ["sos", "help me", "emergency", "urgent", "critical", "mayday"]
HIGH_URGENCY_KEYWORDS = [
    "bleeding", "blood", "stuck", "trapped", "drowning", "fire", "burning",
    "can't breathe", "chest pain", "unconscious", "choking", "severe pain",
    "broken bone", "head injury", "allergic reaction", "poisoned", "dying"
]

# Indian Emergency Helpline Numbers
INDIAN_HELPLINES = {
    "unified_emergency": "112",
    "police": "100",
    "fire": "101",
    "ambulance": "102",
    "emergency_ambulance": "108",
    "women_helpline": "1091",
    "child_helpline": "1098",
    "senior_citizen": "14567",
    "ndrf": "1078",
    "disaster_management": "011-26701728",
    "mental_health": "112",
    "poison_control": "1800-11-0039",
    "blood_bank": "1910",
    "covid": "1075",
    "railway": "182",
    "road_accident": "1033",
    "cyclone_warning": "1070",
    "earthquake": "011-26701728",
    "flood": "1070"
}

# Emergency-specific helpline mappings
EMERGENCY_HELPLINES = {
    "fire": "🔥 Fire Emergency: Call 101 (Fire Station)",
    "medical": "🚑 Medical Emergency: Call 108 (Emergency Ambulance) or 102 (Ambulance)",
    "police": "👮 Police Emergency: Call 100 (Police) or 112 (Unified Emergency)",
    "bleeding": "🚑 Medical Emergency: Call 108 (Emergency Ambulance) immediately",
    "choking": "🚑 Medical Emergency: Call 108 (Emergency Ambulance) immediately",
    "drowning": "🚑 Medical Emergency: Call 108 (Emergency Ambulance) immediately",
    "heart attack": "🚑 Cardiac Emergency: Call 108 (Emergency Ambulance) immediately",
    "stroke": "🚑 Stroke Emergency: Call 108 (Emergency Ambulance) immediately",
    "burn": "🔥 Burn Emergency: Call 101 (Fire) or 108 (Emergency Ambulance)",
    "earthquake": "🆘 Disaster Emergency: Call 1078 (NDRF) or 112 (Unified Emergency)",
    "flood": "🆘 Disaster Emergency: Call 1078 (NDRF) or 112 (Unified Emergency)",
    "cyclone": "🆘 Disaster Emergency: Call 1078 (NDRF) or 112 (Unified Emergency)",
    "poison": "🧠 Poison Emergency: Call 1800-11-0039 (Poison Control) or 108 (Ambulance)",
    "accident": "🚨 Accident Emergency: Call 1033 (Road Accident) or 108 (Ambulance)",
    "asthma": "🚑 Medical Emergency: Call 108 (Emergency Ambulance) immediately",
    "allergy": "🚑 Allergic Reaction: Call 108 (Emergency Ambulance) immediately",
    "seizure": "🚑 Medical Emergency: Call 108 (Emergency Ambulance) immediately",
    "diabetes": "🚑 Medical Emergency: Call 108 (Emergency Ambulance) immediately",
    "mental": "🧠 Mental Health Emergency: Call 112 (Unified Emergency) or 1098 (Child Helpline if under 18)",
    "pregnancy": "🚑 Medical Emergency: Call 108 (Emergency Ambulance) immediately",
    "general": "🚨 Emergency: Call 112 (Unified Emergency) for immediate assistance"
}

HELPLINE_TEXT = """Indian Emergency Numbers:
🚨 112 - Unified Emergency (Police, Fire, Ambulance)
👮 100 - Police
🔥 101 - Fire
🚑 102 - Ambulance
🚑 108 - Emergency Ambulance
👩 1091 - Women Helpline
👶 1098 - Child Helpline
👴 14567 - Senior Citizen Helpline
🆘 1078 - NDRF (National Disaster Response Force)
🧠 1800-11-0039 - Poison Control
🩸 1910 - Blood Bank
🚨 1033 - Road Accident Emergency"""

# BLE Settings (for SOS device)
BLE_DEVICE_NAME = "SOS_BEACON"  # Your BLE device name
BLE_SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
