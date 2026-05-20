import os

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

# Ollama Settings (local offline model)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3n:latest")

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

# BLE Settings (for SOS device)
BLE_DEVICE_NAME = "SOS_BEACON"  # Your BLE device name
BLE_SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
