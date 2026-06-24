# Disaster Prediction Integration for Crisis-AI Voice Assistant

This document describes the integration of the unified disaster prediction engine into the Crisis-AI voice assistant system.

## Overview

The disaster prediction integration allows users to ask voice queries about disaster risks and receive real-time predictions with probability, confidence, risk level, and recommendations.

## Voice Flow

```
User: "What is the earthquake probability in Karnataka?"
  ↓
1. Speech to Text (Vosk)
  ↓
2. Intent Classification (DisasterIntentClassifier)
  ↓
3. Entity Extraction (Location, Disaster Type)
  ↓
4. API Request (DisasterAPIClient → Disaster Engine)
  ↓
5. Response Generation (DisasterResponseGenerator)
  ↓
6. Voice Output (pyttsx3)
```

## Components

### 1. DisasterIntentClassifier (`disaster_intent_classifier.py`)

Detects disaster prediction intents in user queries and extracts entities.

**Features:**
- Intent classification (disaster_prediction, general_query, emergency)
- Disaster type detection (earthquake, flood, cyclone, drought, heatwave)
- Location extraction (Indian states, districts, cities)
- Entity extraction for disaster-specific features

**Supported Query Patterns:**
- "What is the earthquake probability in Karnataka?"
- "What's the flood risk in Patna?"
- "Tell me about cyclone warning in Odisha"
- "Heatwave risk in Rajasthan"

### 2. DisasterAPIClient (`disaster_api_client.py`)

Client for communicating with the disaster prediction API.

**Features:**
- HTTP requests to unified disaster prediction engine
- Automatic location data building from entities
- Health check and status monitoring
- Error handling and timeout management

**API Endpoints Used:**
- `POST /predict` - Single prediction
- `GET /health` - Health check
- `GET /supported_disasters` - List supported disasters

### 3. DisasterResponseGenerator (`disaster_response_generator.py`)

Converts prediction results into natural language responses for voice output.

**Features:**
- Risk-appropriate response generation
- Confidence-based phrasing
- Recommendation summarization
- Error response generation

**Response Examples:**
- **Low Risk:** "The earthquake probability in Bangalore is 15.5 percent, which indicates low risk. This prediction has high confidence. Here are some recommendations: Standard building codes are sufficient. Regular structural inspections recommended."
- **High Risk:** "Warning. The flood probability in Patna is 78.5 percent, which indicates very high risk. This prediction has high confidence. Please take immediate action. Urgent recommendations: URGENT: Prepare for imminent flooding. Implement flood barriers immediately."

### 4. DisasterIntegration (`disaster_integration.py`)

Main integration module that coordinates all components.

**Features:**
- Intent classification and routing
- API communication
- Response generation
- Error handling
- Configuration management

## Installation

### 1. Install Dependencies

```bash
pip install requests
```

### 2. Start the Disaster Prediction Engine

```bash
cd ml_model/disaster_engine
python api.py
```

The API will be available at `http://localhost:8000`

### 3. Configure Voice Assistant

Add to your `.env` file or set environment variables:

```bash
# Disaster Prediction API Settings
DISASTER_API_URL=http://localhost:8000
DISASTER_API_ENABLED=true
```

Or modify `config.py` directly:

```python
DISASTER_API_URL = "http://localhost:8000"
DISASTER_API_ENABLED = True
```

## Usage

### Starting the Voice Assistant

```bash
cd Responses/Src
python main_voice_assistant.py
```

### Example Voice Queries

**Earthquake:**
- "What is the earthquake probability in Karnataka?"
- "Seismic risk in Delhi"
- "Earthquake chance in Gujarat"

**Flood:**
- "What's the flood risk in Patna?"
- "Flooding probability in Assam"
- "River level warning in Bihar"

**Cyclone:**
- "Cyclone warning in Odisha"
- "Storm risk in West Bengal"
- "Hurricane probability in Andhra Pradesh"

**Drought:**
- "Drought risk in Maharashtra"
- "Water shortage probability in Rajasthan"
- "Rainfall deficit in Karnataka"

**Heatwave:**
- "Heatwave risk in Rajasthan"
- "Temperature warning in Delhi"
- "Heat stroke probability in Uttar Pradesh"

## Architecture

### Module Structure

```
Responses/Src/
├── disaster_intent_classifier.py    # Intent classification
├── disaster_api_client.py           # API communication
├── disaster_response_generator.py    # Response generation
├── disaster_integration.py          # Main integration
└── main_voice_assistant.py          # Updated with disaster integration
```

### Data Flow

```
User Voice Input
    ↓
VoiceHandler (Vosk STT)
    ↓
CrisisVoiceAssistant.process_voice_input()
    ↓
EmergencyDetector (check for SOS)
    ↓
DisasterIntegration.process_query()
    ↓
DisasterIntentClassifier.classify()
    ↓
DisasterAPIClient.predict()
    ↓
Disaster Prediction Engine API
    ↓
DisasterResponseGenerator.generate_response()
    ↓
VoiceHandler.speak() (pyttsx3 TTS)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISASTER_API_URL` | `http://localhost:8000` | Disaster prediction API URL |
| `DISASTER_API_ENABLED` | `true` | Enable/disable disaster prediction |

### Config File Settings

Edit `config.py`:

```python
# Disaster Prediction API Settings
DISASTER_API_URL = os.environ.get("DISASTER_API_URL", "http://localhost:8000")
DISASTER_API_ENABLED = os.environ.get("DISASTER_API_ENABLED", "true").lower() == "true"
```

## Testing

### Test Intent Classifier

```bash
python disaster_intent_classifier.py
```

### Test API Client

```bash
python disaster_api_client.py
```

### Test Response Generator

```bash
python disaster_response_generator.py
```

### Test Full Integration

```bash
python disaster_integration.py
```

## Error Handling

The integration handles various error scenarios:

1. **API Unavailable:** Returns user-friendly error message
2. **Invalid Location:** Prompts for valid Indian location
3. **Unsupported Disaster:** Informs user of supported types
4. **Network Timeout:** Graceful degradation with error message
5. **Malformed Response:** Error handling with fallback

## Supported Locations

### Indian States (All 28 states + 8 UTs)
- Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Goa
- Gujarat, Haryana, Himachal Pradesh, Jharkhand, Karnataka, Kerala
- Madhya Pradesh, Maharashtra, Manipur, Meghalaya, Mizoram, Nagaland
- Odisha, Punjab, Rajasthan, Sikkim, Tamil Nadu, Telangana, Tripura
- Uttar Pradesh, Uttarakhand, West Bengal, Delhi, Jammu and Kashmir
- Ladakh, Puducherry, Chandigarh, Andaman and Nicobar, Lakshadweep

### Major Cities/Districts (Sample)
- Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Ahmedabad
- Pune, Jaipur, Lucknow, Kanpur, Nagpur, Indore, Thane, Bhopal
- Visakhapatnam, Pimpri, Patna, Vadodara, Ghaziabad, Ludhiana
- And many more...

## Disaster Types

| Disaster Type | Model Status | Features Required |
|---------------|--------------|-------------------|
| Earthquake | ✅ Production | Seismic zone, fault proximity, historical frequency |
| Flood | ✅ Production | Rainfall, river level, soil moisture, elevation |
| Cyclone | ⚠️ Placeholder | Wind speed, pressure, distance from coast |
| Drought | ⚠️ Placeholder | Rainfall deficit, soil moisture, reservoir level |
| Heatwave | ⚠️ Placeholder | Temperature, humidity, heat index |

## Performance Considerations

- **API Latency:** Typical response time 100-500ms
- **Voice Processing:** STT + TTS adds ~2-3 seconds
- **Concurrent Requests:** Voice assistant processes one request at a time
- **Caching:** Not implemented (could be added for repeated queries)

## Troubleshooting

### API Not Available

**Symptom:** "Disaster prediction API is not available"

**Solution:**
1. Check if disaster engine is running: `curl http://localhost:8000/health`
2. Start the disaster engine: `cd ml_model/disaster_engine && python api.py`
3. Check firewall/network settings

### Location Not Recognized

**Symptom:** "I couldn't find the location"

**Solution:**
1. Use full state name (e.g., "Karnataka" not "Kar")
2. Use major city names
3. Check spelling

### Disaster Type Not Supported

**Symptom:** "That disaster type is not currently supported"

**Solution:**
1. Use supported disaster types: earthquake, flood, cyclone, drought, heatwave
2. Check disaster engine status: `curl http://localhost:8000/supported_disasters`

## Future Enhancements

- [ ] Add geocoding for better location resolution
- [ ] Implement caching for repeated queries
- [ ] Add temporal features (seasonality, trends)
- [ ] Support for multi-location queries
- [ ] Historical disaster data integration
- [ ] Real-time weather data integration
- [ ] Alert notifications for high-risk predictions

## License

This is part of the Crisis AI Response project.
