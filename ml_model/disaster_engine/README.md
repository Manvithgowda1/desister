# Unified Disaster Prediction Engine

Production-ready disaster prediction system for India with automatic routing to appropriate models based on disaster type.

## Architecture Overview

The unified disaster prediction engine provides a single API endpoint that automatically routes prediction requests to specialized models:

- **Earthquake** → XGBoost Earthquake Model
- **Flood** → XGBoost Flood Model
- **Cyclone** → Cyclone Model (placeholder)
- **Drought** → Drought Model (placeholder)
- **Heatwave** → Heatwave Model (placeholder)

## Features

- **Automatic Model Routing**: Dispatcher automatically selects the appropriate model based on disaster type
- **Standardized Output**: All models return consistent response format with probability, confidence, risk level, and recommendations
- **Production-Ready**: FastAPI with CORS, health checks, and batch prediction support
- **Extensible**: Easy to add new disaster types by implementing the base model interface
- **Configuration Management**: Centralized configuration for all models and API settings

## Installation

```bash
cd ml_model/disaster_engine
pip install -r requirements.txt
```

## Quick Start

### 1. Train Individual Models

First, train the individual disaster models:

```bash
# Train earthquake model
cd ../
python train.py

# Train flood model
python flood_train.py
```

### 2. Start the Unified API

```bash
cd disaster_engine
python api.py
```

The API will be available at `http://localhost:8000`

## API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "location": {
      "district": "Patna",
      "state": "Bihar",
      "latitude": 25.6,
      "longitude": 85.1,
      "basin": "Ganga",
      "rainfall_mm": 1200.0,
      "river_water_level_m": 15.0,
      "soil_moisture": 0.65,
      "elevation_m": 55.0,
      "historical_flood_events": 5
    },
    "disaster_type": "flood"
  }'
```

Response:
```json
{
  "disaster_type": "flood",
  "probability": 78.45,
  "confidence": 85.2,
  "risk_level": "very_high",
  "recommendations": [
    "URGENT: Prepare for imminent flooding",
    "Implement flood barriers immediately",
    "Prepare for immediate evacuation",
    ...
  ],
  "metadata": {
    "district": "Patna",
    "state": "Bihar",
    "basin": "Ganga",
    "model_name": "Flood XGBoost"
  },
  "model_version": "1.0.0",
  "prediction_timestamp": "2024-01-15T10:30:00"
}
```

### Earthquake Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "location": {
      "latitude": 26.0,
      "longitude": 91.0,
      "seismic_zone": 5,
      "historical_eq_frequency": 8,
      "avg_historical_magnitude": 5.8,
      "fault_proximity_km": 10,
      "population_density": 400
    },
    "disaster_type": "earthquake"
  }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "location": {
          "district": "Patna",
          "state": "Bihar",
          "basin": "Ganga",
          "rainfall_mm": 1200.0,
          "river_water_level_m": 15.0,
          "soil_moisture": 0.65,
          "elevation_m": 55.0,
          "historical_flood_events": 5
        },
        "disaster_type": "flood"
      },
      {
        "location": {
          "latitude": 26.0,
          "longitude": 91.0,
          "seismic_zone": 5,
          "historical_eq_frequency": 8,
          "avg_historical_magnitude": 5.8,
          "fault_proximity_km": 10,
          "population_density": 400
        },
        "disaster_type": "earthquake"
      }
    ]
  }'
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check with model status
- `GET /supported_disasters` - List of supported disaster types
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions

## Risk Levels

- **Low**: < 25% probability
- **Moderate**: 25-50% probability
- **High**: 50-75% probability
- **Very High**: 75-90% probability
- **Extreme**: > 90% probability

## Module Structure

```
disaster_engine/
├── __init__.py              # Package initialization
├── base_model.py            # Abstract base class for all disaster models
├── dispatcher.py            # Central dispatcher for routing predictions
├── config.py                # Configuration management
├── api.py                   # Unified FastAPI endpoint
├── requirements.txt         # Dependencies
├── README.md                # This file
└── models/
    ├── __init__.py
    ├── earthquake_model.py  # Earthquake model adapter
    ├── flood_model.py       # Flood model adapter
    ├── cyclone_model.py     # Cyclone model (placeholder)
    ├── drought_model.py     # Drought model (placeholder)
    └── heatwave_model.py    # Heatwave model (placeholder)
```

## Adding a New Disaster Type

### 1. Create Model Class

Create a new model class in `models/` that extends `BaseDisasterModel`:

```python
from ..base_model import BaseDisasterModel, DisasterType, DisasterPrediction, RiskLevel

class NewDisasterModel(BaseDisasterModel):
    def __init__(self):
        super().__init__(
            model_name="New Disaster Model",
            model_version="1.0.0"
        )
    
    def load_model(self) -> bool:
        # Load your trained model
        self.is_loaded = True
        return True
    
    def get_required_features(self) -> List[str]:
        return ['feature1', 'feature2', 'feature3']
    
    def preprocess_features(self, features: Dict[str, Any]) -> Any:
        # Preprocess features for your model
        return preprocessed_features
    
    def predict(self, features: Dict[str, Any]) -> DisasterPrediction:
        # Make prediction and return standardized output
        return DisasterPrediction(...)
    
    def get_recommendations(self, prediction: DisasterPrediction) -> List[str]:
        # Generate disaster-specific recommendations
        return ["recommendation1", "recommendation2"]
```

### 2. Register Model

Update `api.py` to register the new model:

```python
from.models import NewDisasterModel

# In initialize_dispatcher()
new_model = NewDisasterModel()
dispatcher.register_model(DisasterType.NEW_DISASTER, new_model)
```

### 3. Update DisasterType Enum

Add the new disaster type to `base_model.py`:

```python
class DisasterType(Enum):
    EARTHQUAKE = "earthquake"
    FLOOD = "flood"
    # ... existing types
    NEW_DISASTER = "new_disaster"
```

## Configuration

Edit `config.py` to customize:

```python
config = DisasterEngineConfig(
    api=APIConfig(
        host="0.0.0.0",
        port=8000,
        debug=False
    ),
    risk_thresholds={
        "low": 0.25,
        "moderate": 0.50,
        "high": 0.75,
        "very_high": 0.90,
        "extreme": 1.0
    }
)
```

## Model Status

| Disaster Type | Model Status | Data Source |
|---------------|--------------|-------------|
| Earthquake | ✅ Production | XGBoost trained on synthetic seismic data |
| Flood | ✅ Production | XGBoost trained on synthetic flood data |
| Cyclone | ⚠️ Placeholder | Requires real cyclone data |
| Drought | ⚠️ Placeholder | Requires real drought data |
| Heatwave | ⚠️ Placeholder | Requires real heatwave data |

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY ml_model/ .
RUN pip install disaster_engine/requirements.txt
CMD ["python", "disaster_engine/api.py"]
```

### Using Gunicorn

```bash
gunicorn disaster_engine.api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Environment Variables

Set environment variables for configuration:

```bash
export DISASTER_ENGINE_HOST=0.0.0.0
export DISASTER_ENGINE_PORT=8000
export DISASTER_ENGINE_LOG_LEVEL=INFO
export DISASTER_ENGINE_MODEL_DIR=/path/to/models
```

## Monitoring

The API includes health check endpoint:

```bash
curl http://localhost:8000/health
```

Returns model loading status and supported disaster types.

## Error Handling

The API handles errors gracefully:

- **400 Bad Request**: Invalid input or missing required features
- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Prediction error

## Performance Considerations

- Batch predictions are more efficient than individual calls
- Models are loaded once at startup
- Consider implementing caching for repeated predictions
- Monitor model loading time and prediction latency

## Security

- Implement rate limiting for production
- Add authentication/authorization as needed
- Validate all input features
- Sanitize error messages to avoid information leakage

## Testing

```python
from disaster_engine import get_dispatcher, DisasterType

# Test dispatcher
dispatcher = get_dispatcher()
prediction = dispatcher.predict(
    location={'latitude': 26.0, 'longitude': 91.0},
    disaster_type='earthquake'
)
print(f"Probability: {prediction.probability}")
print(f"Risk Level: {prediction.risk_level}")
```

## License

This is part of the Crisis AI Response project.
