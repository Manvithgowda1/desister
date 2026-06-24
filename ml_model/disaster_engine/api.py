"""
api.py - Unified FastAPI endpoint for disaster prediction
Routes predictions to appropriate disaster models based on disaster type.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disaster_engine import (
    DisasterType, 
    DisasterPrediction, 
    RiskLevel,
    get_dispatcher
)
from disaster_engine.models import (
    EarthquakeModel, 
    FloodModel, 
    CycloneModel, 
    DroughtModel, 
    HeatwaveModel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class LocationInfo(BaseModel):
    """Location information for prediction."""
    district: Optional[str] = Field(None, description="District name")
    state: Optional[str] = Field(None, description="State name")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    
    # Disaster-specific features (optional, model-dependent)
    seismic_zone: Optional[int] = Field(None, ge=2, le=5, description="Seismic zone (2-5)")
    historical_eq_frequency: Optional[int] = Field(None, ge=0, description="Historical earthquake frequency")
    avg_historical_magnitude: Optional[float] = Field(None, ge=0, description="Average historical magnitude")
    fault_proximity_km: Optional[float] = Field(None, ge=0, description="Fault proximity in km")
    population_density: Optional[float] = Field(None, ge=0, description="Population density")
    
    basin: Optional[str] = Field(None, description="River basin name")
    rainfall_mm: Optional[float] = Field(None, ge=0, description="Rainfall in mm")
    river_water_level_m: Optional[float] = Field(None, ge=0, description="River water level in meters")
    soil_moisture: Optional[float] = Field(None, ge=0, le=1, description="Soil moisture (0-1)")
    elevation_m: Optional[float] = Field(None, ge=0, description="Elevation in meters")
    historical_flood_events: Optional[int] = Field(None, ge=0, description="Historical flood events")
    
    wind_speed_kmh: Optional[float] = Field(None, ge=0, description="Wind speed in km/h")
    pressure_hpa: Optional[float] = Field(None, ge=0, description="Atmospheric pressure in hPa")
    sea_surface_temp_c: Optional[float] = Field(None, description="Sea surface temperature")
    distance_from_coast_km: Optional[float] = Field(None, ge=0, description="Distance from coast")
    historical_cyclone_frequency: Optional[int] = Field(None, ge=0, description="Historical cyclone frequency")
    
    rainfall_deficit_mm: Optional[float] = Field(None, ge=0, description="Rainfall deficit in mm")
    reservoir_level_percent: Optional[float] = Field(None, ge=0, le=100, description="Reservoir level percentage")
    groundwater_level_m: Optional[float] = Field(None, description="Groundwater level in meters")
    historical_drought_frequency: Optional[int] = Field(None, ge=0, description="Historical drought frequency")
    
    temperature_c: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity_percent: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    heat_index_c: Optional[float] = Field(None, description="Heat index in Celsius")
    duration_days: Optional[int] = Field(None, ge=0, description="Duration in days")
    night_temperature_c: Optional[float] = Field(None, description="Night temperature in Celsius")
    historical_heatwave_frequency: Optional[int] = Field(None, ge=0, description="Historical heatwave frequency")


class DisasterPredictionRequest(BaseModel):
    """Request model for disaster prediction."""
    location: LocationInfo
    disaster_type: str = Field(..., description="Type of disaster: earthquake, flood, cyclone, drought, heatwave")


class DisasterPredictionResponse(BaseModel):
    """Response model for disaster prediction."""
    disaster_type: str
    probability: float = Field(..., ge=0, le=100, description="Disaster probability percentage")
    confidence: float = Field(..., ge=0, le=100, description="Confidence percentage")
    risk_level: str
    recommendations: List[str]
    metadata: Dict[str, Any]
    model_version: str
    prediction_timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    predictions: List[DisasterPredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[DisasterPredictionResponse]
    total_predictions: int
    high_risk_count: int
    average_probability: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: Dict[str, bool]
    supported_disasters: List[str]
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="Unified Disaster Prediction API for India",
    description="Production-ready disaster prediction system with automatic routing to appropriate models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dispatcher
dispatcher = None


def initialize_dispatcher():
    """Initialize the disaster dispatcher with all models."""
    global dispatcher
    dispatcher = get_dispatcher()
    
    # Register all models
    logger.info("Registering disaster models...")
    
    # Earthquake model
    earthquake_model = EarthquakeModel()
    dispatcher.register_model(DisasterType.EARTHQUAKE, earthquake_model)
    
    # Flood model
    flood_model = FloodModel()
    dispatcher.register_model(DisasterType.FLOOD, flood_model)
    
    # Cyclone model (placeholder)
    cyclone_model = CycloneModel()
    dispatcher.register_model(DisasterType.CYCLONE, cyclone_model)
    
    # Drought model (placeholder)
    drought_model = DroughtModel()
    dispatcher.register_model(DisasterType.DROUGHT, drought_model)
    
    # Heatwave model (placeholder)
    heatwave_model = HeatwaveModel()
    dispatcher.register_model(DisasterType.HEATWAVE, heatwave_model)
    
    # Load all models
    logger.info("Loading all models...")
    success = dispatcher.load_all_models()
    
    if success:
        logger.info("All models loaded successfully")
    else:
        logger.warning("Some models failed to load")
    
    return success


def prediction_to_response(prediction: DisasterPrediction) -> DisasterPredictionResponse:
    """Convert DisasterPrediction to API response."""
    return DisasterPredictionResponse(
        disaster_type=prediction.disaster_type.value,
        probability=round(prediction.probability * 100, 2),
        confidence=round(prediction.confidence * 100, 2),
        risk_level=prediction.risk_level.value,
        recommendations=prediction.recommendations,
        metadata=prediction.metadata,
        model_version=prediction.model_version,
        prediction_timestamp=prediction.prediction_timestamp
    )


@app.on_event("startup")
async def startup_event():
    """Initialize dispatcher on startup."""
    initialize_dispatcher()


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Unified Disaster Prediction API for India",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "health": "/health",
            "supported_disasters": "/supported_disasters"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if dispatcher is None:
        initialize_dispatcher()
    
    return HealthResponse(
        status="healthy",
        models_loaded=dispatcher.get_model_status(),
        supported_disasters=dispatcher.get_supported_disasters(),
        timestamp=datetime.now().isoformat()
    )


@app.get("/supported_disasters")
async def get_supported_disasters():
    """Get list of supported disaster types."""
    return {
        "supported_disasters": dispatcher.get_supported_disasters() if dispatcher else [],
        "descriptions": {
            "earthquake": "Seismic activity prediction using XGBoost",
            "flood": "Flood risk prediction using XGBoost",
            "cyclone": "Cyclone prediction (placeholder model)",
            "drought": "Drought prediction (placeholder model)",
            "heatwave": "Heatwave prediction (placeholder model)"
        }
    }


@app.post("/predict", response_model=DisasterPredictionResponse)
async def predict_disaster(request: DisasterPredictionRequest):
    """
    Predict disaster risk for a single location.
    
    Automatically routes to the appropriate model based on disaster_type.
    """
    if dispatcher is None:
        initialize_dispatcher()
    
    try:
        # Convert location to dict
        location_dict = request.location.dict(exclude_none=True)
        
        # Make prediction
        prediction = dispatcher.predict(
            location=location_dict,
            disaster_type=request.disaster_type
        )
        
        # Convert to response
        return prediction_to_response(prediction)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict_disaster(request: BatchPredictionRequest):
    """
    Predict disaster risk for multiple locations.
    
    Automatically routes each prediction to the appropriate model.
    """
    if dispatcher is None:
        initialize_dispatcher()
    
    try:
        # Convert requests to dispatcher format
        requests_list = [
            {
                'location': pred.location.dict(exclude_none=True),
                'disaster_type': pred.disaster_type
            }
            for pred in request.predictions
        ]
        
        # Make batch predictions
        predictions = dispatcher.batch_predict(requests_list)
        
        # Convert to responses
        responses = [prediction_to_response(pred) for pred in predictions]
        
        # Calculate statistics
        high_risk_count = sum(
            1 for r in responses 
            if r.risk_level in ['high', 'very_high', 'extreme']
        )
        avg_probability = sum(r.probability for r in responses) / len(responses) if responses else 0
        
        return BatchPredictionResponse(
            predictions=responses,
            total_predictions=len(responses),
            high_risk_count=high_risk_count,
            average_probability=round(avg_probability, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Initialize dispatcher
    initialize_dispatcher()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
