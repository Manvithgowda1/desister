"""
flood_api.py - FastAPI endpoint for flood prediction
REST API for predicting flood risk with probability and risk category.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import joblib
from datetime import datetime

from flood_preprocessing import FloodPreprocessor


# Pydantic models for request/response
class FloodPredictionRequest(BaseModel):
    """Request model for flood prediction."""
    district: str = Field(..., description="District name in India")
    state: str = Field(..., description="State name in India")
    basin: str = Field(..., description="River basin name")
    rainfall_mm: float = Field(..., ge=0, le=5000, description="Rainfall in millimeters")
    river_water_level_m: float = Field(..., ge=0, le=50, description="River water level in meters")
    soil_moisture: float = Field(..., ge=0, le=1, description="Soil moisture (0-1)")
    elevation_m: float = Field(..., ge=0, le=3000, description="Elevation in meters")
    historical_flood_events: int = Field(..., ge=0, le=100, description="Number of historical flood events")


class FloodPredictionResponse(BaseModel):
    """Response model for flood prediction."""
    district: str
    state: str
    flood_probability: float = Field(..., ge=0, le=100, description="Flood probability percentage")
    risk_category: str = Field(..., description="Risk category: Low, Moderate, High, or Very High")
    prediction_timestamp: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    predictions: List[FloodPredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[FloodPredictionResponse]
    total_predictions: int
    high_risk_count: int
    average_probability: float


# Initialize FastAPI app
app = FastAPI(
    title="Flood Prediction API for India",
    description="XGBoost-based flood risk prediction system for Indian districts",
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

# Global variables for model and preprocessor
model = None
preprocessor = None
model_loaded = False
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')


def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor, model_loaded
    
    try:
        # Load model
        model_path = os.path.join(MODEL_DIR, 'flood_xgboost_model.json')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        # Load preprocessor
        preprocessor_path = os.path.join(MODEL_DIR, 'flood_preprocessor.pkl')
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
        
        preprocessor = FloodPreprocessor.load(MODEL_DIR)
        
        model_loaded = True
        print("Model and preprocessor loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


def get_risk_category(probability: float) -> str:
    """
    Convert probability to risk category.
    
    Args:
        probability: Flood probability (0-100)
        
    Returns:
        Risk category string
    """
    if probability < 25:
        return "Low"
    elif probability < 50:
        return "Moderate"
    elif probability < 75:
        return "High"
    else:
        return "Very High"


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model_and_preprocessor()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Flood Prediction API for India",
        "version": "1.0.0",
        "status": "active" if model_loaded else "model_not_loaded",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "health": "/health",
            "model_info": "/model_info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model_info")
async def model_info():
    """Get model information."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "features": preprocessor.get_feature_names() if preprocessor else [],
        "model_loaded": model_loaded,
        "model_dir": MODEL_DIR
    }


@app.post("/predict", response_model=FloodPredictionResponse)
async def predict_flood(request: FloodPredictionRequest):
    """
    Predict flood risk for a single location.
    
    Args:
        request: FloodPredictionRequest with features
        
    Returns:
        FloodPredictionResponse with probability and risk category
    """
    if not model_loaded:
        if not load_model_and_preprocessor():
            raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([{
            'district': request.district,
            'state': request.state,
            'basin': request.basin,
            'rainfall_mm': request.rainfall_mm,
            'river_water_level_m': request.river_water_level_m,
            'soil_moisture': request.soil_moisture,
            'elevation_m': request.elevation_m,
            'historical_flood_events': request.historical_flood_events
        }])
        
        # Preprocess
        input_processed = preprocessor.transform(input_data)
        
        # Predict
        probability = model.predict_proba(input_processed)[0, 1]
        probability_percentage = probability * 100
        
        # Get risk category
        risk_category = get_risk_category(probability_percentage)
        
        return FloodPredictionResponse(
            district=request.district,
            state=request.state,
            flood_probability=round(probability_percentage, 2),
            risk_category=risk_category,
            prediction_timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict_flood(request: BatchPredictionRequest):
    """
    Predict flood risk for multiple locations.
    
    Args:
        request: BatchPredictionRequest with multiple predictions
        
    Returns:
        BatchPredictionResponse with all predictions
    """
    if not model_loaded:
        if not load_model_and_preprocessor():
            raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Convert requests to DataFrame
        input_data = pd.DataFrame([{
            'district': pred.district,
            'state': pred.state,
            'basin': pred.basin,
            'rainfall_mm': pred.rainfall_mm,
            'river_water_level_m': pred.river_water_level_m,
            'soil_moisture': pred.soil_moisture,
            'elevation_m': pred.elevation_m,
            'historical_flood_events': pred.historical_flood_events
        } for pred in request.predictions])
        
        # Preprocess
        input_processed = preprocessor.transform(input_data)
        
        # Predict
        probabilities = model.predict_proba(input_processed)[:, 1]
        probabilities_percentage = probabilities * 100
        
        # Create responses
        predictions = []
        high_risk_count = 0
        
        for i, (pred, prob) in enumerate(zip(request.predictions, probabilities_percentage)):
            risk_category = get_risk_category(prob)
            if risk_category in ["High", "Very High"]:
                high_risk_count += 1
            
            predictions.append(FloodPredictionResponse(
                district=pred.district,
                state=pred.state,
                flood_probability=round(prob, 2),
                risk_category=risk_category,
                prediction_timestamp=datetime.now().isoformat(),
                model_version="1.0.0"
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            high_risk_count=high_risk_count,
            average_probability=round(probabilities_percentage.mean(), 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Ensure model is loaded before starting server
    if not load_model_and_preprocessor():
        print("Warning: Model could not be loaded. API will return errors.")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
