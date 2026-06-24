"""
dispatcher.py - Disaster prediction router and dispatcher
Routes prediction requests to appropriate disaster models based on disaster type.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

from base_model import (
    BaseDisasterModel, 
    DisasterType, 
    DisasterPrediction,
    RiskLevel
)


class DisasterDispatcher:
    """Central dispatcher for routing predictions to appropriate models."""
    
    def __init__(self):
        self.models: Dict[DisasterType, BaseDisasterModel] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_model(self, disaster_type: DisasterType, model: BaseDisasterModel):
        """
        Register a disaster model for a specific disaster type.
        
        Args:
            disaster_type: Type of disaster this model handles
            model: Model instance
        """
        self.models[disaster_type] = model
        self.logger.info(f"Registered model for {disaster_type.value}: {model.model_name}")
        
    def load_all_models(self) -> bool:
        """
        Load all registered models.
        
        Returns:
            True if all models loaded successfully
        """
        success = True
        for disaster_type, model in self.models.items():
            try:
                if not model.load_model():
                    self.logger.error(f"Failed to load model for {disaster_type.value}")
                    success = False
                else:
                    self.logger.info(f"Successfully loaded model for {disaster_type.value}")
            except Exception as e:
                self.logger.error(f"Error loading model for {disaster_type.value}: {str(e)}")
                success = False
        return success
    
    def predict(self, location: Dict[str, Any], disaster_type: str) -> DisasterPrediction:
        """
        Route prediction request to appropriate model.
        
        Args:
            location: Location information (lat, lon, district, state, etc.)
            disaster_type: Type of disaster to predict
            
        Returns:
            DisasterPrediction with results
            
        Raises:
            ValueError: If disaster type is not supported or model not loaded
        """
        try:
            # Convert string to enum
            disaster_enum = DisasterType(disaster_type.lower())
        except ValueError:
            raise ValueError(f"Unsupported disaster type: {disaster_type}. "
                           f"Supported types: {[dt.value for dt in DisasterType]}")
        
        # Check if model is registered
        if disaster_enum not in self.models:
            raise ValueError(f"No model registered for disaster type: {disaster_type}")
        
        model = self.models[disaster_enum]
        
        # Check if model is loaded
        if not model.is_loaded:
            raise RuntimeError(f"Model for {disaster_type} is not loaded")
        
        # Prepare features
        features = {
            'location': location,
            'disaster_type': disaster_type
        }
        
        # Validate features
        if not model.validate_features(features):
            raise ValueError(f"Missing required features for {disaster_type} model")
        
        # Make prediction
        prediction = model.predict(features)
        
        self.logger.info(f"Prediction made for {disaster_type} at {location.get('district', 'unknown')}")
        
        return prediction
    
    def batch_predict(self, requests: list) -> list:
        """
        Make batch predictions for multiple requests.
        
        Args:
            requests: List of dictionaries with 'location' and 'disaster_type'
            
        Returns:
            List of DisasterPrediction objects
        """
        predictions = []
        for request in requests:
            try:
                prediction = self.predict(
                    location=request['location'],
                    disaster_type=request['disaster_type']
                )
                predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"Batch prediction failed: {str(e)}")
                # Return error prediction
                predictions.append(self._create_error_prediction(
                    location=request.get('location', {}),
                    disaster_type=request.get('disaster_type', 'unknown'),
                    error=str(e)
                ))
        
        return predictions
    
    def _create_error_prediction(self, location: Dict[str, Any], disaster_type: str, 
                                 error: str) -> DisasterPrediction:
        """Create an error prediction object."""
        return DisasterPrediction(
            disaster_type=DisasterType(disaster_type.lower()) if disaster_type in [dt.value for dt in DisasterType] else DisasterType.EARTHQUAKE,
            probability=0.0,
            confidence=0.0,
            risk_level=RiskLevel.LOW,
            recommendations=[f"Error: {error}"],
            metadata={'error': error, 'location': location},
            model_version="error",
            prediction_timestamp=datetime.now().isoformat()
        )
    
    def get_supported_disasters(self) -> list:
        """Get list of supported disaster types."""
        return [dt.value for dt in self.models.keys()]
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get loading status of all registered models."""
        return {
            dt.value: model.is_loaded 
            for dt, model in self.models.items()
        }


# Singleton instance
_dispatcher_instance: Optional[DisasterDispatcher] = None


def get_dispatcher() -> DisasterDispatcher:
    """Get the singleton dispatcher instance."""
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = DisasterDispatcher()
    return _dispatcher_instance
