"""
flood_model.py - Flood prediction model adapter
Wraps the flood XGBoost model for the unified disaster engine.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import xgboost as xgb
import joblib

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from flood_preprocessing import FloodPreprocessor

from ..base_model import BaseDisasterModel, DisasterType, DisasterPrediction, RiskLevel


class FloodModel(BaseDisasterModel):
    """Flood prediction model adapter."""
    
    def __init__(self, model_dir: str = None):
        super().__init__(
            model_name="Flood XGBoost",
            model_version="1.0.0"
        )
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "saved_models"
        )
        self.model = None
        self.preprocessor = None
        
    def load_model(self) -> bool:
        """Load the flood model and preprocessor."""
        try:
            # Load XGBoost model
            model_path = os.path.join(self.model_dir, "flood_xgboost_model.json")
            if not os.path.exists(model_path):
                print(f"Warning: Flood model not found at {model_path}")
                return False
            
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            
            # Load preprocessor
            preprocessor_path = os.path.join(self.model_dir, "flood_preprocessor.pkl")
            if os.path.exists(preprocessor_path):
                self.preprocessor = FloodPreprocessor.load(self.model_dir)
            else:
                print(f"Warning: Flood preprocessor not found at {preprocessor_path}")
                return False
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading flood model: {str(e)}")
            return False
    
    def get_required_features(self) -> List[str]:
        """Get required features for flood prediction."""
        return [
            'district', 'state', 'basin', 'rainfall_mm',
            'river_water_level_m', 'soil_moisture', 'elevation_m',
            'historical_flood_events'
        ]
    
    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess raw features for flood prediction.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Preprocessed feature array
        """
        # Extract location and create feature dict
        location = features.get('location', {})
        
        feature_dict = {
            'district': location.get('district', 'Unknown'),
            'state': location.get('state', 'Unknown'),
            'basin': location.get('basin', 'Unknown'),
            'rainfall_mm': location.get('rainfall_mm', 500.0),
            'river_water_level_m': location.get('river_water_level_m', 5.0),
            'soil_moisture': location.get('soil_moisture', 0.5),
            'elevation_m': location.get('elevation_m', 100.0),
            'historical_flood_events': location.get('historical_flood_events', 0)
        }
        
        # Create DataFrame
        df = pd.DataFrame([feature_dict])
        
        # Preprocess using the flood preprocessor
        X = self.preprocessor.transform(df)
        
        return X
    
    def predict(self, features: Dict[str, Any]) -> DisasterPrediction:
        """
        Make flood prediction.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            DisasterPrediction with results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Preprocess
        X = self.preprocess_features(features)
        
        # Make prediction
        probability = self.model.predict_proba(X)[0, 1]
        
        # Calculate confidence (based on probability distance from 0.5)
        confidence = 1.0 - abs(probability - 0.5) * 2
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine risk level
        risk_level = self.calculate_risk_level(probability)
        
        # Generate recommendations
        recommendations = self.get_recommendations(
            DisasterPrediction(
                disaster_type=DisasterType.FLOOD,
                probability=probability,
                confidence=confidence,
                risk_level=risk_level,
                recommendations=[],
                metadata={},
                model_version=self.model_version,
                prediction_timestamp=datetime.now().isoformat()
            )
        )
        
        # Metadata
        location = features.get('location', {})
        metadata = {
            'district': location.get('district'),
            'state': location.get('state'),
            'basin': location.get('basin'),
            'rainfall_mm': location.get('rainfall_mm'),
            'river_level_m': location.get('river_water_level_m'),
            'model_name': self.model_name
        }
        
        return DisasterPrediction(
            disaster_type=DisasterType.FLOOD,
            probability=probability,
            confidence=confidence,
            risk_level=risk_level,
            recommendations=recommendations,
            metadata=metadata,
            model_version=self.model_version,
            prediction_timestamp=datetime.now().isoformat()
        )
    
    def get_recommendations(self, prediction: DisasterPrediction) -> List[str]:
        """
        Generate flood-specific recommendations.
        
        Args:
            prediction: Disaster prediction object
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        risk = prediction.risk_level
        prob = prediction.probability
        
        if risk == RiskLevel.LOW:
            recommendations.extend([
                "Standard drainage maintenance sufficient",
                "Monitor local weather forecasts during monsoon",
                "Keep basic emergency supplies ready",
                "Ensure proper drainage around property"
            ])
        elif risk == RiskLevel.MODERATE:
            recommendations.extend([
                "Clear drainage systems regularly",
                "Elevate electrical systems and appliances",
                "Prepare flood emergency kit with documents",
                "Know local evacuation routes",
                "Monitor river levels during heavy rainfall"
            ])
        elif risk == RiskLevel.HIGH:
            recommendations.extend([
                "Install flood barriers or sandbags preparation",
                "Move valuables to higher ground",
                "Prepare for possible evacuation",
                "Stock emergency supplies for 72+ hours",
                "Install water pump systems",
                "Review insurance coverage for flood damage"
            ])
        elif risk == RiskLevel.VERY_HIGH:
            recommendations.extend([
                "URGENT: Prepare for imminent flooding",
                "Implement flood barriers immediately",
                "Prepare for immediate evacuation",
                "Stock emergency supplies for 7+ days",
                "Secure vehicles and move to higher ground",
                "Coordinate with local emergency services",
                "Monitor official flood warnings continuously"
            ])
        elif risk == RiskLevel.EXTREME:
            recommendations.extend([
                "CRITICAL: Extreme flood risk - evacuate immediately if advised",
                "Move to designated safe zones or higher elevation",
                "Take essential documents and medications only",
                "Follow evacuation orders without delay",
                "Coordinate with emergency responders",
                "Stock emergency supplies for 14+ days if sheltering in place"
            ])
        
        # Add general recommendations
        recommendations.extend([
            "Stay tuned to All India Radio and DD News for updates",
            "Follow IMD (India Meteorological Department) alerts",
            "Avoid walking or driving through flood waters",
            "Keep emergency contact numbers handy"
        ])
        
        return recommendations
