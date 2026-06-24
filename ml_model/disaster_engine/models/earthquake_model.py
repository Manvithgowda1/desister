"""
earthquake_model.py - Earthquake prediction model adapter
Wraps the existing earthquake XGBoost model for the unified disaster engine.
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
from feature_engineering import engineer_features, scale_features, get_all_feature_columns

from ..base_model import BaseDisasterModel, DisasterType, DisasterPrediction, RiskLevel


class EarthquakeModel(BaseDisasterModel):
    """Earthquake prediction model adapter."""
    
    def __init__(self, model_dir: str = None):
        super().__init__(
            model_name="Earthquake XGBoost",
            model_version="1.0.0"
        )
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "saved_models"
        )
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def load_model(self) -> bool:
        """Load the earthquake model and scaler."""
        try:
            # Load XGBoost model
            model_path = os.path.join(self.model_dir, "earthquake_model.json")
            if not os.path.exists(model_path):
                model_path = os.path.join(self.model_dir, "earthquake_model.pkl")
            
            if os.path.exists(model_path):
                if model_path.endswith('.json'):
                    self.model = xgb.XGBClassifier()
                    self.model.load_model(model_path)
                else:
                    self.model = joblib.load(model_path)
            else:
                print(f"Warning: Earthquake model not found at {model_path}")
                return False
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load feature columns
            cols_path = os.path.join(self.model_dir, "feature_columns.txt")
            if os.path.exists(cols_path):
                with open(cols_path, 'r') as f:
                    self.feature_columns = f.read().strip().split('\n')
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading earthquake model: {str(e)}")
            return False
    
    def get_required_features(self) -> List[str]:
        """Get required features for earthquake prediction."""
        return [
            'latitude', 'longitude', 'seismic_zone',
            'historical_eq_frequency', 'avg_historical_magnitude',
            'fault_proximity_km', 'population_density'
        ]
    
    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess raw features for earthquake prediction.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Preprocessed feature array
        """
        # Extract location and create feature dict
        location = features.get('location', {})
        
        feature_dict = {
            'latitude': location.get('latitude', 0.0),
            'longitude': location.get('longitude', 0.0),
            'seismic_zone': location.get('seismic_zone', 3),
            'historical_eq_frequency': location.get('historical_eq_frequency', 1),
            'avg_historical_magnitude': location.get('avg_historical_magnitude', 3.0),
            'fault_proximity_km': location.get('fault_proximity_km', 100.0),
            'population_density': location.get('population_density', 100.0)
        }
        
        # Create DataFrame
        df = pd.DataFrame([feature_dict])
        
        # Apply feature engineering
        df_engineered = engineer_features(df)
        
        # Get feature columns
        if self.feature_columns:
            feature_cols = self.feature_columns
        else:
            feature_cols = get_all_feature_columns()
        
        # Extract features
        X = df_engineered[feature_cols].values
        
        # Scale if scaler available
        if self.scaler:
            X = self.scaler.transform(X)
        
        return X
    
    def predict(self, features: Dict[str, Any]) -> DisasterPrediction:
        """
        Make earthquake prediction.
        
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
                disaster_type=DisasterType.EARTHQUAKE,
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
            'latitude': location.get('latitude'),
            'longitude': location.get('longitude'),
            'seismic_zone': location.get('seismic_zone'),
            'model_name': self.model_name
        }
        
        return DisasterPrediction(
            disaster_type=DisasterType.EARTHQUAKE,
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
        Generate earthquake-specific recommendations.
        
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
                "Standard building codes are sufficient",
                "Regular structural inspections recommended",
                "Maintain emergency kit for general preparedness"
            ])
        elif risk == RiskLevel.MODERATE:
            recommendations.extend([
                "Ensure building meets seismic safety standards",
                "Secure heavy furniture and fixtures",
                "Practice earthquake drills regularly",
                "Identify safe zones in your building"
            ])
        elif risk == RiskLevel.HIGH:
            recommendations.extend([
                "Retrofit building to meet enhanced seismic standards",
                "Install seismic sensors and early warning systems",
                "Stock emergency supplies for 72+ hours",
                "Develop and practice evacuation plans",
                "Consider structural reinforcement consultation"
            ])
        elif risk == RiskLevel.VERY_HIGH:
            recommendations.extend([
                "URGENT: Immediate structural assessment required",
                "Implement comprehensive seismic retrofitting",
                "Install real-time monitoring systems",
                "Prepare for immediate evacuation capability",
                "Coordinate with local emergency services",
                "Stock emergency supplies for 7+ days"
            ])
        elif risk == RiskLevel.EXTREME:
            recommendations.extend([
                "CRITICAL: Extreme seismic risk - immediate action required",
                "Consider relocation or major structural overhaul",
                "24/7 seismic monitoring mandatory",
                "Full emergency protocol implementation",
                "Coordinate with government disaster management agencies",
                "Stock emergency supplies for 14+ days"
            ])
        
        # Add general recommendations
        recommendations.extend([
            "Stay informed through official channels",
            "Follow local authority guidelines",
            "Keep important documents in waterproof containers"
        ])
        
        return recommendations
