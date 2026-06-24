"""
base_model.py - Abstract base class for disaster prediction models
Defines the interface that all disaster models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class DisasterType(Enum):
    """Enumeration of supported disaster types."""
    EARTHQUAKE = "earthquake"
    FLOOD = "flood"
    CYCLONE = "cyclone"
    DROUGHT = "drought"
    HEATWAVE = "heatwave"


class RiskLevel(Enum):
    """Standardized risk levels across all disaster types."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class DisasterPrediction:
    """Standardized prediction output across all disaster models."""
    disaster_type: DisasterType
    probability: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    risk_level: RiskLevel
    recommendations: List[str]
    metadata: Dict[str, Any]
    model_version: str
    prediction_timestamp: str


@dataclass
class DisasterFeatures:
    """Base class for disaster-specific features."""
    location: Dict[str, Any]  # Contains lat, lon, district, state, etc.
    disaster_type: DisasterType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary."""
        return {
            'location': self.location,
            'disaster_type': self.disaster_type.value
        }


class BaseDisasterModel(ABC):
    """Abstract base class for all disaster prediction models."""
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        self.model_name = model_name
        self.model_version = model_version
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the model from disk.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def preprocess_features(self, features: Dict[str, Any]) -> Any:
        """
        Preprocess raw features for the model.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Preprocessed features in model-specific format
        """
        pass
    
    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> DisasterPrediction:
        """
        Make a prediction for the given features.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            DisasterPrediction with standardized output
        """
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Get list of required feature names for this model.
        
        Returns:
            List of feature names
        """
        pass
    
    @abstractmethod
    def get_recommendations(self, prediction: DisasterPrediction) -> List[str]:
        """
        Generate recommendations based on prediction.
        
        Args:
            prediction: Disaster prediction object
            
        Returns:
            List of recommendation strings
        """
        pass
    
    def calculate_risk_level(self, probability: float) -> RiskLevel:
        """
        Convert probability to risk level.
        
        Args:
            probability: Disaster probability (0.0 to 1.0)
            
        Returns:
            RiskLevel enum value
        """
        if probability < 0.25:
            return RiskLevel.LOW
        elif probability < 0.50:
            return RiskLevel.MODERATE
        elif probability < 0.75:
            return RiskLevel.HIGH
        elif probability < 0.90:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME
    
    def validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate that all required features are present.
        
        Args:
            features: Feature dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required = self.get_required_features()
        return all(feature in features for feature in required)
