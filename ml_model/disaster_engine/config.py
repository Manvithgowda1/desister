"""
config.py - Configuration management for disaster prediction engine
Centralized configuration for all disaster models and API settings.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for individual disaster models."""
    model_name: str
    model_path: str
    preprocessor_path: str = ""
    required_features: list = field(default_factory=list)
    enabled: bool = True
    version: str = "1.0.0"


@dataclass
class APIConfig:
    """Configuration for API settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])
    log_level: str = "info"
    max_batch_size: int = 100


@dataclass
class DisasterEngineConfig:
    """Main configuration for the disaster prediction engine."""
    # Base directory for models
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models"
    ))
    
    # API configuration
    api: APIConfig = field(default_factory=APIConfig)
    
    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    # Risk thresholds
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.25,
        "moderate": 0.50,
        "high": 0.75,
        "very_high": 0.90,
        "extreme": 1.0
    })
    
    # Feature validation
    validate_features: bool = True
    strict_validation: bool = False
    
    # Caching
    enable_cache: bool = True
    cache_ttl: int = 300  # seconds
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "disaster_engine.log"
    
    def __post_init__(self):
        """Initialize default model configurations."""
        if not self.models:
            self._init_default_models()
    
    def _init_default_models(self):
        """Initialize default model configurations."""
        self.models = {
            "earthquake": ModelConfig(
                model_name="Earthquake XGBoost",
                model_path=os.path.join(self.model_dir, "earthquake_model.json"),
                preprocessor_path=os.path.join(self.model_dir, "scaler.pkl"),
                required_features=[
                    'latitude', 'longitude', 'seismic_zone',
                    'historical_eq_frequency', 'avg_historical_magnitude',
                    'fault_proximity_km', 'population_density'
                ],
                enabled=True,
                version="1.0.0"
            ),
            "flood": ModelConfig(
                model_name="Flood XGBoost",
                model_path=os.path.join(self.model_dir, "flood_xgboost_model.json"),
                preprocessor_path=os.path.join(self.model_dir, "flood_preprocessor.pkl"),
                required_features=[
                    'district', 'state', 'basin', 'rainfall_mm',
                    'river_water_level_m', 'soil_moisture', 'elevation_m',
                    'historical_flood_events'
                ],
                enabled=True,
                version="1.0.0"
            ),
            "cyclone": ModelConfig(
                model_name="Cyclone Model (Placeholder)",
                model_path="",
                required_features=[
                    'latitude', 'longitude', 'district', 'state',
                    'wind_speed_kmh', 'pressure_hpa', 'sea_surface_temp_c',
                    'distance_from_coast_km', 'historical_cyclone_frequency'
                ],
                enabled=True,
                version="1.0.0"
            ),
            "drought": ModelConfig(
                model_name="Drought Model (Placeholder)",
                model_path="",
                required_features=[
                    'latitude', 'longitude', 'district', 'state',
                    'rainfall_deficit_mm', 'soil_moisture', 'temperature_c',
                    'reservoir_level_percent', 'groundwater_level_m',
                    'historical_drought_frequency'
                ],
                enabled=True,
                version="1.0.0"
            ),
            "heatwave": ModelConfig(
                model_name="Heatwave Model (Placeholder)",
                model_path="",
                required_features=[
                    'latitude', 'longitude', 'district', 'state',
                    'temperature_c', 'humidity_percent', 'heat_index_c',
                    'duration_days', 'night_temperature_c',
                    'historical_heatwave_frequency'
                ],
                enabled=True,
                version="1.0.0"
            )
        }
    
    def get_model_config(self, disaster_type: str) -> ModelConfig:
        """Get configuration for a specific disaster type."""
        return self.models.get(disaster_type.lower())
    
    def is_model_enabled(self, disaster_type: str) -> bool:
        """Check if a model is enabled."""
        config = self.get_model_config(disaster_type)
        return config.enabled if config else False
    
    def update_model_config(self, disaster_type: str, **kwargs):
        """Update configuration for a specific model."""
        if disaster_type.lower() in self.models:
            for key, value in kwargs.items():
                if hasattr(self.models[disaster_type.lower()], key):
                    setattr(self.models[disaster_type.lower()], key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DisasterEngineConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'base_dir': self.base_dir,
            'model_dir': self.model_dir,
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug,
                'cors_origins': self.api.cors_origins,
                'log_level': self.api.log_level,
                'max_batch_size': self.api.max_batch_size
            },
            'models': {
                name: {
                    'model_name': config.model_name,
                    'model_path': config.model_path,
                    'preprocessor_path': config.preprocessor_path,
                    'required_features': config.required_features,
                    'enabled': config.enabled,
                    'version': config.version
                }
                for name, config in self.models.items()
            },
            'risk_thresholds': self.risk_thresholds,
            'validate_features': self.validate_features,
            'strict_validation': self.strict_validation,
            'enable_cache': self.enable_cache,
            'cache_ttl': self.cache_ttl,
            'log_level': self.log_level,
            'log_file': self.log_file
        }


# Global configuration instance
_config: DisasterEngineConfig = None


def get_config() -> DisasterEngineConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = DisasterEngineConfig()
    return _config


def set_config(config: DisasterEngineConfig):
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config_from_file(config_path: str) -> DisasterEngineConfig:
    """Load configuration from JSON file."""
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = DisasterEngineConfig.from_dict(config_dict)
    set_config(config)
    return config


def save_config_to_file(config: DisasterEngineConfig, config_path: str):
    """Save configuration to JSON file."""
    import json
    
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
