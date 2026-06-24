"""
disaster_api_client.py - Client for interacting with the disaster prediction API
Handles communication with the unified disaster prediction engine.
"""

import requests
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DisasterPredictionResult:
    """Result from disaster prediction API."""
    disaster_type: str
    probability: float
    confidence: float
    risk_level: str
    recommendations: list
    metadata: dict
    model_version: str
    prediction_timestamp: str


class DisasterAPIClient:
    """Client for the disaster prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.timeout = 10  # seconds
        
    def _build_location_data(self, entities: Dict[str, Any], disaster_type: str) -> Dict[str, Any]:
        """
        Build location data from extracted entities.
        
        Args:
            entities: Extracted entities from intent classifier
            disaster_type: Type of disaster
            
        Returns:
            Location data dictionary for API
        """
        location_data = {
            'district': entities.get('district'),
            'state': entities.get('state')
        }
        
        # Add disaster-specific features based on type
        if disaster_type == 'earthquake':
            # Default values for earthquake if not provided
            location_data.update({
                'latitude': entities.get('latitude', 20.0),  # Default to central India
                'longitude': entities.get('longitude', 78.0),
                'seismic_zone': entities.get('seismic_zone', 3),
                'historical_eq_frequency': entities.get('historical_eq_frequency', 1),
                'avg_historical_magnitude': entities.get('avg_historical_magnitude', 3.0),
                'fault_proximity_km': entities.get('fault_proximity_km', 100.0),
                'population_density': entities.get('population_density', 100.0)
            })
        
        elif disaster_type == 'flood':
            # Default values for flood if not provided
            location_data.update({
                'basin': entities.get('basin', 'Unknown'),
                'rainfall_mm': entities.get('rainfall_mm', 500.0),
                'river_water_level_m': entities.get('river_water_level_m', 5.0),
                'soil_moisture': entities.get('soil_moisture', 0.5),
                'elevation_m': entities.get('elevation_m', 100.0),
                'historical_flood_events': entities.get('historical_flood_events', 0)
            })
        
        elif disaster_type == 'cyclone':
            location_data.update({
                'wind_speed_kmh': entities.get('wind_speed_kmh', 50.0),
                'pressure_hpa': entities.get('pressure_hpa', 1013.0),
                'distance_from_coast_km': entities.get('distance_from_coast_km', 100.0)
            })
        
        elif disaster_type == 'drought':
            location_data.update({
                'rainfall_deficit_mm': entities.get('rainfall_deficit_mm', 0),
                'soil_moisture': entities.get('soil_moisture', 0.5),
                'reservoir_level_percent': entities.get('reservoir_level_percent', 50.0)
            })
        
        elif disaster_type == 'heatwave':
            location_data.update({
                'temperature_c': entities.get('temperature_c', 35.0),
                'humidity_percent': entities.get('humidity_percent', 50.0)
            })
        
        # Remove None values
        return {k: v for k, v in location_data.items() if v is not None}
    
    def predict(self, disaster_type: str, entities: Dict[str, Any]) -> Optional[DisasterPredictionResult]:
        """
        Make a prediction request to the disaster API.
        
        Args:
            disaster_type: Type of disaster (earthquake, flood, cyclone, drought, heatwave)
            entities: Extracted entities from intent classifier
            
        Returns:
            DisasterPredictionResult or None if request fails
        """
        try:
            # Build request payload
            location_data = self._build_location_data(entities, disaster_type)
            
            payload = {
                'location': location_data,
                'disaster_type': disaster_type
            }
            
            # Make API request
            url = f"{self.base_url}/predict"
            response = requests.post(url, json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                return DisasterPredictionResult(
                    disaster_type=data['disaster_type'],
                    probability=data['probability'],
                    confidence=data['confidence'],
                    risk_level=data['risk_level'],
                    recommendations=data['recommendations'],
                    metadata=data['metadata'],
                    model_version=data['model_version'],
                    prediction_timestamp=data['prediction_timestamp']
                )
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Disaster API request timed out")
            return None
        except requests.exceptions.ConnectionError:
            print("Could not connect to disaster prediction API")
            return None
        except Exception as e:
            print(f"Error calling disaster API: {str(e)}")
            return None
    
    def health_check(self) -> bool:
        """
        Check if the disaster API is healthy.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return False
    
    def get_supported_disasters(self) -> Optional[list]:
        """
        Get list of supported disaster types from API.
        
        Returns:
            List of supported disaster types or None if request fails
        """
        try:
            url = f"{self.base_url}/supported_disasters"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('supported_disasters', [])
            return None
        except Exception as e:
            print(f"Failed to get supported disasters: {str(e)}")
            return None


# Singleton instance
_api_client_instance: Optional[DisasterAPIClient] = None


def get_api_client(base_url: str = "http://localhost:8000") -> DisasterAPIClient:
    """Get the singleton API client instance."""
    global _api_client_instance
    if _api_client_instance is None:
        _api_client_instance = DisasterAPIClient(base_url)
    return _api_client_instance


if __name__ == "__main__":
    # Test the client
    client = DisasterAPIClient()
    
    # Check health
    print("Checking API health...")
    if client.health_check():
        print("API is healthy")
    else:
        print("API is not available")
    
    # Get supported disasters
    print("\nSupported disasters:")
    disasters = client.get_supported_disasters()
    if disasters:
        for disaster in disasters:
            print(f"  - {disaster}")
    
    # Test prediction
    print("\nTesting earthquake prediction for Karnataka...")
    entities = {
        'state': 'Karnataka',
        'district': 'Bangalore',
        'disaster_type': 'earthquake'
    }
    
    result = client.predict('earthquake', entities)
    if result:
        print(f"Probability: {result.probability}%")
        print(f"Risk Level: {result.risk_level}")
        print(f"Recommendations: {result.recommendations[:3]}")
    else:
        print("Prediction failed")
