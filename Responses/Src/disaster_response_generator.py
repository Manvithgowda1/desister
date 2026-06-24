"""
disaster_response_generator.py - Natural language response generator for disaster predictions
Converts disaster prediction results into human-readable responses for voice output.
"""

from typing import Dict, List, Any
from disaster_api_client import DisasterPredictionResult


class DisasterResponseGenerator:
    """Generates natural language responses from disaster prediction results."""
    
    def __init__(self):
        # Risk level descriptions
        self.risk_descriptions = {
            'low': 'low risk',
            'moderate': 'moderate risk',
            'high': 'high risk',
            'very_high': 'very high risk',
            'extreme': 'extreme risk'
        }
        
        # Disaster type names
        self.disaster_names = {
            'earthquake': 'earthquake',
            'flood': 'flood',
            'cyclone': 'cyclone',
            'drought': 'drought',
            'heatwave': 'heatwave'
        }
    
    def generate_response(self, result: DisasterPredictionResult, location: str = None) -> str:
        """
        Generate a natural language response from prediction result.
        
        Args:
            result: Disaster prediction result
            location: Location name (optional)
            
        Returns:
            Natural language response string
        """
        location_str = location or result.metadata.get('district') or result.metadata.get('state') or 'this area'
        disaster_name = self.disaster_names.get(result.disaster_type, result.disaster_type)
        risk_desc = self.risk_descriptions.get(result.risk_level, result.risk_level)
        
        # Build response based on risk level
        if result.risk_level in ['low', 'moderate']:
            response = self._generate_low_risk_response(disaster_name, location_str, result)
        elif result.risk_level == 'high':
            response = self._generate_high_risk_response(disaster_name, location_str, result)
        else:  # very_high or extreme
            response = self._generate_critical_risk_response(disaster_name, location_str, result)
        
        return response
    
    def _generate_low_risk_response(self, disaster: str, location: str, result: DisasterPredictionResult) -> str:
        """Generate response for low/moderate risk."""
        response = f"The {disaster} probability in {location} is {result.probability:.1f} percent, "
        response += f"which indicates {self.risk_descriptions[result.risk_level]}. "
        
        if result.confidence > 70:
            response += "This prediction has high confidence. "
        else:
            response += "This prediction has moderate confidence. "
        
        # Add top recommendations
        if result.recommendations:
            top_recommendations = result.recommendations[:2]
            response += "Here are some recommendations: " + ". ".join(top_recommendations) + "."
        
        return response
    
    def _generate_high_risk_response(self, disaster: str, location: str, result: DisasterPredictionResult) -> str:
        """Generate response for high risk."""
        response = f"The {disaster} probability in {location} is {result.probability:.1f} percent, "
        response += f"which indicates {self.risk_descriptions[result.risk_level]}. "
        
        if result.confidence > 70:
            response += "This prediction has high confidence. "
        else:
            response += "This prediction has moderate confidence. "
        
        response += "You should take precautions. "
        
        # Add top recommendations
        if result.recommendations:
            top_recommendations = result.recommendations[:3]
            response += "Key recommendations include: " + ". ".join(top_recommendations) + "."
        
        return response
    
    def _generate_critical_risk_response(self, disaster: str, location: str, result: DisasterPredictionResult) -> str:
        """Generate response for very high/extreme risk."""
        response = f"Warning. The {disaster} probability in {location} is {result.probability:.1f} percent, "
        response += f"which indicates {self.risk_descriptions[result.risk_level]}. "
        
        if result.confidence > 70:
            response += "This prediction has high confidence. "
        
        response += "Please take immediate action. "
        
        # Add urgent recommendations
        if result.recommendations:
            urgent_recommendations = [r for r in result.recommendations if any(
                keyword in r.lower() for keyword in ['urgent', 'immediate', 'critical', 'evacuate']
            )]
            
            if urgent_recommendations:
                response += "Urgent recommendations: " + ". ".join(urgent_recommendations[:2]) + "."
            else:
                response += "Key recommendations: " + ". ".join(result.recommendations[:3]) + "."
        
        return response
    
    def generate_error_response(self, error_type: str, location: str = None) -> str:
        """
        Generate error response when prediction fails.
        
        Args:
            error_type: Type of error (api_unavailable, invalid_location, etc.)
            location: Location name (optional)
            
        Returns:
            Error response string
        """
        location_str = location or 'this area'
        
        if error_type == 'api_unavailable':
            return f"I'm sorry, but the disaster prediction service is currently unavailable. Please try again later."
        elif error_type == 'invalid_location':
            return f"I couldn't find the location {location_str}. Please specify a valid Indian state or district."
        elif error_type == 'invalid_disaster':
            return f"I'm sorry, but that disaster type is not currently supported."
        else:
            return f"I encountered an error while processing your request. Please try again."
    
    def generate_recommendations_summary(self, recommendations: List[str], max_count: int = 3) -> str:
        """
        Generate a summary of recommendations.
        
        Args:
            recommendations: List of recommendation strings
            max_count: Maximum number of recommendations to include
            
        Returns:
            Summary string
        """
        if not recommendations:
            return "No specific recommendations available."
        
        selected = recommendations[:max_count]
        return "Recommendations: " + ". ".join(selected) + "."
    
    def format_probability(self, probability: float) -> str:
        """Format probability for speech."""
        if probability < 10:
            return f"{probability:.1f} percent"
        elif probability < 100:
            return f"{probability:.0f} percent"
        else:
            return "100 percent"


# Singleton instance
_generator_instance: DisasterResponseGenerator = None


def get_response_generator() -> DisasterResponseGenerator:
    """Get the singleton response generator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = DisasterResponseGenerator()
    return _generator_instance


if __name__ == "__main__":
    # Test the generator
    generator = DisasterResponseGenerator()
    
    # Test low risk
    low_risk_result = DisasterPredictionResult(
        disaster_type='earthquake',
        probability=15.5,
        confidence=85.0,
        risk_level='low',
        recommendations=['Standard building codes are sufficient', 'Regular structural inspections recommended'],
        metadata={'district': 'Bangalore', 'state': 'Karnataka'},
        model_version='1.0.0',
        prediction_timestamp='2024-01-15T10:30:00'
    )
    
    print("Low risk response:")
    print(generator.generate_response(low_risk_result, "Bangalore"))
    print()
    
    # Test high risk
    high_risk_result = DisasterPredictionResult(
        disaster_type='flood',
        probability=78.5,
        confidence=90.0,
        risk_level='very_high',
        recommendations=['URGENT: Prepare for imminent flooding', 'Implement flood barriers immediately', 'Prepare for immediate evacuation'],
        metadata={'district': 'Patna', 'state': 'Bihar'},
        model_version='1.0.0',
        prediction_timestamp='2024-01-15T10:30:00'
    )
    
    print("High risk response:")
    print(generator.generate_response(high_risk_result, "Patna"))
