"""
disaster_integration.py - Integration module for disaster prediction in voice assistant
Coordinates intent classification, API calls, and response generation for disaster queries.
"""

import os
import sys
from typing import Dict, Any, Optional

# Add src to path for config import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import DISASTER_API_URL, DISASTER_API_ENABLED
except ImportError:
    # Use defaults if config variables not available
    DISASTER_API_URL = os.environ.get("DISASTER_API_URL", "http://localhost:8000")
    DISASTER_API_ENABLED = os.environ.get("DISASTER_API_ENABLED", "true").lower() == "true"

from disaster_intent_classifier import DisasterIntentClassifier, IntentType, get_classifier
from disaster_api_client import DisasterAPIClient, get_api_client, DisasterPredictionResult
from disaster_response_generator import DisasterResponseGenerator, get_response_generator


class DisasterIntegration:
    """Integration module for disaster prediction in voice assistant."""
    
    def __init__(self, api_url: str = None):
        # Use config URL if not provided
        if api_url is None:
            api_url = DISASTER_API_URL
        
        self.classifier = get_classifier()
        self.api_client = get_api_client(api_url)
        self.response_generator = get_response_generator()
        self.enabled = DISASTER_API_ENABLED
        
        # Check if API is available
        if self.enabled:
            self.api_available = self.api_client.health_check()
            if not self.api_available:
                print("Warning: Disaster prediction API is not available")
        else:
            self.api_available = False
            print("Disaster prediction is disabled in configuration")
    
    def process_query(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Process a disaster prediction query.
        
        Args:
            text: User query text
            
        Returns:
            Dictionary with response data or None if not a disaster query
        """
        if not self.enabled or not self.api_available:
            return None
        
        # Classify intent
        intent, disaster_type, entities = self.classifier.classify(text)
        
        if intent != IntentType.DISASTER_PREDICTION:
            return None
        
        print(f"Disaster prediction detected: {disaster_type}")
        print(f"Entities: {entities}")
        
        # Make prediction
        result = self.api_client.predict(disaster_type, entities)
        
        if result is None:
            # API call failed
            location = entities.get('location') or entities.get('state') or entities.get('district')
            error_response = self.response_generator.generate_error_response(
                'api_unavailable',
                location
            )
            return {
                'is_disaster_query': True,
                'success': False,
                'response_text': error_response,
                'disaster_type': disaster_type,
                'error': 'api_unavailable'
            }
        
        # Generate natural language response
        location = entities.get('location') or entities.get('state') or entities.get('district')
        response_text = self.response_generator.generate_response(result, location)
        
        return {
            'is_disaster_query': True,
            'success': True,
            'response_text': response_text,
            'disaster_type': disaster_type,
            'probability': result.probability,
            'risk_level': result.risk_level,
            'confidence': result.confidence,
            'recommendations': result.recommendations,
            'metadata': result.metadata
        }
    
    def is_disaster_query(self, text: str) -> bool:
        """
        Check if a query is a disaster prediction query.
        
        Args:
            text: User query text
            
        Returns:
            True if disaster query, False otherwise
        """
        intent, _, _ = self.classifier.classify(text)
        return intent == IntentType.DISASTER_PREDICTION
    
    def set_api_url(self, api_url: str):
        """Update the API URL."""
        self.api_client = get_api_client(api_url)
        self.api_available = self.api_client.health_check()
    
    def enable(self):
        """Enable disaster prediction."""
        self.enabled = True
    
    def disable(self):
        """Disable disaster prediction."""
        self.enabled = False


# Singleton instance
_integration_instance: Optional[DisasterIntegration] = None


def get_disaster_integration(api_url: str = "http://localhost:8000") -> DisasterIntegration:
    """Get the singleton disaster integration instance."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = DisasterIntegration(api_url)
    return _integration_instance


if __name__ == "__main__":
    # Test the integration
    integration = DisasterIntegration()
    
    test_queries = [
        "What is the earthquake probability in Karnataka?",
        "What's the flood risk in Patna?",
        "Tell me about cyclone warning in Odisha",
        "What's the weather today?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = integration.process_query(query)
        if result:
            print(f"Response: {result['response_text']}")
        else:
            print("Not a disaster query or API unavailable")
