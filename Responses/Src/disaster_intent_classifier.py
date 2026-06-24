"""
disaster_intent_classifier.py - Intent classifier for disaster prediction queries
Detects if a user query is asking for disaster probability/risk prediction.
"""

import re
from typing import Dict, Tuple, Optional
from enum import Enum


class IntentType(Enum):
    """Types of intents detected in user queries."""
    DISASTER_PREDICTION = "disaster_prediction"
    GENERAL_QUERY = "general_query"
    EMERGENCY = "emergency"


class DisasterIntentClassifier:
    """Classifier for detecting disaster prediction intents in user queries."""
    
    def __init__(self):
        # Disaster type keywords
        self.disaster_keywords = {
            'earthquake': ['earthquake', 'seismic', 'tremor', 'quake', 'earth quake', 'seismic activity'],
            'flood': ['flood', 'flooding', 'water level', 'river level', 'inundation', 'flash flood'],
            'cyclone': ['cyclone', 'hurricane', 'storm', 'typhoon', 'wind storm', 'tropical cyclone'],
            'drought': ['drought', 'dry', 'water shortage', 'rainfall deficit', 'water scarcity'],
            'heatwave': ['heatwave', 'heat wave', 'hot weather', 'extreme heat', 'temperature', 'heat stroke']
        }
        
        # Prediction/risk query patterns
        self.prediction_patterns = [
            r'probability',
            r'risk',
            r'chance',
            r'likely',
            r'predict',
            r'forecast',
            r'warning',
            r'alert',
            r'status',
            r'condition'
        ]
        
        # Indian states and union territories
        self.indian_states = [
            'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa',
            'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka', 'kerala',
            'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland',
            'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura',
            'uttar pradesh', 'uttarakhand', 'west bengal', 'delhi', 'jammu and kashmir',
            'ladakh', 'puducherry', 'chandigarh', 'andaman and nicobar', 'lakshadweep'
        ]
        
        # Common Indian cities/districts (sample)
        self.indian_cities = [
            'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'ahmedabad',
            'pune', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore', 'thane', 'bhopal',
            'visakhapatnam', 'pimpri', 'patna', 'vadodara', 'ghaziabad', 'ludhiana',
            'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 'varanasi', 'srinagar',
            'aurangabad', 'dhanbad', 'amritsar', 'navi mumbai', 'allahabad', 'ranchi',
            'howrah', 'coimbatore', 'jabalpur', 'gwalior', 'vijayawada', 'jodhpur',
            'madurai', 'raipur', 'kota', 'guwahati', 'chandigarh', 'solapur', 'hubli'
        ]
    
    def classify(self, text: str) -> Tuple[IntentType, Optional[str], Optional[Dict]]:
        """
        Classify the intent of a user query.
        
        Args:
            text: User query text
            
        Returns:
            Tuple of (intent_type, disaster_type, extracted_entities)
        """
        text_lower = text.lower()
        
        # Check for emergency first
        if self._is_emergency(text_lower):
            return IntentType.EMERGENCY, None, {}
        
        # Check for disaster prediction intent
        disaster_type = self._extract_disaster_type(text_lower)
        
        if disaster_type:
            # Check if it's a prediction/risk query
            if self._is_prediction_query(text_lower):
                entities = self._extract_entities(text_lower, disaster_type)
                return IntentType.DISASTER_PREDICTION, disaster_type, entities
        
        return IntentType.GENERAL_QUERY, None, {}
    
    def _is_emergency(self, text: str) -> bool:
        """Check if the query is an emergency."""
        emergency_keywords = ['sos', 'help me', 'emergency', 'urgent', 'critical', 'mayday',
                           'dying', 'bleeding', 'trapped', 'fire', 'drowning']
        return any(keyword in text for keyword in emergency_keywords)
    
    def _extract_disaster_type(self, text: str) -> Optional[str]:
        """Extract disaster type from text."""
        for disaster, keywords in self.disaster_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return disaster
        return None
    
    def _is_prediction_query(self, text: str) -> bool:
        """Check if the query is asking for prediction/risk."""
        for pattern in self.prediction_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _extract_entities(self, text: str, disaster_type: str) -> Dict:
        """
        Extract entities from the query.
        
        Args:
            text: User query text
            disaster_type: Type of disaster
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            'location': None,
            'state': None,
            'district': None,
            'disaster_type': disaster_type
        }
        
        # Extract state
        for state in self.indian_states:
            if state in text:
                entities['state'] = state.title()
                break
        
        # Extract city/district
        for city in self.indian_cities:
            if city in text:
                entities['district'] = city.title()
                entities['location'] = city.title()
                break
        
        # If no city found but state found, use state as location
        if not entities['location'] and entities['state']:
            entities['location'] = entities['state']
        
        # Extract numeric values (could be temperature, rainfall, etc.)
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            entities['numeric_values'] = [float(n) for n in numbers]
        
        return entities
    
    def get_disaster_type_from_text(self, text: str) -> Optional[str]:
        """Get disaster type from text (helper method)."""
        return self._extract_disaster_type(text.lower())
    
    def get_location_from_text(self, text: str) -> Optional[Dict]:
        """Get location information from text (helper method)."""
        text_lower = text.lower()
        location = {'state': None, 'district': None, 'city': None}
        
        for state in self.indian_states:
            if state in text_lower:
                location['state'] = state.title()
                break
        
        for city in self.indian_cities:
            if city in text_lower:
                location['district'] = city.title()
                location['city'] = city.title()
                break
        
        return location if any(location.values()) else None


# Singleton instance
_classifier_instance: Optional[DisasterIntentClassifier] = None


def get_classifier() -> DisasterIntentClassifier:
    """Get the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DisasterIntentClassifier()
    return _classifier_instance


if __name__ == "__main__":
    # Test the classifier
    classifier = DisasterIntentClassifier()
    
    test_queries = [
        "What is the earthquake probability in Karnataka?",
        "What's the flood risk in Patna?",
        "Tell me about cyclone warning in Odisha",
        "Help me, I'm trapped in a fire",
        "What's the weather today?",
        "Heatwave risk in Rajasthan",
        "Drought probability in Maharashtra"
    ]
    
    for query in test_queries:
        intent, disaster_type, entities = classifier.classify(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {intent.value}")
        print(f"Disaster Type: {disaster_type}")
        print(f"Entities: {entities}")
