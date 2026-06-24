import json
import requests
import time
from config import *
from image_output import resolve_response_media
from risk_engine import RiskAssessmentEngine


try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ Heavy ML packages not found. Running in Cloud/Vercel mode (RAG disabled).")
    ML_AVAILABLE = False

# Singleton cache for SentenceTransformer model to avoid reloading
_model_cache = None
_index_cache = None
_texts_cache = None
_metadata_cache = None


# Keywords that indicate a disaster/emergency related query
DISASTER_KEYWORDS = [
    "fire", "flood", "earthquake", "tsunami", "cyclone", "hurricane",
    "tornado", "landslide", "avalanche", "storm", "lightning",
    "bleeding", "blood", "wound", "cut", "injury", "fracture", "broken bone",
    "burn", "burned", "burning", "smoke", "explosion",
    "choking", "drowning", "unconscious", "fainted", "collapse",
    "heart attack", "chest pain", "stroke", "seizure", "allergic",
    "poison", "poisoned", "snake bite", "bite", "sting",
    "accident", "crash", "trapped", "stuck", "rescue",
    "cpr", "first aid", "resuscitation", "heimlich",
    "emergency", "sos", "help me", "urgent", "critical", "mayday", "danger",
    "dying", "dead", "severe pain", "can't breathe", "not breathing",
    "head injury", "spinal", "neck injury",
    "disaster", "evacuation", "evacuate", "shelter",
    "gas leak", "chemical", "radiation", "nuclear",
    "war", "attack", "bomb", "shooting", "violence",
    "lost", "missing", "stranded", "survival",
    "food", "water", "hungry", "thirsty", "dehydration",
    "hypothermia", "heatstroke", "heat stroke", "frostbite",
    "pandemic", "epidemic", "outbreak", "infection", "fever",
    "ambulance", "hospital", "doctor", "medical", "clinic", "nurse",
    "police", "theft", "robbery", "kidnap",
    "ndrf", "sdrf", "112", "108", "100", "101", "102", "1078",
    "safety", "preparedness", "kit", "emergency kit",
    "crisis", "calamity", "havoc", "destruction",
    # Additional medical emergency keywords
    "cardiac", "arrest", "palpitations", "irregular heartbeat",
    "diabetes", "insulin", "sugar", "hypoglycemia",
    "asthma", "inhaler", "wheezing", "shortness of breath",
    "allergy", "allergic reaction", "anaphylaxis", "swelling",
    "seizure", "convulsion", "epilepsy",
    "overdose", "drug", "medication",
    "suicide", "depression", "mental health",
    "pregnancy", "labor", "contractions",
    "vomiting", "nausea", "dehydration",
    "fracture", "sprain", "dislocation",
    "concussion", "trauma", "shock"
]

class QueryEngine:
    def __init__(self):
        print("🧠 Initializing Query Engine...")
        
        self.risk_engine = RiskAssessmentEngine()
        
        # Use singleton cache for ML components to avoid reloading
        global _model_cache, _index_cache, _texts_cache, _metadata_cache
        
        if ML_AVAILABLE:
            # Load sentence transformer from cache or create new
            if _model_cache is None:
                print("Loading SentenceTransformer model (cached for future instances)...")
                _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
            self.model = _model_cache
            
            # Load FAISS index and metadata from cache or create new
            if _index_cache is None:
                try:
                    _index_cache = faiss.read_index(FAISS_INDEX_PATH)
                    with open(METADATA_PATH, 'r') as f:
                        data = json.load(f)
                    _texts_cache = data["texts"]
                    _metadata_cache = data["meta"]
                    print(f"✅ RAG loaded: {len(_texts_cache)} documents")
                except Exception as e:
                    print(f"❌ Error loading RAG data: {e}")
                    _index_cache = None
                    _texts_cache = []
                    _metadata_cache = []
            
            self.index = _index_cache
            self.texts = _texts_cache
            self.metadata = _metadata_cache
        else:
            print("⚠️ Skipping FAISS and SentenceTransformer initialization (Vercel Mode)")
            self.model = None
            self.index = None
            self.texts = []
            self.metadata = []
        
        # Load emergency FAQ
        try:
            with open(FAQ_PATH, 'r') as f:
                faq_data = json.load(f)
            self.emergency_faqs = faq_data["faqs"]
            print(f"✅ Emergency FAQ loaded: {len(self.emergency_faqs)} entries")
        except Exception as e:
            print(f"⚠️ Could not load emergency FAQ: {e}")
            self.emergency_faqs = []
    
    def search_emergency_faq(self, query_text):
        """Search predefined emergency FAQ - only for very specific matches"""
        query_lower = query_text.lower()
        
        best_match = None
        best_score = 0
        
        for faq in self.emergency_faqs:
            score = 0
            for keyword in faq["keywords"]:
                if keyword.lower() in query_lower:
                    score += 1
            
            # Normalize score by number of keywords
            normalized_score = score / len(faq["keywords"]) if faq["keywords"] else 0
            
            # Increased threshold to 0.7 for more specific matches only
            if normalized_score > best_score and normalized_score > 0.7:
                best_score = normalized_score
                best_match = faq
        
        return best_match
    
    def search_rag_database(self, query_text, top_k=3, confidence_threshold=0.65):
        """Search FAISS index for relevant context - no caching for dynamic responses."""
        if not ML_AVAILABLE or not self.index or len(self.texts) == 0 or self.model is None:
            return None, 0.0
        
        try:
            query_vec = self.model.encode([query_text])
            D, I = self.index.search(np.array(query_vec), top_k)
            
            if I[0][0] == -1:  # No results
                return None, 0.0
            
            # Get best match and calculate cosine similarity
            best_idx = I[0][0]
            best_text = self.texts[best_idx]
            
            text_vec = self.model.encode([best_text])
            similarity = cosine_similarity(query_vec, text_vec)[0][0]
            
            if similarity > confidence_threshold:
                return best_text, similarity
            
            return None, similarity
            
        except Exception as e:
            print(f"❌ RAG search error: {e}")
            return None, 0.0
    
    def groq_available(self):
        """Return True if Groq API key is configured."""
        return bool(GROQ_API_KEY)



    def get_specific_helpline(self, query_text):
        """Get specific helpline based on emergency type detected in query."""
        query_lower = query_text.lower()
        
        # Check for specific emergency types
        emergency_keywords = {
            "fire": ["fire", "burn", "burning", "smoke", "explosion"],
            "medical": ["medical", "doctor", "hospital", "clinic", "nurse", "injury", "fracture", "broken bone", "wound", "cut", "pain", "sick", "illness"],
            "bleeding": ["bleeding", "blood", "hemorrhage"],
            "choking": ["choking", "suffocate"],
            "drowning": ["drowning", "underwater"],
            "heart attack": ["heart attack", "cardiac", "chest pain", "palpitations", "arrest"],
            "stroke": ["stroke", "paralysis"],
            "poison": ["poison", "poisoned", "toxic", "overdose", "drug"],
            "accident": ["accident", "crash", "collision", "hit"],
            "earthquake": ["earthquake", "tremor"],
            "flood": ["flood", "flooding", "water logging"],
            "cyclone": ["cyclone", "storm", "hurricane", "tornado"],
            "police": ["police", "theft", "robbery", "kidnap", "assault", "attack"],
            "asthma": ["asthma", "inhaler", "wheezing", "shortness of breath"],
            "allergy": ["allergy", "allergic reaction", "anaphylaxis", "swelling"],
            "seizure": ["seizure", "convulsion", "epilepsy"],
            "diabetes": ["diabetes", "insulin", "sugar", "hypoglycemia"],
            "mental": ["suicide", "depression", "mental health"],
            "pregnancy": ["pregnancy", "labor", "contractions"]
        }
        
        for emergency_type, keywords in emergency_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return EMERGENCY_HELPLINES.get(emergency_type, EMERGENCY_HELPLINES["general"])
        
        # If no specific emergency detected, no helpline added
        return None

    def call_groq(self, prompt):
        """Call Groq cloud API (fast inference) with optimized timeout and dynamic temperature."""
        import hashlib
        
        # Vary temperature based on prompt hash to prevent repetitive responses
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        temperature = 0.3 + (prompt_hash % 5) * 0.1  # Varies between 0.3 and 0.7
        
        try:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY.strip()}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are CRISIS-AI, an emergency assistant for users in India. Give short, actionable responses."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": 150,  # Reduced from 200 for faster response
                    "top_p": 0.8
                },
                timeout=8  # Reduced from 10 for faster response
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = response.json().get("error", {}).get("message", f"HTTP {response.status_code}")
                print(f"Groq API error: {error_msg}")
                return None  # Signal to try fallback
                
        except requests.exceptions.ReadTimeout:
            print("Groq API timeout")
            return None
        except Exception as e:
            print(f"Groq error: {str(e)[:100]}")
            return None

    def call_ai(self, prompt):
        """Call Groq API (fast cloud). Returns None if offline or failed."""
        from image_output import is_online
        if self.groq_available() and is_online():
            print("🌐 Using Groq API...")
            return self.call_groq(prompt)
        print("⚠️ Groq API unavailable or offline")
        return None

    def analyze_crisis_urgency(self, query_text):
        """Analyze urgency level of the crisis into 4 tiers.
        
        Returns a dict with:
          - level: 'critical', 'high', 'medium', or 'low'
          - label: human-readable label
          - color: CSS color for UI display
          - icon: emoji icon
          - matched_keywords: list of matched keywords
        """
        query_lower = query_text.lower()
        matched = []

        # Critical: immediate life-threatening situations
        CRITICAL_KEYWORDS = [
            "dying", "not breathing", "can't breathe", "heart attack",
            "drowning", "unconscious", "choking", "severe bleeding",
            "stroke", "cardiac arrest", "anaphylaxis", "suicide",
            "shooting", "bomb", "stabbed", "electrocuted",
        ]
        for kw in CRITICAL_KEYWORDS:
            if kw in query_lower:
                matched.append(kw)
        if matched:
            return {
                "level": "critical",
                "label": "CRITICAL — Life Threatening",
                "color": "#dc2626",
                "icon": "\U0001f6a8",
                "matched_keywords": matched,
            }

        # High: serious emergency needing immediate action
        for kw in HIGH_URGENCY_KEYWORDS:
            if kw in query_lower:
                matched.append(kw)
        for kw in SOS_KEYWORDS:
            if kw in query_lower:
                matched.append(kw)
        if matched:
            return {
                "level": "high",
                "label": "HIGH — Immediate Action Needed",
                "color": "#ea580c",
                "icon": "\u26a0\ufe0f",
                "matched_keywords": matched,
            }

        # Medium: serious but not immediately life-threatening
        MEDIUM_KEYWORDS = [
            "accident", "crash", "flood", "earthquake", "cyclone",
            "storm", "landslide", "tsunami", "hurricane", "tornado",
            "fracture", "broken bone", "burn", "injury", "wound",
            "evacuation", "evacuate", "gas leak", "chemical",
            "missing", "lost", "stranded", "pandemic", "outbreak",
            "shelter", "disaster", "avalanche", "explosion",
        ]
        for kw in MEDIUM_KEYWORDS:
            if kw in query_lower:
                matched.append(kw)
        if matched:
            return {
                "level": "medium",
                "label": "MEDIUM — Urgent Situation",
                "color": "#ca8a04",
                "icon": "\U0001f7e1",
                "matched_keywords": matched,
            }

        # Low: general safety / preparedness query
        return {
            "level": "low",
            "label": "LOW — Safety Guidance",
            "color": "#2563eb",
            "icon": "\u2139\ufe0f",
            "matched_keywords": [],
        }

    def is_disaster_related(self, query_text):
        """Check if the query is related to a disaster, emergency, or safety topic."""
        query_lower = query_text.lower()
        for keyword in DISASTER_KEYWORDS:
            if keyword in query_lower:
                return True
        # Also check if FAQ or RAG has a match (those are curated emergency content)
        if self.search_emergency_faq(query_text):
            return True
        _, similarity = self.search_rag_database(query_text)
        if similarity > 0.5:
            return True
        return False

    def _build_text_response(self, query_text):
        """Run FAQ / RAG / AI pipeline and return answer text + optional FAQ match."""
        print(f"Processing: {query_text}")

        # Get RAG context for dynamic AI generation
        rag_result, similarity = self.search_rag_database(query_text)
        context = ""
        if rag_result:
            print(f"Found in RAG database (similarity: {similarity:.2f})")
            context = rag_result

        # Always use AI for dynamic response - FAQ only as strict fallback
        prompt = self.create_crisis_prompt(query_text, context)
        answer = self.call_ai(prompt)
        
        if answer:
            # Add specific helpline based on emergency type
            specific_helpline = self.get_specific_helpline(query_text)
            if specific_helpline:
                answer = f"{answer}\n\n{specific_helpline}"
            return answer, None
            
        # Fallback: try FAQ only if AI completely fails
        faq_match = self.search_emergency_faq(query_text)
        if faq_match:
            print("AI unavailable, using offline FAQ answer")
            return faq_match["response"], faq_match
            
        print("AI unavailable, using offline fallback")
        return self._rag_fallback_response(query_text, context), None

    def _rag_fallback_response(self, query_text, context):
        """Use RAG context directly when no AI model is available."""
        snippet = (context or "").strip()
        if len(snippet) > 500:
            snippet = snippet[:497] + "..."
        
        # Add specific helpline based on emergency type
        specific_helpline = self.get_specific_helpline(query_text)
        helpline_suffix = f"\n\n{specific_helpline}" if specific_helpline else ""
        
        # Analyze urgency for better fallback response
        urgency = self.analyze_crisis_urgency(query_text)
        level = urgency["level"]
        
        if snippet:
            # Provide structured response based on urgency
            if level in ["critical", "high"]:
                return (
                    f"⚠️ {level.upper()} EMERGENCY:\n\n{snippet}\n\n"
                    f"IMMEDIATE ACTION REQUIRED. Follow the guidance above."
                    f"{helpline_suffix}"
                )
            else:
                return (
                    f"Emergency Guidance:\n\n{snippet}"
                    f"{helpline_suffix}"
                )
        
        # No RAG context - provide basic guidance based on urgency
        if level == "critical":
            return (
                "🚨 CRITICAL EMERGENCY:\n\n"
                "1. Call 112 or 108 immediately\n"
                "2. Describe your emergency with specific keywords (fire, bleeding, choking, etc.)\n"
                "3. Stay on the line with emergency services\n\n"
                "Note: AI model unavailable. Configure GROQ_API_KEY for detailed guidance."
                f"{helpline_suffix}"
            )
        elif level == "high":
            return (
                "⚠️ HIGH URGENCY:\n\n"
                "1. Call 112 or 108 if in danger\n"
                "2. Describe your emergency with specific keywords for better guidance\n"
                "3. Stay calm and follow basic safety protocols\n\n"
                "Note: AI model unavailable. Configure GROQ_API_KEY for detailed guidance."
                f"{helpline_suffix}"
            )
        else:
            return (
                "Emergency Guidance:\n\n"
                "Please describe your emergency with specific keywords like:\n"
                "- fire, burning, smoke\n"
                "- bleeding, blood, injury\n"
                "- choking, can't breathe\n"
                "- heart attack, chest pain\n"
                "- earthquake, flood, cyclone\n\n"
                "This will help provide relevant guidance. "
                "Configure GROQ_API_KEY for AI-powered detailed responses."
                f"{helpline_suffix}"
            )

    def detect_risk_query(self, query_text):
        """Check if the query is a disaster risk prediction request."""
        query_lower = query_text.lower()
        risk_keywords = ["probability", "probabality", "risk", "prone", "likely", "chance", "hazard", "threat", "possibility", "forecast", "prediction", "predict", "score"]
        
        has_risk_keyword = any(kw in query_lower for kw in risk_keywords)
        
        disaster_type = None
        for d in ["earthquake", "flood", "cyclone", "landslide", "drought", "heatwave"]:
            if d in query_lower:
                disaster_type = d
                break
                
        is_most_likely_query = "most likely" in query_lower or "which disaster" in query_lower or "what disaster" in query_lower
        
        location = None
        for loc in ["bengaluru", "bangalore", "mysuru", "mysore", "kodagu", "coorg", "wayanad", "alappuzha", "alleppey", "chennai", "nilgiris", "ooty", "karnataka", "kerala", "tamil nadu"]:
            if loc in query_lower:
                if loc == "bangalore": location = "bengaluru"
                elif loc == "mysore": location = "mysuru"
                elif loc == "coorg": location = "kodagu"
                elif loc == "alleppey": location = "alappuzha"
                elif loc == "ooty": location = "nilgiris"
                else: location = loc
                break
                
        if not location:
            for indicator in ["district", "state", "my location", "here", "karnataka", "kerala", "tamil nadu"]:
                if indicator in query_lower:
                    location = "karnataka"  # Default fallback state
                    break
                    
        if has_risk_keyword or is_most_likely_query:
            if location or disaster_type:
                return True, location or "karnataka", disaster_type, is_most_likely_query
                
        return False, None, None, False

    def process_query(self, query_text):
        """
        Main query pipeline.
        """
        # 1. Detect if this is a Disaster Risk Prediction query
        is_risk, location, disaster_type, is_most_likely = self.detect_risk_query(query_text)
        
        if is_risk:
            print(f"Risk prediction query detected. Location: {location}, Disaster: {disaster_type}, Most Likely: {is_most_likely}")
            risk_data = self.risk_engine.calculate_risk(location, disaster_type)
            
            # Sort or select data if "most likely" query
            if is_most_likely:
                risk_data = sorted(risk_data, key=lambda x: x["score"], reverse=True)
            
            # Use ML model data directly - no Groq API for risk predictions
            if is_most_likely or len(risk_data) > 1:
                r = risk_data[0] # The highest risk one
                answer = f"The most likely disaster risk in {r['location']} is {r['disaster']} at a {r['level']} level (Score: {r['score']}/100). {r['reason']} {r['details']} Please remain prepared."
            else:
                r = risk_data[0]
                answer = f"The risk of {r['disaster']} in {r['location']} is currently {r['level']} (Score: {r['score']}/100). {r['reason']} {r['details']} Precautions are advised."

            # Append specific helpline based on disaster type
            specific_helpline = self.get_specific_helpline(query_text)
            if specific_helpline:
                answer = f"{answer}\n\n{specific_helpline}"
            else:
                answer = f"{answer}\n\n{EMERGENCY_HELPLINES['general']}"

            urgency = {
                "level": "medium",
                "label": "MEDIUM — Risk Prediction Info",
                "color": "#ca8a04",
                "icon": "🔮",
                "matched_keywords": ["risk"]
            }
            
            # Resolve media/images if any
            result = resolve_response_media(query_text, answer, None, urgency["level"])
            result["urgency"] = urgency
            result["risk_prediction"] = risk_data
            return result


        # Default standard crisis processing
        urgency = self.analyze_crisis_urgency(query_text)
        text, faq_match = self._build_text_response(query_text)
        result = resolve_response_media(query_text, text, faq_match, urgency["level"])
        result["urgency"] = urgency
        return result
    
    def create_crisis_prompt(self, query_text, context=""):
        """Create optimized prompt for crisis situations, tailored by urgency and specific query details."""
        urgency = self.analyze_crisis_urgency(query_text)
        level = urgency["level"]
        
        # Extract specific details from query for more contextual responses
        query_lower = query_text.lower()
        specific_context = ""
        
        # Identify specific emergency types for targeted responses
        emergency_types = {
            "fire": "fire emergency",
            "flood": "flooding situation",
            "earthquake": "earthquake",
            "cyclone": "cyclone/storm",
            "bleeding": "bleeding injury",
            "choking": "choking emergency",
            "drowning": "drowning situation",
            "heart attack": "cardiac emergency",
            "stroke": "stroke emergency",
            "burn": "burn injury",
            "accident": "accident",
            "poison": "poisoning",
            "asthma": "asthma attack",
            "seizure": "seizure"
        }
        
        detected_emergency = None
        for keyword, emergency_name in emergency_types.items():
            if keyword in query_lower:
                detected_emergency = emergency_name
                break
        
        if detected_emergency:
            specific_context = f"\nSPECIFIC SITUATION: {detected_emergency}"
        
        # Add unique instruction variations based on query hash to prevent repetitive responses
        import hashlib
        query_hash = int(hashlib.md5(query_text.encode()).hexdigest()[:8], 16) % 3
        
        instruction_variations = [
            "Focus on the most critical action first.",
            "Prioritize immediate safety above all else.",
            "Address the most urgent need immediately."
        ]
        
        variation_instruction = instruction_variations[query_hash]

        if level == "critical":
            base_prompt = f"""You are CRISIS-AI, an emergency assistant for users in India.
This is a CRITICAL LIFE-THREATENING emergency. Respond in 40-60 words with 3-5 numbered steps.

RULES:
- FIRST step MUST be to call 112 or 108 immediately
- {variation_instruction}
- Be extremely direct and urgent — every second counts
- Use simple language suitable for panicked people
- No disclaimers — only immediate actionable steps
- Tailor response specifically to the user's situation{specific_context}
- Vary your response structure and wording for this specific query"""
            urgency_text = "\n\n🚨🚨 CRITICAL: LIFE IN IMMEDIATE DANGER! Prioritize calling 112/108 FIRST."

        elif level == "high":
            base_prompt = f"""You are CRISIS-AI, an emergency assistant for users in India. Respond in 40-80 words with 3-5 numbered steps.

RULES:
- {variation_instruction}
- Use simple, clear language for audio output
- Give actionable steps only
- No disclaimers or long explanations
- Include relevant Indian emergency numbers: 112 (unified), 100 (police), 101 (fire), 108 (emergency ambulance)
- Tailor response specifically to the user's situation{specific_context}
- Vary your response structure and wording for this specific query"""
            urgency_text = "\n\n🚨 HIGH URGENCY — Person may be in immediate danger! Mention calling 112 or 108."

        elif level == "medium":
            base_prompt = f"""You are CRISIS-AI, an emergency assistant for users in India. Respond in 50-100 words with 3-5 numbered steps.

RULES:
- {variation_instruction}
- Provide clear safety steps in order of priority
- Use simple language suitable for stressed individuals
- Give practical, actionable guidance
- Mention NDRF (1078) if relevant
- Tailor response specifically to the user's situation{specific_context}
- Vary your response structure and wording for this specific query"""
            urgency_text = "\n\n⚠️ URGENT situation — provide clear safety steps."

        else:  # low
            base_prompt = f"""You are CRISIS-AI, an emergency assistant for users in India. 

RULES:
- If the query is an emergency or disaster, provide practical safety advice in 50-100 words with actionable steps.
- If the user asks a completely unrelated non-emergency question, politely decline.
- Use clear, calm language.
- Tailor response specifically to the user's situation{specific_context}
- Vary your response structure and wording for this specific query"""
            urgency_text = "\n\nℹ️ Provide practical safety guidance."

        context_text = ""
        if context and context.strip():
            context_text = f"\n\nRELEVANT INFO:\n{context}\n"

        final_prompt = f"""{base_prompt}{urgency_text}{context_text}

USER QUERY: {query_text}

Provide a unique, varied response:"""

        return final_prompt