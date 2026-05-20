import json
import requests
from config import *
from image_output import resolve_response_media

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ Heavy ML packages not found. Running in Cloud/Vercel mode (RAG disabled).")
    ML_AVAILABLE = False

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
    "ambulance", "hospital", "doctor", "medical",
    "police", "theft", "robbery", "kidnap",
    "ndrf", "sdrf", "112", "108", "100", "101", "102", "1078",
    "safety", "preparedness", "kit", "emergency kit",
    "crisis", "calamity", "havoc", "destruction",
]

class QueryEngine:
    def __init__(self):
        print("🧠 Initializing Query Engine...")
        
        self.index = None
        self.texts = []
        self.metadata = []
        self.model = None

        if ML_AVAILABLE:
            # Load sentence transformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Load FAISS index and metadata
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                with open(METADATA_PATH, 'r') as f:
                    data = json.load(f)
                self.texts = data["texts"]
                self.metadata = data["meta"]
                print(f"✅ RAG loaded: {len(self.texts)} documents")
            except Exception as e:
                print(f"❌ Error loading RAG data: {e}")
        else:
            print("⚠️ Skipping FAISS and SentenceTransformer initialization (Vercel Mode)")
        
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
        """Search predefined emergency FAQ first"""
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
            
            if normalized_score > best_score and normalized_score > 0.3:  # At least 30% match
                best_score = normalized_score
                best_match = faq
        
        return best_match
    
    def search_rag_database(self, query_text, top_k=3, confidence_threshold=0.65):
        """Search FAISS index for relevant context."""
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

    def ollama_available(self):
        """Return True if Ollama API is reachable."""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def _rag_fallback_response(self, query_text, context):
        """Use RAG context directly when no AI model is available."""
        snippet = (context or "").strip()
        if len(snippet) > 500:
            snippet = snippet[:497] + "..."
        if snippet:
            return (
                f"Based on emergency guidance: {snippet} "
                "If this is life-threatening, call 112 (India Emergency) or 108 (Ambulance) immediately."
            )
        return (
            "No AI model is available right now. "
            "Describe your emergency with words like fire, bleeding, or choking "
            "so I can match built-in guidance. "
            "Set GROQ_API_KEY for cloud AI, or run: ollama serve"
        )

    def call_groq(self, prompt):
        """Call Groq cloud API (fast inference)."""
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
                    "temperature": 0.3,
                    "max_tokens": 250,
                    "top_p": 0.8
                },
                timeout=15
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

    def call_ollama(self, prompt):
        """Call local Ollama Gemma model (offline fallback)."""
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.8,
                        "num_predict": 120,
                        "num_ctx": 1024
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return None
                
        except Exception as e:
            print(f"Ollama error: {str(e)[:100]}")
            return None

    def call_ai(self, prompt):
        """Try Groq first (fast cloud), then Ollama (local), then return None."""
        # 1. Try Groq (fast cloud)
        if self.groq_available():
            print("🌐 Using Groq API...")
            result = self.call_groq(prompt)
            if result:
                return result
            print("⚠️ Groq failed, trying Ollama fallback...")

        # 2. Try Ollama (local)
        if self.ollama_available():
            print("🖥️ Using Ollama (local)...")
            result = self.call_ollama(prompt)
            if result:
                return result

        # 3. No AI available
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

        faq_match = self.search_emergency_faq(query_text)
        context = ""
        
        if faq_match:
            print("Found in Emergency FAQ")
            context = faq_match["response"]
        else:
            rag_result, similarity = self.search_rag_database(query_text)
            if rag_result:
                print(f"Found in RAG database (similarity: {similarity:.2f})")
                context = rag_result

        # Always try to use AI for a dynamic response first
        prompt = self.create_crisis_prompt(query_text, context)
        answer = self.call_ai(prompt)
        
        if answer:
            return answer, faq_match
            
        # Fallbacks if AI is not available
        if faq_match:
            print("AI unavailable, using offline FAQ answer")
            return faq_match["response"], faq_match
            
        print("AI unavailable, using offline fallback")
        return self._rag_fallback_response(query_text, context), None

    def process_query(self, query_text):
        """
        Main query pipeline.

        Returns a dict:
          - text: spoken/displayed answer
          - images: list of {url, caption, topic} when online and visual guide applies
          - online: internet reachable
          - visual_guide_available: images included in this response
          - offline_text_only: visual guide exists but device is offline
          - urgency: dict with level, label, color, icon, matched_keywords
        """
        urgency = self.analyze_crisis_urgency(query_text)
        text, faq_match = self._build_text_response(query_text)
        result = resolve_response_media(query_text, text, faq_match, urgency["level"])
        result["urgency"] = urgency
        return result
    
    def create_crisis_prompt(self, query_text, context=""):
        """Create optimized prompt for crisis situations, tailored by urgency."""
        urgency = self.analyze_crisis_urgency(query_text)
        level = urgency["level"]

        if level == "critical":
            base_prompt = """You are CRISIS-AI, an emergency assistant for users in India.
This is a CRITICAL LIFE-THREATENING emergency. Respond in 50-80 words with 3-5 numbered steps.

RULES:
- FIRST step MUST be to call 112 or 108 immediately
- Start with the most life-saving action
- Be extremely direct and urgent — every second counts
- Use simple language suitable for panicked people
- No disclaimers — only immediate actionable steps
- Use Indian emergency numbers: 112 (unified), 108 (emergency ambulance)"""
            urgency_text = "\n\n🚨🚨 CRITICAL: LIFE IN IMMEDIATE DANGER! Prioritize calling 112/108 FIRST."

        elif level == "high":
            base_prompt = """You are CRISIS-AI, an emergency assistant for users in India. Respond in 50-100 words with 3-5 numbered steps.

RULES:
- Start with most life-threatening issue first
- Use simple, clear language for audio output
- Give actionable steps only
- No disclaimers or long explanations
- Use Indian emergency numbers: 112 (unified), 100 (police), 101 (fire), 102 (ambulance), 108 (emergency ambulance)
- Reference Indian agencies like NDRF, SDRF where relevant"""
            urgency_text = "\n\n🚨 HIGH URGENCY — Person may be in immediate danger! Mention calling 112 or 108."

        elif level == "medium":
            base_prompt = """You are CRISIS-AI, an emergency assistant for users in India. Respond in 60-120 words with 3-5 numbered steps.

RULES:
- Provide clear safety steps in order of priority
- Include relevant emergency numbers if applicable
- Use simple language suitable for stressed individuals
- Give practical, actionable guidance
- Mention NDRF (1078), SDRF, or local authorities if relevant
- Include preparedness tips if time allows"""
            urgency_text = "\n\n⚠️ URGENT situation — provide clear safety steps and emergency contacts."

        else:  # low
            base_prompt = """You are CRISIS-AI, an emergency assistant for users in India. 

RULES:
- If the query is an emergency or disaster of ANY kind, provide practical safety and preparedness advice in 60-120 words with actionable steps.
- If the user asks a completely unrelated non-emergency question (like jokes, math, general chat), politely say: "I am CRISIS-AI, an emergency and disaster response assistant. Please describe your emergency, and I will provide immediate guidance."
- Use clear, calm language.
- Include relevant emergency numbers for reference if applicable.
- Mention useful resources (NDRF app, local emergency services)"""
            urgency_text = "\n\nℹ️ Provide practical safety guidance, or politely decline if completely unrelated."

        context_text = ""
        if context and context.strip():
            context_text = f"\n\nRELEVANT INFO:\n{context}\n"

        final_prompt = f"""{base_prompt}{urgency_text}{context_text}

USER QUERY: {query_text}

Respond appropriately:"""

        return final_prompt