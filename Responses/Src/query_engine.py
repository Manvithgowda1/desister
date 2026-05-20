import faiss
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import *
from image_output import resolve_response_media

class QueryEngine:
    def __init__(self):
        print("🧠 Initializing Query Engine...")
        
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
        """Search RAG database using vector similarity"""
        if not self.index or not self.texts:
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
        """Analyze urgency level of the crisis"""
        query_lower = query_text.lower()
        
        urgency_level = "low"
        
        # Check for high urgency keywords
        for keyword in HIGH_URGENCY_KEYWORDS:
            if keyword in query_lower:
                urgency_level = "high"
                break
        
        # Check for SOS keywords
        for keyword in SOS_KEYWORDS:
            if keyword in query_lower:
                urgency_level = "high"
                break
        
        return urgency_level

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
        """
        text, faq_match = self._build_text_response(query_text)
        return resolve_response_media(query_text, text, faq_match)
    
    def create_crisis_prompt(self, query_text, context=""):
        """Create optimized prompt for crisis situations"""
        urgency = self.analyze_crisis_urgency(query_text)
        
        base_prompt = """You are CRISIS-AI, an emergency assistant for users in India. Respond in 50-100 words with 3-5 numbered steps.

RULES:
- Start with most life-threatening issue first
- Use simple, clear language for audio output
- Give actionable steps only
- No disclaimers or long explanations
- Use Indian emergency numbers: 112 (unified), 100 (police), 101 (fire), 102 (ambulance), 108 (emergency ambulance), 1078 (disaster helpline / NDRF)
- Reference Indian agencies like NDRF, SDRF, local municipality where relevant"""

        if urgency == "high":
            urgency_text = "\n🚨 HIGH URGENCY - Person may be in immediate danger! Mention calling 112 or 108."
        else:
            urgency_text = "\n⚠️ Provide practical safety steps."
        
        context_text = ""
        if context and context.strip():
            context_text = f"\n\nRELEVANT INFO:\n{context}\n"
        
        final_prompt = f"""{base_prompt}{urgency_text}{context_text}

USER EMERGENCY: {query_text}

Respond with numbered steps:"""
        
        return final_prompt