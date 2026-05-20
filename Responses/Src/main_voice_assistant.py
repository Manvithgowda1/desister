"""
Voice Assistant
Main application that coordinates all components
"""

import os
import sys
import threading
import time
import queue  # FIXED: Added missing import
import json
import sounddevice as sd

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice_handler import VoiceHandler
from query_engine import QueryEngine
from emergency_detector import EmergencyDetector
from response_display import present_result
from config import *

class CrisisVoiceAssistant:
    def __init__(self):
        print("🚁 Initializing Crisis Response Voice Assistant...")
        print("=" * 60)
        
        # Initialize components
        try:
            self.voice_handler = VoiceHandler()
            self.query_engine = QueryEngine()
            self.emergency_detector = EmergencyDetector()
            
            self.is_running = False
            self.processing_lock = threading.Lock()
            
            print("✅ All components initialized successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            sys.exit(1)
    
    def process_voice_input(self, text):
        """Process voice input and generate response"""
        # Prevent overlapping processing
        if not self.processing_lock.acquire(blocking=False):
            print("⏳ Still processing previous request...")
            return
        
        try:
            # Check for emergency/SOS first
            is_emergency, keyword = self.emergency_detector.detect_sos_in_text(text)
            
            if is_emergency:
                print(f"EMERGENCY DETECTED: {keyword}")

                emergency_ack = f"Emergency detected: {keyword}. Getting help now."
                self.voice_handler.speak_urgent(emergency_ack)

                def background_sos():
                    try:
                        self.emergency_detector.handle_emergency(text, keyword)
                    except Exception as e:
                        print(f"SOS trigger error: {e}")

                threading.Thread(target=background_sos, daemon=True).start()

                print("Getting emergency guidance...")
                result = self.query_engine.process_query(text)
                present_result(result, open_images=True)
                cleaned_response = self._clean_response_for_tts(result["text"])
                self.voice_handler.speak(cleaned_response, block=True)

            else:
                result = self.query_engine.process_query(text)
                present_result(result, open_images=True)
                cleaned_response = self._clean_response_for_tts(result["text"])
                self.voice_handler.speak(cleaned_response, block=True)
                
        except Exception as e:
            error_msg = "I encountered an error. Please try again."
            print(f"Processing error: {e}")
            self.voice_handler.speak(error_msg, block=True)
        
        finally:
            self.processing_lock.release()
    
    def _clean_response_for_tts(self, response):
        """Clean AI response for better TTS"""
        if not response:
            return "I couldn't generate a response. Please try again."
        
        # Remove excessive formatting
        cleaned = response.replace('**', '').replace('*', '')
        cleaned = cleaned.replace('⚠️', 'Warning:').replace('🚨', 'Emergency:')
        cleaned = cleaned.replace('✅', 'Step:').replace('❌', 'Error:')
        
        # Remove excessive line breaks
        cleaned = ' '.join(cleaned.split())
        
        # Limit length for TTS
        # if len(cleaned) > 400:
        #     sentences = cleaned.split('. ')
        #     result = ""
        #     for sentence in sentences:
        #         if len(result + sentence) < 350:
        #             result += sentence + ". "
        #         else:
        #             break
        #     cleaned = result + "Ask for more details if needed."
        
        return cleaned.strip()
    
    def start(self):
        """Start the voice assistant"""
        self.is_running = True
        
        print("🎤 Crisis Voice Assistant is now active!")
        print("📢 Available commands:")
        print("   - Say anything for help")
        print("   - Say 'stop listening' to pause")
        print("   - Say 'start listening' to resume")
        print("   - Press Ctrl+C to exit")
        print("-" * 60)
        
        try:
            # Start voice listening loop
            while self.is_running:
                try:
                    self.voice_handler.listen_for_speech(self.process_voice_input)
                    
                    # If listening stopped, wait for restart command
                    if self.is_running:
                        print("🔄 Say 'start listening' to resume, or Ctrl+C to exit")
                        self.wait_for_restart()
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Voice loop error: {e}")
                    time.sleep(2)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def wait_for_restart(self):
        """Wait for restart command (reuses voice handler partial/silence logic)."""
        print("Say 'start listening' to resume (30 second timeout)...")

        restart_event = threading.Event()

        def on_restart(text):
            lower = text.lower()
            if "start listening" in lower:
                print("Resuming voice assistant...")
                restart_event.set()
                self.voice_handler.is_listening = False
            elif "exit" in lower or "quit" in lower:
                self.is_running = False
                restart_event.set()
                self.voice_handler.is_listening = False

        def listen_thread():
            self.voice_handler.listen_for_speech(on_restart)

        t = threading.Thread(target=listen_thread, daemon=True)
        t.start()
        if not restart_event.wait(timeout=30):
            self.voice_handler.is_listening = False
            if self.is_running:
                print("Timeout — auto-resuming...")
        t.join(timeout=2)
    
    def stop(self):
        """Stop the voice assistant"""
        self.is_running = False
        
        # Clean up components
        try:
            self.voice_handler.cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        print("\n🛑 Crisis Voice Assistant stopped.")
        print("Stay safe! 🚁")

def run_text_mode():
    """Text-only mode for testing without a microphone."""
    print("CRISIS-AI text mode (FAQ + RAG; Ollama optional)")
    print("Type a question or 'quit' to exit.")
    print("-" * 60)

    from query_engine import QueryEngine
    from emergency_detector import EmergencyDetector

    engine = QueryEngine()
    detector = EmergencyDetector()

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            break

        is_emergency, keyword = detector.detect_sos_in_text(text)
        if is_emergency:
            print(f"[Emergency detected: {keyword}]")
            detector.handle_emergency(text, keyword)

        result = engine.process_query(text)
        present_result(result, open_images=True)


def main():
    """Main entry point"""
    print("CRISIS RESPONSE VOICE ASSISTANT")
    print("===================================")

    if "--text" in sys.argv or "-t" in sys.argv:
        run_text_mode()
        return

    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("Ollama is running (full AI responses enabled)")
        else:
            print("Ollama may not be configured — FAQ and RAG still work")
    except Exception:
        print("Ollama not running — FAQ and RAG still work")
        print("  For AI answers: ollama serve  &&  ollama pull gemma3n:latest")
        print("  Or use text mode: python run.py --text")

    assistant = CrisisVoiceAssistant()

    try:
        assistant.start()
    except Exception as e:
        print(f"Assistant crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Goodbye!")

if __name__ == "__main__":
    main()