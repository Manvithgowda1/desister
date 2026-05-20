import json
import os
import queue
import sys
import threading
import time

import pyttsx3
import sounddevice as sd
from vosk import KaldiRecognizer, Model

from config import (
    AUDIO_INPUT_DEVICE,
    SAMPLE_RATE,
    SILENCE_END_SECONDS,
    TTS_RATE,
    TTS_VOLUME,
    VOSK_MODEL_PATH,
)


class VoiceHandler:
    def __init__(self):
        print("Initializing Voice Handler...")

        if not os.path.exists(VOSK_MODEL_PATH):
            raise FileNotFoundError(f"Vosk model not found at: {VOSK_MODEL_PATH}")

        self._print_audio_devices()

        print(f"Loading Vosk model from: {VOSK_MODEL_PATH}")
        self.model = Model(VOSK_MODEL_PATH)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(True)

        self.audio_queue = queue.Queue(maxsize=200)
        self.is_listening = False
        self.pause_listening = False

        self._last_partial = ""
        self._last_voice_at = 0.0

        self.is_speaking = False
        self._speech_done = threading.Event()
        self._speech_done.set()
        self._stop_tts = False
        self._tts_lock = threading.Lock()

        self._tts_queue = queue.Queue()
        self._tts_worker = threading.Thread(target=self._tts_worker_loop, daemon=True)
        self._tts_worker.start()

        print("Voice handler ready.")

    def _print_audio_devices(self):
        try:
            default_in, default_out = sd.default.device
            inp = sd.query_devices(default_in)
            print(f"Microphone: [{default_in}] {inp['name']}")
            out = sd.query_devices(default_out)
            print(f"Speaker:    [{default_out}] {out['name']}")
        except Exception as exc:
            print(f"Could not list audio devices: {exc}")

    def _reset_recognizer(self):
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(True)
        self._last_partial = ""

    def _drain_audio_queue(self):
        dropped = 0
        while True:
            try:
                self.audio_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            print(f"Cleared {dropped} buffered audio chunk(s)")

    def _configure_tts_engine(self, engine, rate=None):
        engine.setProperty("rate", rate if rate is not None else TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        if sys.platform == "win32":
            voices = engine.getProperty("voices")
            # Prefer a clear English voice (Zira on many Windows installs)
            for voice in voices:
                name = (voice.name or "").lower()
                if "zira" in name or "english" in name:
                    engine.setProperty("voice", voice.id)
                    break

    def _tts_worker_loop(self):
        """Single TTS thread — required for reliable pyttsx3 on Windows (COM)."""
        com_initialized = False
        if sys.platform == "win32":
            try:
                import pythoncom

                pythoncom.CoInitialize()
                com_initialized = True
            except ImportError:
                pass

        engine = None
        try:
            engine = pyttsx3.init()
            self._configure_tts_engine(engine)
        except Exception as exc:
            print(f"TTS engine init error: {exc}")

        while True:
            item = self._tts_queue.get()
            if item is None:
                break

            text, rate, done_event = item
            if not text or not text.strip():
                done_event.set()
                continue

            self.pause_listening = True
            self._drain_audio_queue()
            self.is_speaking = True
            self._speech_done.clear()
            self._stop_tts = False

            try:
                if engine is None:
                    engine = pyttsx3.init()
                    self._configure_tts_engine(engine)

                self._configure_tts_engine(engine, rate=rate)
                for chunk in self._split_text_into_chunks(text):
                    if self._stop_tts:
                        break
                    engine.say(chunk)
                    engine.runAndWait()
            except Exception as exc:
                print(f"TTS error: {exc}")
                try:
                    engine = pyttsx3.init()
                    self._configure_tts_engine(engine)
                except Exception:
                    engine = None
            finally:
                self.is_speaking = False
                self.pause_listening = False
                self._drain_audio_queue()
                self._reset_recognizer()
                self._speech_done.set()
                done_event.set()

        if com_initialized:
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            except Exception:
                pass

    def speak(self, text, block=False):
        """Queue speech. Set block=True to wait until playback finishes."""
        if not text or not text.strip():
            return

        print(f"Speaking: {text[:120]}{'...' if len(text) > 120 else ''}")
        self.stop_current_speech()
        done = threading.Event()
        self._tts_queue.put((text.strip(), TTS_RATE, done))
        if block:
            done.wait(timeout=300)

    def speak_urgent(self, text):
        """Urgent message — blocks until spoken."""
        if not text or not text.strip():
            return
        print(f"URGENT: {text}")
        self.stop_current_speech()
        done = threading.Event()
        self._tts_queue.put((text.strip(), TTS_RATE + 25, done))
        done.wait(timeout=60)

    def wait_until_idle(self, timeout=300):
        """Wait until TTS has finished."""
        self._speech_done.wait(timeout=timeout)

    def stop_current_speech(self):
        self._stop_tts = True
        if self.is_speaking:
            drained = []
            while not self._tts_queue.empty():
                try:
                    drained.append(self._tts_queue.get_nowait())
                except queue.Empty:
                    break
            for item in drained:
                _, _, evt = item
                evt.set()
        self._speech_done.wait(timeout=2)
        self.is_speaking = False
        self.pause_listening = False

    def _split_text_into_chunks(self, text, max_length=400):
        if len(text) <= max_length:
            return [text]
        chunks = []
        for sentence in text.replace("!", ".").replace("?", ".").split(". "):
            sentence = sentence.strip()
            if not sentence:
                continue
            if not sentence.endswith("."):
                sentence += "."
            if len(sentence) <= max_length:
                chunks.append(sentence)
            else:
                words = sentence.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= max_length:
                        current = f"{current} {word}".strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = word
                if current:
                    chunks.append(current)
        return chunks or [text[:max_length]]

    def _handle_recognized_text(self, text, callback_function):
        text = text.strip()
        if not text:
            return

        print(f"You said: {text}")

        if "stop listening" in text.lower():
            self.speak("Voice assistant paused.", block=True)
            self.is_listening = False
            return

        if self.is_speaking:
            print("Interrupting current response...")
            self.stop_current_speech()

        self._drain_audio_queue()
        self._reset_recognizer()

        # Process in background so microphone keeps capturing
        threading.Thread(
            target=callback_function,
            args=(text,),
            daemon=True,
        ).start()

    def _process_audio_chunk(self, data, callback_function):
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").strip()
            self._last_partial = ""
            if text:
                self._handle_recognized_text(text, callback_function)
            return

        partial = json.loads(self.recognizer.PartialResult()).get("partial", "").strip()
        if partial:
            self._last_partial = partial
            self._last_voice_at = time.time()
            return

        if (
            self._last_partial
            and self._last_voice_at
            and (time.time() - self._last_voice_at) >= SILENCE_END_SECONDS
        ):
            text = self._last_partial
            self._last_partial = ""
            self._last_voice_at = 0.0
            self._reset_recognizer()
            self._handle_recognized_text(text, callback_function)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio input status: {status}")
        if self.pause_listening:
            return
        try:
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.audio_queue.put_nowait(bytes(indata))

    def listen_for_speech(self, callback_function):
        self.is_listening = True
        self._reset_recognizer()
        self._drain_audio_queue()
        print("Listening... (say 'stop listening' to pause)")

        try:
            stream_kwargs = dict(
                samplerate=SAMPLE_RATE,
                blocksize=4000,
                dtype="int16",
                channels=1,
                callback=self.audio_callback,
            )
            if AUDIO_INPUT_DEVICE is not None:
                stream_kwargs["device"] = AUDIO_INPUT_DEVICE

            with sd.RawInputStream(**stream_kwargs):
                while self.is_listening:
                    if self.pause_listening:
                        time.sleep(0.05)
                        continue

                    try:
                        data = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        if (
                            self._last_partial
                            and self._last_voice_at
                            and (time.time() - self._last_voice_at) >= SILENCE_END_SECONDS
                        ):
                            text = self._last_partial
                            self._last_partial = ""
                            self._last_voice_at = 0.0
                            self._reset_recognizer()
                            self._handle_recognized_text(text, callback_function)
                        continue

                    self._process_audio_chunk(data, callback_function)

        except KeyboardInterrupt:
            print("Voice input stopped by user")
        except Exception as exc:
            print(f"Voice input error: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            self.is_listening = False

    def stop_listening(self):
        self.is_listening = False
        self.stop_current_speech()

    def cleanup(self):
        self.is_listening = False
        self.stop_current_speech()
        self._tts_queue.put(None)
        self._drain_audio_queue()
        print("Voice handler cleaned up")

    def get_status(self):
        return {
            "listening": self.is_listening,
            "speaking": self.is_speaking,
            "paused": self.pause_listening,
        }
