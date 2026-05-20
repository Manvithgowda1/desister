#!/usr/bin/env python3
"""Launch Crisis-AI voice assistant from the repository root."""

import os
import sys

# Avoid UnicodeEncodeError on Windows consoles (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "Responses", "Src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.chdir(SRC)

from main_voice_assistant import main  # noqa: E402

if __name__ == "__main__":
    if "--web" in sys.argv or "-w" in sys.argv:
        try:
            from frontend.app import start_server
            start_server()
        except ImportError as e:
            print(f"Error starting web server: {e}")
            print("Did you install Flask? Run: pip install -r requirements.txt")
        sys.exit()

    if "--list-audio" in sys.argv:
        import sounddevice as sd

        print(sd.query_devices())
        print(f"\nDefault input: {sd.default.device[0]}  output: {sd.default.device[1]}")
        print("Set AUDIO_INPUT_DEVICE in Responses/Src/config.py to the input index you want.")
    else:
        main()
