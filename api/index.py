import os
import sys

# Set up paths to find all project modules
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "Responses", "Src")
FRONTEND = os.path.join(ROOT, "frontend")

if SRC not in sys.path:
    sys.path.insert(0, SRC)
if FRONTEND not in sys.path:
    sys.path.insert(0, FRONTEND)

# Import the Flask app from frontend/app.py
from app import app
