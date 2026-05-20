#!/usr/bin/env python3
"""One-time setup: dependencies, Vosk speech model, and FAISS RAG index."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
VOSK_DIR = ROOT / "Voice_Assistant" / "Models" / "vosk-model-small-en-us-0.15"
VOSK_ZIP_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
FAISS_PATH = ROOT / "Voice_Assistant" / "Data" / "rag_index.faiss"
METADATA_PATH = ROOT / "Voice_Assistant" / "Data" / "rag_metadata.json"
REQUIREMENTS = ROOT / "requirements.txt"


def run_pip_install() -> None:
    print("Installing Python dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        cwd=ROOT,
    )


def vosk_model_ready() -> bool:
    if not VOSK_DIR.is_dir():
        return False
    # Full model includes acoustic model weights
    return (VOSK_DIR / "am").is_dir() or any(VOSK_DIR.glob("**/final.mdl"))


def download_vosk_model() -> None:
    if vosk_model_ready():
        print(f"Vosk model already present at {VOSK_DIR}")
        return

    print("Downloading Vosk model (~40 MB)...")
    zip_path = ROOT / "vosk-model-small-en-us-0.15.zip"
    urlretrieve(VOSK_ZIP_URL, zip_path)

    extract_root = ROOT / "Voice_Assistant" / "Models"
    extract_root.mkdir(parents=True, exist_ok=True)

    print("Extracting Vosk model...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    zip_path.unlink(missing_ok=True)

    if not vosk_model_ready():
        raise RuntimeError("Vosk model extraction failed — check Voice_Assistant/Models/")

    print(f"Vosk model ready at {VOSK_DIR}")


def build_faiss_index() -> None:
    if FAISS_PATH.is_file():
        print(f"FAISS index already exists at {FAISS_PATH}")
        return

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Missing metadata: {METADATA_PATH}")

    print("Building FAISS index from rag_metadata.json (may take a few minutes)...")
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    with open(METADATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    texts = data["texts"]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"Wrote FAISS index ({len(texts)} vectors) to {FAISS_PATH}")


def main() -> int:
    os.chdir(ROOT)
    try:
        run_pip_install()
        download_vosk_model()
        build_faiss_index()
    except Exception as exc:
        print(f"Setup failed: {exc}")
        return 1

    print()
    print("Setup complete. Run the assistant:")
    print(f"  cd {ROOT}")
    print("  python run.py              # voice mode (needs microphone)")
    print("  python run.py --text       # type questions (no mic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
