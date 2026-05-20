"""Present query results in CLI / voice (text always; images when online)."""

from __future__ import annotations

import sys
import webbrowser
from typing import Any


def present_result(result: dict[str, Any], *, open_images: bool = False) -> str:
    """Print text response; show or open images only when online."""
    text = result.get("text", "")
    print(f"\nCrisis-AI: {text}")

    if result.get("visual_guide_available") and result.get("images"):
        print("\nVisual guide (online):")
        for img in result["images"]:
            print(f"  - {img.get('caption', 'Guide')}: {img['url']}")
        if open_images and sys.platform == "win32":
            try:
                webbrowser.open(result["images"][0]["url"])
            except Exception:
                pass
    elif result.get("offline_text_only"):
        print("\n(Offline — text only. Connect to the internet for visual guides.)")

    return text
