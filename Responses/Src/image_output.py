"""Online-only visual guides for emergencies (text-only when offline)."""

from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib.parse import quote, urlparse

from config import IMAGE_CATALOG_PATH

ALLOWED_IMAGE_HOSTS = (
    "upload.wikimedia.org",
    "commons.wikimedia.org",
    "image.pollinations.ai",
)

# Wikimedia requires a descriptive User-Agent for programmatic access.
WIKIMEDIA_HTTP_HEADERS = {
    "User-Agent": (
        "CrisisAI/1.0 (emergency education; "
        "https://github.com/innovatoryuvarajan/gemma-crisis-Ai-response)"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def is_online(timeout: float = 2.0) -> bool:
    """True when the device can reach the public internet."""
    for host, port in (("1.1.1.1", 53), ("8.8.8.8", 53)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def _load_catalog() -> dict[str, Any]:
    if not os.path.isfile(IMAGE_CATALOG_PATH):
        return {}
    with open(IMAGE_CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f).get("topics", {})


def _url_allowed(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        return host in ALLOWED_IMAGE_HOSTS
    except Exception:
        return False


def proxy_target_allowed(url: str) -> bool:
    """Strict allowlist for /api/image-proxy (Wikimedia Commons and Pollinations AI)."""
    try:
        p = urlparse(url)
        if p.scheme != "https":
            return False
        host = (p.hostname or "").lower()
        if host == "image.pollinations.ai":
            return True
        if host != "upload.wikimedia.org":
            return False
        path = (p.path or "").lower()
        if not path.startswith("/wikipedia/commons/"):
            return False
        if any(path.endswith(ext) for ext in (".pdf", ".djvu", ".svg", ".webm", ".ogg")):
            return False
        return True
    except Exception:
        return False


def detect_image_topic(query_text: str, faq_match: dict | None = None) -> str | None:
    """Pick a visual-guide topic from FAQ metadata or keyword catalog."""
    if faq_match:
        topic = faq_match.get("image_topic")
        if topic:
            return topic

    query_lower = query_text.lower()
    topics = _load_catalog()
    best_topic = None
    best_score = 0

    for topic_id, meta in topics.items():
        keywords = meta.get("keywords", [])
        if not keywords:
            continue
        score = sum(1 for kw in keywords if kw.lower() in query_lower)
        if score > best_score:
            best_score = score
            best_topic = topic_id

    return best_topic if best_score > 0 else None


def get_images_for_topic(topic: str) -> list[dict[str, str]]:
    """Return whitelisted image entries for a topic."""
    topics = _load_catalog()
    meta = topics.get(topic)
    if not meta:
        return []

    images = []
    for entry in meta.get("images", []):
        url = (entry.get("url") or "").strip()
        if not url or not _url_allowed(url):
            continue
        images.append(
            {
                "url": url,
                "caption": (entry.get("caption") or topic.replace("_", " ").title()).strip(),
                "topic": topic,
            }
        )
    return images


def resolve_response_media(
    query_text: str,
    response_text: str,
    faq_match: dict | None = None,
    urgency_level: str = "low",
) -> dict[str, Any]:
    """
    Build API/CLI payload: text always; images only when online and topic matches.
    """
    topic = detect_image_topic(query_text, faq_match)
    online = is_online()
    would_show = bool(topic and get_images_for_topic(topic))

    payload: dict[str, Any] = {
        "text": response_text,
        "online": online,
        "images": [],
        "image_topic": topic,
        "visual_guide_available": False,
    }

    if online:
        if topic:
            images = get_images_for_topic(topic)
            if images:
                for im in images:
                    url = im["url"]
                    im["src"] = f"/api/image-proxy?u={quote(url, safe='')}"
                payload["images"] = images
                payload["visual_guide_available"] = True
        
        # If no predefined images exist, but it is an emergency (urgency is medium, high, critical),
        # try to generate a helpful diagram/schematic using Pollinations AI.
        if not payload["images"] and urgency_level in ("medium", "high", "critical"):
            clean_query = query_text.strip().replace("\n", " ")
            if len(clean_query) > 100:
                clean_query = clean_query[:100] + "..."
            
            prompt_str = f"Safety instruction diagram for {clean_query}, first aid steps, clean vector infographic style"
            gen_url = f"https://image.pollinations.ai/prompt/{quote(prompt_str)}?width=800&height=600&nologo=true"
            
            payload["images"] = [{
                "url": gen_url,
                "src": f"/api/image-proxy?u={quote(gen_url, safe='')}",
                "caption": f"AI-Generated Safety Guide: {clean_query}",
                "topic": "generated_emergency_guide"
            }]
            payload["visual_guide_available"] = True

    if would_show and not online:
        payload["offline_text_only"] = True

    return payload
