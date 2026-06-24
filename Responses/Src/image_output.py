"""Online-only visual guides for emergencies (text-only when offline)."""

from __future__ import annotations

import json
import os
import socket
import time
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

_LAST_ONLINE_CHECK_TIME = 0.0
_LAST_ONLINE_CHECK_RESULT = False
ONLINE_CHECK_CACHE_TTL = 60.0  # seconds


def is_online(timeout: float = 1.0) -> bool:
    """True when the device can reach the public internet (cached for 60s)."""
    global _LAST_ONLINE_CHECK_TIME, _LAST_ONLINE_CHECK_RESULT
    now = time.time()
    if now - _LAST_ONLINE_CHECK_TIME < ONLINE_CHECK_CACHE_TTL:
        return _LAST_ONLINE_CHECK_RESULT

    result = False
    for host, port in (("1.1.1.1", 53), ("8.8.8.8", 53)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                result = True
                break
        except OSError:
            continue

    _LAST_ONLINE_CHECK_RESULT = result
    _LAST_ONLINE_CHECK_TIME = now
    return result


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
    """Pick a visual-guide topic from FAQ metadata or keyword catalog.

    For FAQ matches the topic is taken directly from FAQ metadata (curated).
    For free-text catalog scans we require at least TWO keyword hits so that a
    single common word (e.g. "water", "help") doesn't trigger an unrelated
    visual guide.
    """
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

    # Require at least 2 keyword matches to avoid false positives from
    # single common words like "water", "hot", "help", etc.
    return best_topic if best_score >= 2 else None


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


# Session-level deduplication: track recently shown image topics so the same
# visual guide isn't repeated on back-to-back responses.
_RECENT_IMAGE_TOPICS: list[str] = []
_MAX_RECENT_TOPICS = 5  # remember last 5 topics shown


def resolve_response_media(
    query_text: str,
    response_text: str,
    faq_match: dict | None = None,
    urgency_level: str = "low",
) -> dict[str, Any]:
    """
    Build API/CLI payload: text always; images only when online, topic matches,
    AND the same topic hasn't been shown recently.
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

    if online and topic:
        # Skip images if this exact topic was already shown recently
        if topic not in _RECENT_IMAGE_TOPICS:
            images = get_images_for_topic(topic)
            if images:
                for im in images:
                    url = im["url"]
                    im["src"] = f"/api/image-proxy?u={quote(url, safe='')}"
                payload["images"] = images
                payload["visual_guide_available"] = True

                # Record this topic so it won't repeat immediately
                _RECENT_IMAGE_TOPICS.append(topic)
                if len(_RECENT_IMAGE_TOPICS) > _MAX_RECENT_TOPICS:
                    _RECENT_IMAGE_TOPICS.pop(0)

    if would_show and not online:
        payload["offline_text_only"] = True

    return payload

