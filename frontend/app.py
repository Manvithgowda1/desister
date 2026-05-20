import os
import sys
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, abort
import requests

# Ensure we're running from the correct directory so paths work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "Responses", "Src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Change directory so config.py relative paths resolve correctly
os.chdir(SRC)

from query_engine import QueryEngine
from emergency_detector import EmergencyDetector
from image_output import is_online, proxy_target_allowed, WIKIMEDIA_HTTP_HEADERS

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

print("Initializing backend components for Web App...")
query_engine = QueryEngine()
emergency_detector = EmergencyDetector()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def status():
    online = is_online()
    return jsonify({
        "online": online,
        "mode": "online" if online else "offline",
        "images_enabled": online,
        "message": (
            "Visual guides available when relevant"
            if online
            else "Offline mode — text responses only"
        ),
    })


@app.route("/api/image-proxy", methods=["GET"])
def image_proxy():
    """Serve Wikimedia images with a compliant User-Agent (browser hotlink often fails)."""
    from urllib.parse import unquote

    raw = request.args.get("u", "")
    url = unquote(raw).strip()
    if not url or not proxy_target_allowed(url):
        abort(403)

    try:
        upstream = requests.get(
            url,
            headers=WIKIMEDIA_HTTP_HEADERS,
            stream=True,
            timeout=60,
        )
    except requests.RequestException:
        abort(502)

    if upstream.status_code != 200:
        upstream.close()
        abort(502)

    content_type = (upstream.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()
    if not content_type.startswith("image/"):
        upstream.close()
        abort(502)

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=16384):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return Response(
        stream_with_context(generate()),
        mimetype=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    is_emergency, keyword = emergency_detector.detect_sos_in_text(text)
    if is_emergency:
        print(f"EMERGENCY DETECTED in web UI: {keyword}")
        emergency_detector.handle_emergency(text, keyword)
    
    result = query_engine.process_query(text)
    
    return jsonify({
        "response": result["text"],
        "is_emergency": is_emergency,
        "keyword": keyword,
        "online": result.get("online", False),
        "images": result.get("images", []),
        "visual_guide_available": result.get("visual_guide_available", False),
        "offline_text_only": result.get("offline_text_only", False),
        "image_topic": result.get("image_topic"),
        "urgency": result.get("urgency", {}),
    })

def start_server():
    print("\n🌐 Starting Crisis-AI Web Interface...")
    print("👉 Open http://127.0.0.1:5000 in your browser\n")
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == '__main__':
    start_server()
