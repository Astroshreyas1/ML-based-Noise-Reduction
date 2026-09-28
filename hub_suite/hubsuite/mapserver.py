"""Live map + transcript server (Flask, Server-Sent Events). Port 8770 by default.

    GET  /                  map page (Leaflet + milsymbol; OSM tiles when online, a 1 km MGRS grid always)
    GET  /events            SSE stream: partial | transmission | marker | enrich
    GET  /api/state         snapshot (markers + recent transmissions)
    GET  /api/audio/<id>    the enhanced audio of a transmission (click a marker to hear the source)
    POST /api/ingest        {"id", "text", "audio_path"?} -- external STT can feed the map
    POST /api/marker        {"grid", "label", "kind"} -- operator marker
    POST /api/enrich        {"id", "fields"} -- System-2 proposals (grounded before use)
"""
from __future__ import annotations

import json
import queue
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

from .situation import Situation

PAGE = Path(__file__).with_name("map.html")


def create_app(sit: Situation) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE.read_text(encoding="utf-8").replace("__SQUARE__", sit.square)

    @app.get("/api/state")
    def state():
        return jsonify(sit.snapshot())

    @app.get("/events")
    def events():
        q = sit.subscribe()

        def gen():
            yield "retry: 1000\n\n"
            while True:
                try:
                    ev = q.get(timeout=15)
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev['data'], default=str)}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
        return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/audio/<tx_id>")
    def audio(tx_id: str):
        tx = sit.tx.get(tx_id)
        if tx is None or not tx.audio_path or not Path(tx.audio_path).exists():
            return ("", 404)
        return send_file(tx.audio_path, mimetype="audio/wav")

    @app.post("/api/ingest")
    def ingest():
        d = request.get_json(force=True)
        tx = sit.ingest(str(d["id"]), str(d["text"]), audio_path=d.get("audio_path"))
        return jsonify({"id": tx.id, "extraction": tx.extraction, "latency": tx.latency})

    @app.post("/api/marker")
    def marker():
        d = request.get_json(force=True)
        m = sit.operator_marker(str(d["grid"]), str(d.get("label", "note")), str(d.get("kind", "note")))
        return jsonify({"ok": m is not None, "id": m.id if m else None})

    @app.post("/api/enrich")
    def enrich():
        d = request.get_json(force=True)
        return jsonify(sit.enrich(str(d["id"]), dict(d.get("fields", {})), str(d.get("source", "llm"))))

    return app


def serve(sit: Situation, port: int = 8770) -> None:
    import threading
    app = create_app(sit)
    th = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False),
                          daemon=True)
    th.start()
