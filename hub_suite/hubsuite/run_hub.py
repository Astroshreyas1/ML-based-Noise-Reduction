"""The hub, end to end, on the sandbox scenario (or on recorded audio):

    radio audio -> [streaming HubNet, CUDA graph, 18 ms] -> officer headset (+ comfort bed, ducked)
                                                        -> segmenter -> live ASR (partials) -> final ASR
                                                        -> rules + Laya -> situation -> live map (SSE)
                                                        -> System-2 LLM (async, grounded)
    python -m hubsuite.run_hub --events 12 [--laya] [--llm http://127.0.0.1:8080/v1] [--comfort line_alive]
    open http://127.0.0.1:8770

Reports per transmission: end-of-speech -> final text -> marker latency, and marker position error vs truth.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .comfort import ComfortBed, ComfortConfig
from .mapserver import serve
from .sim import Scenario, position_error_m
from .situation import Situation
from .stream_hub import GraphedStreamingHubNet, StreamingHubNet
from .stt import FinalASR, LiveASR, Segmenter

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
HOP = 96


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "artefacts" / "models" / "hubnet_hub1_early.pt")
    ap.add_argument("--events", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--laya", action="store_true", help="System-1 typed decisions with Laya")
    ap.add_argument("--laya-checkpoint", default=None)
    ap.add_argument("--llm", default=None, help="OpenAI-compatible endpoint for System 2, e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--comfort", default="off", choices=["off", "line_alive", "pacer"])
    ap.add_argument("--asr", default="small.en")
    ap.add_argument("--realtime", action="store_true", help="pace the stream at 1x (default: as fast as possible)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--vad", action="store_true", help="segment by VAD instead of the squelch line")
    ap.add_argument("--hold", type=float, default=0.0, help="keep the map server up N seconds after the run")
    a = ap.parse_args()

    out = ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    (out / "audio").mkdir(parents=True, exist_ok=True)
    laya = None
    if a.laya:
        from .decisions import load_laya
        laya = load_laya(checkpoint=a.laya_checkpoint)
    sit = Situation(laya=laya)
    serve(sit, a.port)
    print(f"map: http://127.0.0.1:{a.port}", flush=True)
    try:
        hub = GraphedStreamingHubNet.load(str(a.model), "cuda")
    except Exception as e:                                      # no CUDA: eager CPU (slower than real time)
        print("CUDA graph unavailable, eager CPU:", e)
        hub = StreamingHubNet.load(str(a.model), "cpu")
    bed = ComfortBed(ComfortConfig(mode=a.comfort))
    live, final = LiveASR(), FinalASR(a.asr)
    seg = Segmenter()
    sys2 = None
    if a.llm:
        from .sys2 import System2
        sys2 = System2(a.llm, sit)
    sc = Scenario(seed=a.seed)
    rows = []
    for k in range(a.events):
        sc.advance(float(sc.rng.uniform(20, 60)))
        ev = sc.speak(sc.next_event())
        stream = np.concatenate([ev.audio, np.zeros(int(0.8 * SR), np.float32)])
        stream = stream[: len(stream) // HOP * HOP]
        tx_id = f"tx{k:03d}"
        live.reset()
        seg = Segmenter()                                       # one transmission per squelch opening
        enhanced, partial = [], ""
        hop_ms = []
        t_speech_end = None
        speech_end_sample = len(ev.audio) - int(0.3 * SR)
        closed = None
        for i in range(0, len(stream), HOP):
            t0 = time.perf_counter()
            x_hop = stream[i: i + HOP]
            y_hop = hub.step(x_hop)
            y_out = bed.process(y_hop, x_hop)                    # officer headset (bed ducked by the INPUT)
            hop_ms.append((time.perf_counter() - t0) * 1000)
            enhanced.append(y_out)
            if i // HOP % 16 == 0:                              # live ASR every ~100 ms
                partial = live.feed(np.concatenate(enhanced[-16:]))
                sit.partial(tx_id, partial)
            if t_speech_end is None and i >= speech_end_sample:
                t_speech_end = time.perf_counter()
            sq = (i < len(ev.audio)) if not a.vad else None      # receiver squelch: open while the carrier is up
            done = seg.push(y_hop, sq, partial)
            if done is not None and closed is None:
                closed = (time.perf_counter(), done)
            if a.realtime:
                time.sleep(max(0.0, HOP / SR - (time.perf_counter() - t0)))
        y = np.concatenate(enhanced)
        path = out / "audio" / f"{tx_id}.wav"
        sf.write(path, y, SR)
        t_close = closed[0] if closed else time.perf_counter()
        text, asr_s, words = final.transcribe(closed[1] if closed else y)
        tx = sit.ingest(tx_id, text, audio_path=str(path),
                        extra_latency={"segment_close_ms": (t_close - (t_speech_end or t_close)) * 1000,
                                       "final_asr_ms": asr_s * 1000}, words=words)
        t_marker = time.perf_counter()
        if sys2 is not None:
            threading.Thread(target=sys2.enrich, args=(tx_id,), daemon=True).start()
        mk = [m for m in sit.markers.values() if m.tx == tx_id]
        err = position_error_m(ev.truth["grid"], mk[0].lat, mk[0].lon) if (mk and ev.truth.get("grid")) else None
        row = {"tx": tx_id, "kind": ev.kind, "truth_text": ev.text, "asr_text": text, "link": ev.info.get("link"),
               "snr_db": round(ev.info.get("snr_db", 0), 1), "truth_grid": ev.truth.get("grid"),
               "extracted_grids": tx.extraction.get("grids"), "marker_kind": mk[0].kind if mk else None,
               "truth_kind": ev.truth.get("kind"), "pos_err_m": err,
               "confirmed": mk[0].details.get("confirmed") if mk else None, "grid_conf": mk[0].details.get("confidence") if mk else None,
               "hub_ms_per_hop_p95": float(np.percentile(hop_ms, 95)),
               "speech_end_to_marker_ms": (t_marker - (t_speech_end or t_marker)) * 1000, **tx.latency}
        rows.append(row)
        print(json.dumps({k2: v for k2, v in row.items() if k2 not in ("truth_text",)}, default=str), flush=True)
    (out / "report.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    gridded = [r for r in rows if r["truth_grid"]]
    hit = [r for r in gridded if r["pos_err_m"] is not None and r["pos_err_m"] < 1000]
    summ = {"events": len(rows), "grid_reports": len(gridded), "markers_within_1km": len(hit),
            "median_speech_end_to_marker_ms": float(np.median([r["speech_end_to_marker_ms"] for r in rows])),
            "hub_ms_per_hop_p95": float(np.median([r["hub_ms_per_hop_p95"] for r in rows]))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("SUMMARY", json.dumps(summ), flush=True)
    if a.hold:
        time.sleep(a.hold)


if __name__ == "__main__":
    main()
