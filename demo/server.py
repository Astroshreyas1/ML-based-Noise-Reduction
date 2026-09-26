"""Demo server: record 6 s in the browser -> battlefield scene layered on it (same
chain as the dataset, test-split noise) -> ANC-Net v3 -> three players.

    .venv/Scripts/python.exe demo/server.py [--ckpt model/runs/v3a/ckpt_best.pt] [--port 8765] [--cuda]

Open http://localhost:8765 . Inference runs on CPU by default so a training run
can keep the GPU. The checkpoint is re-read when its file changes (the trainer
overwrites ckpt_best.pt every 2 k steps).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_from_directory
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from anc_net_v3 import ANCNetV3  # noqa: E402
from ancdata.battlefield import load_battlefield_config  # noqa: E402
from ancdata.audio import active_rms_db  # noqa: E402
from ancdata.loudness import loudness_lufs  # noqa: E402
from losses import si_snr_db  # noqa: E402

SR = 16000
HOP = 64
MIN_S, MAX_S = 3.0, 30.0          # clips shorter than 3 s are padded (event placement needs room); longer are cut
app = Flask(__name__, static_folder=str(ROOT / "demo" / "static"), static_url_path="")
STATE: dict = {}
LOCK = threading.Lock()


def load_model(ckpt: Path, device: torch.device):
    mtime = ckpt.stat().st_mtime
    if STATE.get("ckpt_mtime") == mtime and STATE.get("net") is not None:
        return STATE["net"], STATE["step"]
    for _ in range(5):                                   # the trainer may be mid-write
        try:
            ck = torch.load(ckpt, map_location=device)
            break
        except Exception:
            time.sleep(1.0)
    hp = ck["hp"]
    net = ANCNetV3(tuple(hp["widths"]), hp["gru_hidden"], hp["gru_layers"]).to(device).eval()
    net.load_state_dict(ck["model"])
    STATE.update(net=net, step=int(ck["step"]), ckpt_mtime=mtime)
    print(f"model: {ckpt} step {ck['step']} val {ck.get('val', {}).get('val_sisnri', float('nan')):.2f} dB SI-SNRi")
    return net, int(ck["step"])


def fit_clip(x: np.ndarray) -> np.ndarray:
    """Any length -> [MIN_S, MAX_S] s, a whole number of hops."""
    x = x[: int(MAX_S * SR)]
    n = max(int(MIN_S * SR), len(x))
    n = int(np.ceil(n / HOP) * HOP)
    return np.ascontiguousarray(np.pad(x, (0, n - len(x))), dtype=np.float32)


def decode_upload(blob: bytes) -> np.ndarray:
    x, fs = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != SR:
        g = np.gcd(int(fs), SR)
        x = resample_poly(x, SR // g, fs // g).astype(np.float32)
    return fit_clip(x)


def wav_b64(x: np.ndarray, peak_norm: bool = False) -> str:
    y = np.asarray(x, dtype=np.float32)
    if peak_norm:
        y = y / (np.abs(y).max() + 1e-6) * 0.9
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), SR, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/scenarios")
def scenarios():
    return jsonify([s["name"] for s in STATE["cfg"]["scenarios"]])


def model_info() -> dict:
    """Facts about the loaded checkpoint, its training run and its test scores (for the About card)."""
    net, step = load_model(STATE["ckpt"], STATE["device"])
    ck = torch.load(STATE["ckpt"], map_location="cpu")
    hp = ck.get("hp", {})
    val = ck.get("val", {})
    info = {
        "params": sum(p.numel() for p in net.parameters()), "step": step,
        "val_sisnri": val.get("val_sisnri"), "val_stoi": val.get("val_stoi"),
        "batch": hp.get("batch"), "steps": hp.get("steps"), "lr": hp.get("lr"), "crop_s": hp.get("crop_s"),
        "device": str(STATE["device"]),
    }
    csv_path = STATE["ckpt"].parent / "eval_test.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        cols = [c for c in ("si_snr_in", "si_snr", "stoi_in", "stoi", "pesq_in", "pesq", "event_recall", "event_precision", "blackout") if c in df.columns]
        info["test"] = {c: round(float(df[c].mean()), 3) for c in cols}
        info["test_n"] = int(len(df))
    csv1 = STATE["ckpt"].parent / "eval_test_single_channel.csv"
    if csv1.exists():
        import pandas as pd
        df1 = pd.read_csv(csv1)
        info["test_1ch"] = {c: round(float(df1[c].mean()), 3) for c in ("si_snr", "stoi", "pesq") if c in df1.columns}
    meta = ROOT / "data" / "battlefield_v3" / "train" / "meta.jsonl"
    if meta.exists():
        n = sum(1 for _ in meta.open(encoding="utf-8"))
        info["train_pairs"] = n
        info["train_hours"] = round(n * 6 / 3600, 1)
    return info


@app.route("/model_info")
def model_info_route():
    with LOCK:
        return jsonify(model_info())


@app.route("/process", methods=["POST"])
def process():
    scenario = request.form.get("scenario", "random")
    snr = request.form.get("snr", "")
    if request.form.get("sample"):                        # canned fallback: a held-out Lombard snippet
        snips = STATE["snippets"]
        row = snips.iloc[int(np.random.default_rng().integers(len(snips)))]
        mic, _ = sf.read(str(ROOT / "data" / "snippets" / "lombard6s" / row["path"]), dtype="float32")
        mic = fit_clip(mic)
    else:
        mic = decode_upload(request.files["audio"].read())
    if active_rms_db(mic) < -60:
        return jsonify({"error": "no signal from the microphone"}), 400
    with LOCK:
        # a fresh chain sized to this clip; pools and snippets are shared, so this is ~ms
        chain = build_chain_for(len(mic) / SR)
        rng = np.random.default_rng()                     # unseeded: every request is a new scene
        for _ in range(300):
            idx = int(rng.integers(0, chain.max_examples))
            plan = chain.plan(idx)
            if scenario == "random" or plan["scenario"] == scenario:
                break
        # layering intensity: random unless the page forces a value
        plan["snr_db"] = float(snr) if snr else float(rng.uniform(-10.0, 15.0))
        t0 = time.time()
        pair = chain.render(plan, speech=mic)
        t_layer = time.time() - t0
        net, step = load_model(STATE["ckpt"], STATE["device"])
        with torch.no_grad():
            x = torch.from_numpy(pair.noisy)[None].to(STATE["device"])
            t0 = time.time()
            est, ev_logits, gain, _ = net(x)
            t_inf = time.time() - t0
        est = est[0].cpu().numpy()
        p_ev = torch.softmax(ev_logits[0], -1)[:, 1:].sum(-1).cpu().numpy()
    clean = pair.clean
    boom = pair.noisy[0]
    events = [{"t": e.start / SR, "cat": e.category, "db": round(e.ratio_db, 1)} for e in pair.events if e.category != "hard_negative"]
    m = pair.meta
    out = {
        "seconds": round(len(mic) / SR, 1), "noise_split": STATE["split"],
        "scenario": m["scenario"], "snr_db": round(m["snr_lufs_db"], 1), "snr_effective_db": round(m["snr_effective_db"], 1),
        "layers": [f"{L['pool']} ({L['kind']})" for L in m["layers"]] + [f"{m['bed2']['pool']} (bed2)"] + ([f"{m['wind']['pool']} (wind)"] if m["wind"] else []),
        "events": events, "model_step": step, "t_layer_ms": round(t_layer * 1000), "t_infer_ms": round(t_inf * 1000),
        "si_snr_in": round(float(si_snr_db(torch.from_numpy(boom)[None], torch.from_numpy(clean)[None])[0]), 1),
        "si_snr_out": round(float(si_snr_db(torch.from_numpy(est)[None], torch.from_numpy(clean)[None])[0]), 1),
        "clean": wav_b64(clean, peak_norm=True), "noisy": wav_b64(boom), "enhanced": wav_b64(est, peak_norm=True),
        "p_event": [round(float(v), 3) for v in p_ev[::4]],
        "gain": [round(float(v), 3) for v in gain[0].cpu().numpy()[::4]],
    }
    return jsonify(out)


def build_chain_for(seconds: float):
    from ancdata.battlefield import BattlefieldChain
    return BattlefieldChain(STATE["cfg"], STATE["split"], pools=STATE["pools"], snippets=STATE["snippets_all"], seconds=seconds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "model" / "runs" / "v3a" / "ckpt_best.pt")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "battlefield.yaml")
    ap.add_argument("--split", default="heldout", help="noise pools: heldout (val+test, never trained on), test, val, train, all")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    STATE["device"] = torch.device("cuda" if a.cuda and torch.cuda.is_available() else "cpu")
    STATE["cfg"] = load_battlefield_config(a.config)
    STATE["split"] = a.split
    from ancdata.pools import Pools
    from ancdata.snippets import load_snippets
    STATE["pools"] = Pools(a.split, cache_mb=1024)
    STATE["snippets_all"] = load_snippets(ROOT / "data" / "snippets" / "lombard6s")
    STATE["snippets"] = build_chain_for(6.0).snippets      # held-out snippets for the sample button
    print(f"noise pools ({a.split}): {len(STATE['pools'].df)} files in {len(STATE['pools'].pools())} pools; "
          f"{len(STATE['snippets'])} sample snippets")
    STATE["ckpt"] = a.ckpt
    load_model(a.ckpt, STATE["device"])
    print(f"demo -> http://localhost:{a.port}   (noise: {a.split} pools; inference on {STATE['device']})")
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
