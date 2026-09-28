"""Paired (audio, transcript) data for fine-tuning whisper on what the hub actually hears: radio audio
after HubNet. Two sources:

  sim   finetune_asr/texts.py transmissions -> Piper TTS (904 LibriTTS-R voices, split by voice:
        id % 10 == 0 test, == 1 val, rest train) -> synthetic battlefield noise (0-15 dB) -> tactical
        radio link (CVSD16 / FM / CVSD32, hubsuite/radio_link.py).
  real  battlefield_v31 radio clips whose speech has a true transcript (ATCOSIM, Lombard GRID, Speech
        Commands strings, capped TTS), kept only where the 6 s crop still holds the whole transcript
        (whisper word count within 0.8-1.25x of the transcript's). Keeps whisper anchored to real voices.

Each item is stored twice at 8 kHz (the radio band; the loader upsamples): `radio/` = the link output and
`enh/` = after HubNet, so training can mix both and the ASR does not overfit one HubNet checkpoint.

    python finetune_asr/make_asr_data.py sim  --out data_asr --n-train 24000 --n-val 600 --n-test 600 --workers 8
    python finetune_asr/make_asr_data.py real --out data_asr --v31 ../data/battlefield_v31 --n-train 10000
    python finetune_asr/make_asr_data.py enhance --out data_asr --hubnet artefacts/models/hubnet_best.pt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from texts import make  # noqa: E402

SR = 16000
SR_STORE = 8000
SEED = 20260928
N_VOICES = 904


def split_voices(split: str) -> np.ndarray:
    v = np.arange(N_VOICES)
    return v[v % 10 == 0] if split == "test" else v[v % 10 == 1] if split == "val" else v[v % 10 >= 2]


def _save(path: Path, x16: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = resample_poly(np.asarray(x16, np.float32), 1, 2)
    sf.write(str(path), np.clip(y, -1, 1), SR_STORE, subtype="PCM_16")


# ---- sim ------------------------------------------------------------------------------------------
_VOICE = None


def _sim_one(job: tuple[str, int, str]) -> dict:
    global _VOICE
    from piper import PiperVoice, SynthesisConfig

    from hubsuite.radio_link import radio_link
    from hubsuite.sim import battlefield_noise
    split, i, out = job
    r = np.random.default_rng([SEED, {"train": 0, "val": 1, "test": 2}[split], i])
    if _VOICE is None:
        import onnxruntime as ort
        _VOICE = PiperVoice.load(str(ROOT / "artefacts" / "tts" / "en_US-libritts_r-medium.onnx"))
        so = ort.SessionOptions()                                # one thread per worker: the pool is the parallelism
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        _VOICE.session = ort.InferenceSession(str(ROOT / "artefacts" / "tts" / "en_US-libritts_r-medium.onnx"), so,
                                              providers=["CPUExecutionProvider"])
    t = make(r)
    voice = int(r.choice(split_voices(split)))
    chunks = list(_VOICE.synthesize(t["text"], SynthesisConfig(speaker_id=voice, length_scale=float(r.uniform(0.8, 1.05)))))
    x = np.concatenate([c.audio_float_array for c in chunks])
    g = np.gcd(chunks[0].sample_rate, SR)
    x = resample_poly(x, SR // g, chunks[0].sample_rate // g).astype(np.float32)
    x = np.concatenate([np.zeros(int(r.uniform(0.15, 0.4) * SR), np.float32), x,
                        np.zeros(int(r.uniform(0.2, 0.5) * SR), np.float32)])
    if len(x) > 29 * SR:                                     # whisper window is 30 s
        return {}
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.05
    snr = float(r.uniform(0, 15))
    noise = battlefield_noise(len(x), r, gunfire=t["kind"] in ("contact", "fire", "fire_adjust", "spot"))
    noise *= np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(noise ** 2)) + 1e-9) * 10 ** (-snr / 20)
    mic = np.clip(x + noise, -1, 1).astype(np.float32)
    y, _, _, info = radio_link(mic, x, np.random.default_rng(int(r.integers(2 ** 31))),
                               {"link_p": {"cvsd16": 0.55, "fm": 0.35, "cvsd32": 0.10}})
    sid = f"sim_{split}_{i:06d}"
    _save(Path(out) / split / "radio" / f"{sid}.flac", y)
    return {"id": sid, "src": "sim", "text": t["text"], "grids": t["grids"], "kind": t["kind"], "voice": voice,
            "snr_db": round(snr, 2), "link": info.get("link"), "seconds": round(len(y) / SR, 2)}


def cmd_sim(a) -> None:
    for split, n in (("test", a.n_test), ("val", a.n_val), ("train", a.n_train)):
        man = Path(a.out) / f"{split}_sim.jsonl"
        done = {json.loads(l)["id"] for l in man.open(encoding="utf-8")} if man.exists() else set()
        jobs = [(split, i, a.out) for i in range(n) if f"sim_{split}_{i:06d}" not in done]
        print(f"{split}: {len(jobs)} to render ({len(done)} done)", flush=True)
        with ProcessPoolExecutor(a.workers) as ex, man.open("a", encoding="utf-8") as f:
            for k, rec in enumerate(ex.map(_sim_one, jobs, chunksize=16)):
                if rec:
                    f.write(json.dumps(rec) + "\n")
                if k % 1000 == 999:
                    f.flush()
                    print(f"  {split} {k + 1}/{len(jobs)}", flush=True)


# ---- real -----------------------------------------------------------------------------------------
def _clean_transcript(t: str) -> str:
    t = re.sub(r"\s*[/|]\s*", " ", t)
    return " ".join(t.split())


def cmd_real(a) -> None:
    import pandas as pd
    v31 = Path(a.v31)
    words = {}
    for f in (v31 / "words").glob("*.parquet"):
        for r in pd.read_parquet(f).itertuples(index=False):
            words[str(r.id)] = len(json.loads(r.words))
    rng = np.random.default_rng(SEED)
    for split, n in (("test", a.n_test), ("val", a.n_val), ("train", a.n_train)):
        meta = [json.loads(l) for l in (v31 / split / "meta.jsonl").open(encoding="utf-8")]
        ok = []
        for m in meta:
            t = m.get("speech_transcript")
            if not t or not m.get("radio") or m["speech_corpus"] not in ("atcosim", "lombardgrid", "speechcommands", "tts"):
                continue
            t = _clean_transcript(t)
            nw, nt = words.get(str(m["speech_id"]), 0), len(t.split())
            if nt and 0.8 <= nw / nt <= 1.25:
                ok.append((m, t))
        # GRID's "bin blue at f two now" is not radio procedure: at most 20 % of the real items
        grid_ix = [k for k, (m, _) in enumerate(ok) if m["speech_corpus"] == "lombardgrid"]
        other_ix = [k for k, (m, _) in enumerate(ok) if m["speech_corpus"] != "lombardgrid"]
        n_grid = min(len(grid_ix), int(0.2 * n))
        idx = np.concatenate([rng.permutation(other_ix)[: n - n_grid], rng.permutation(grid_ix)[:n_grid]]).astype(int)
        man = Path(a.out) / f"{split}_real.jsonl"
        with man.open("w", encoding="utf-8") as f:
            for k in idx:
                m, t = ok[k]
                p = next(p for e in (".flac", ".wav") if (p := v31 / split / "noisy" / f"{m['id']}{e}").exists())
                x, fs = sf.read(str(p), dtype="float32", always_2d=True)
                x = x[:, 0]
                if fs != SR:
                    x = resample_poly(x, SR, fs).astype(np.float32)
                sid = f"real_{split}_{m['id']}"
                _save(Path(a.out) / split / "radio" / f"{sid}.flac", x)
                f.write(json.dumps({"id": sid, "src": "real", "text": t, "grids": [], "kind": m["speech_corpus"],
                                    "snr_db": round(m["snr_lufs_db"], 2), "link": m["radio"].get("link"),
                                    "seconds": round(len(x) / SR, 2)}) + "\n")
        print(f"{split}: {len(idx)} real of {len(ok)} eligible", flush=True)


# ---- enhance --------------------------------------------------------------------------------------
def cmd_enhance(a) -> None:
    import torch

    from hubsuite.hub_net import HubNet
    ck = torch.load(a.hubnet, map_location="cpu", weights_only=False)
    hp = ck["hp"]
    net = HubNet(hp["c"], hp["n_blocks"], hp["h_intra"], hp["h_inter"], hp["df_order"])
    net.load_state_dict(ck["model"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(dev).eval()
    print(f"HubNet step {ck.get('step')} val {ck.get('val')}", flush=True)
    for split in ("test", "val", "train"):
        for src in ("sim", "real"):
            man = Path(a.out) / f"{split}_{src}.jsonl"
            if not man.exists():
                continue
            recs = [json.loads(l) for l in man.open(encoding="utf-8")]
            todo = [r for r in recs if a.force or not (Path(a.out) / split / "enh" / f"{r['id']}.flac").exists()]
            print(f"{split}_{src}: enhancing {len(todo)}", flush=True)
            todo.sort(key=lambda r: r["seconds"])
            for b in range(0, len(todo), a.batch):
                grp = todo[b: b + a.batch]
                xs = []
                for r in grp:
                    x, fs = sf.read(str(Path(a.out) / split / "radio" / f"{r['id']}.flac"), dtype="float32")
                    xs.append(resample_poly(x, SR // fs, 1).astype(np.float32))
                L = max(len(x) for x in xs)
                X = torch.from_numpy(np.stack([np.pad(x, (0, L - len(x))) for x in xs])).to(dev)
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    Y, _ = net(X)
                Y = Y.float().cpu().numpy()
                for r, x, y in zip(grp, xs, Y):
                    _save(Path(a.out) / split / "enh" / f"{r['id']}.flac", y[: len(x)])
            (Path(a.out) / "hubnet_used.json").write_text(json.dumps({"path": str(a.hubnet), "step": ck.get("step")}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sim", "real", "enhance"])
    ap.add_argument("--out", default=str(ROOT / "data_asr"))
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--v31", default=str(ROOT.parent / "data" / "battlefield_v31"))
    ap.add_argument("--hubnet", default=str(ROOT / "artefacts" / "models" / "hubnet_hub1_early.pt"))
    ap.add_argument("--batch", type=int, default=4, help="4 fits 6 GB; 32 on a 48 GB card")
    ap.add_argument("--force", action="store_true", help="re-enhance everything (new HubNet checkpoint)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"sim": cmd_sim, "real": cmd_real, "enhance": cmd_enhance}[a.cmd](a)


if __name__ == "__main__":
    main()
