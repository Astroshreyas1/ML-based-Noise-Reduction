"""Battlefield chain (dataset v3): one 6 s (noisy[boom, ref], clean) pair per
index, fully seeded, from real Lombard speech snippets and the noise pools.

    from ancdata.battlefield import BattlefieldChain, load_battlefield_config
    chain = BattlefieldChain(load_battlefield_config("configs/battlefield.yaml"), "train")
    pair = chain.generate(4182993)          # reproducible in isolation

Order of operations (physics, then electronics; each stage is a config key)::

     1. speech        snippet = index mod n_snippets(split); level U(speech.level_db) dBFS active
     2. scenario      weighted draw; bed pool + envelope, foreground events with envelopes / Doppler
     3. scene         bed + events (2-12 dB under) + a second rougher bed + texture hits + capsule buffets,
                      each layer unit-normalised then summed (power sum); wind dry, uncorrelated between mics
     4. SNR           scene scaled so L_K(speech) - L_K(scene) = snr_db   (BS.1770 loudness, not RMS)
     5. transients    gunfire bursts (p) and blasts (p) at a PEAK ratio vs the speech peak; spans logged
     6. boom channel  mic EQ -> self-noise -> saturation -> AGC/limiter -> clip -> 16-bit  (ancdata/channel.py)
     7. TARGET        dry speech x the boom AGC gain trajectory (linear time-varying gain is shared;
                      EQ, floor, saturation, clipping, quantisation are input-only)
     8. ref channel   speech 12-18 dB down + scene 1-3 dB up + the same transients, through the ref electronics
     9. contract      shapes, dtype, finite, |noisy| <= 1, target non-silent, spans in range

Why these choices, with the listening-trial verdicts: docs/NOISE_LAYERING.md.
What the built set contains: docs/DATASET_V3.md.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .audio import EPS, active_rms_db, db_to_lin, rms
from .chain import Event, Pair, assert_contract
from .channel import capsule_channel, ref_channel
from .config import SR, child_rng, sample
from .loudness import gain_for_snr_lufs
from .paths import data_root
from .pools import Pools
from .scene import ENVELOPE_PRIORS, apply_envelope
from .snippets import load_snippets
from .transients import audible_span, lowpass1, render_blast, render_burst

REQUIRED = ("name", "seed", "sample_rate", "segment_seconds", "channels", "target", "speech", "mix", "gunfire",
            "blasts", "wind", "channel", "ref", "scenarios", "pools", "build")


def load_battlefield_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if int(cfg["sample_rate"]) != SR:
        raise ValueError(f"{path}: sample_rate {cfg['sample_rate']} != {SR} (locked)")
    if list(cfg["channels"]) != ["boom", "ref"]:
        raise ValueError(f"{path}: channels must be [boom, ref]")
    if cfg["target"] != "dry_x_agc":
        raise ValueError(f"{path}: target must be 'dry_x_agc'")
    for key in ("gunfire", "blasts"):
        if not 0.0 <= float(cfg[key]["p"]) <= 1.0:
            raise ValueError(f"{path}: {key}.p out of range")
    if not cfg["scenarios"]:
        raise ValueError(f"{path}: no scenarios")
    cfg["_path"] = str(path)
    return cfg


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _crop_at(x: np.ndarray, n: int, offset: int) -> np.ndarray:
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))) + 1)
    offset = int(offset) % (len(x) - n + 1)
    return np.ascontiguousarray(x[offset:offset + n], dtype=np.float32)


def _unit(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    return x / (db_to_lin(active_rms_db(x if ref is None else ref)) + EPS)


def _place(out: np.ndarray, x: np.ndarray, t0: int) -> None:
    m = min(len(x), len(out) - t0)
    if m > 0:
        out[t0: t0 + m] += x[:m]


def _fade(x: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(x) // 2)
    if n > 0:
        x[:n] *= np.linspace(0, 1, n, dtype=np.float32)
        x[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return x


def _local_snr_db(speech: np.ndarray, noise: np.ndarray, a: int, b: int) -> float:
    return float(10.0 * np.log10((np.sum(speech[a:b] ** 2) + EPS) / (np.sum(noise[a:b] ** 2) + EPS)))


# --------------------------------------------------------------------------
# Chain
# --------------------------------------------------------------------------
class BattlefieldChain:
    def __init__(self, cfg: dict[str, Any], split: str, pools: Pools | None = None,
                 snippets: pd.DataFrame | None = None, seconds: float | None = None):
        """`seconds` overrides cfg segment_seconds (the live demo renders clips of any length);
        `split` may also be 'heldout' (val + test) or 'all'."""
        from .pools import split_mask
        if seconds is not None:
            cfg = dict(cfg, segment_seconds=float(seconds))
        self.cfg = cfg
        self.split = split
        self.sr = SR
        self.n = int(round(float(cfg["segment_seconds"]) * SR))
        root = data_root() / cfg["speech"]["snippets"]
        df = snippets if snippets is not None else load_snippets(root)
        if cfg["speech"].get("lombard_class_only", True):
            df = df[df.lombard_class]
        df = df[split_mask(df.split, split)].sort_values("id").reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"no Lombard snippets for split {split!r} under {root}")
        # fixed seeded permutation: any contiguous index range mixes corpora, speakers and efforts
        perm = np.random.default_rng([int(cfg["seed"]), 7]).permutation(len(df))
        self.snippets = df.iloc[perm].reset_index(drop=True)
        self.pools = pools or Pools(split, cache_mb=float(cfg["pools"].get("cache_mb", 512)),
                                    allow_voice=bool(cfg["pools"].get("allow_voice", False)))
        w = np.array([float(s.get("weight", 1.0)) for s in cfg["scenarios"]])
        self._scn_p = w / w.sum()
        self._pool_ok = {p: self.pools.n(p) > 0 for p in self.pools.pools()}

    # ---- sizes ---------------------------------------------------------------
    @property
    def n_snippets(self) -> int:
        return len(self.snippets)

    @property
    def max_examples(self) -> int:
        cap = self.cfg["speech"]["max_scenes_per_snippet"]
        return self.n_snippets * int(cap.get(self.split, 8) if isinstance(cap, dict) else cap)

    # ---- plan ----------------------------------------------------------------
    def _pick_pool(self, alternatives: list[str], rng: np.random.Generator) -> str:
        ok = [p for p in alternatives if self._pool_ok.get(p)]
        if not ok:
            raise KeyError(f"none of {alternatives} has files in split {self.split!r}")
        return ok[int(rng.integers(len(ok)))]

    def _draw_file(self, pool: str, rng: np.random.Generator) -> dict[str, Any]:
        row = self.pools.draw(pool, rng)
        return {"pool": pool, "path": row["path"], "start_s": float(row["start_s"]), "end_s": float(row["end_s"]),
                "duration_s": float(row["duration_s"]), "shot_time": float(row["shot_time"]),
                "group": str(row["group"])}

    def _envelope_params(self, kind: str, rng: np.random.Generator) -> dict[str, Any]:
        e = self.cfg["mix"]["envelopes"]
        return {"kind": kind, "env_seed": int(rng.integers(2 ** 31)),
                "depth_db": sample(e["gust_depth_db"], rng), "rate_hz": sample(e["gust_rate_hz"], rng),
                "ramp_db": sample(e["ramp_db"], rng), "peak_db": sample(e["passby_peak_db"], rng),
                "doppler_pct": float(rng.uniform(0.5, float(e.get("doppler_max_pct", 2.0)))),
                "t0_s": float(rng.uniform(0.0, float(self.cfg["segment_seconds"]) - 2.0))}

    def plan(self, index: int) -> dict[str, Any]:
        cfg = self.cfg
        rng = child_rng(cfg["seed"], index)
        snip = self.snippets.iloc[index % self.n_snippets]
        scn = copy.deepcopy(cfg["scenarios"][int(rng.choice(len(self._scn_p), p=self._scn_p))])
        seg = float(cfg["segment_seconds"])
        plan: dict[str, Any] = {
            "index": int(index), "seed": int(cfg["seed"]), "split": self.split, "scenario": scn["name"],
            "snippet": {"id": snip.id, "path": snip.path, "corpus": snip.corpus, "speaker_id": snip.speaker_id,
                        "effort": snip.effort, "gender": snip.gender, "transcript": snip.transcript,
                        "speech_fraction": float(snip.speech_fraction)},
            "speech_level_db": sample(cfg["speech"]["level_db"], rng),
            "snr_db": sample(cfg["mix"]["snr_db"], rng),
            "layers": [], "bed2": None, "texture": [], "buffets": [], "wind": None, "bursts": [], "blasts": [],
        }
        # bed + events
        bed_pools, bed_env = scn["bed"]
        bed_pool = self._pick_pool(bed_pools, rng)
        plan["layers"].append({**self._draw_file(bed_pool, rng), "offset": int(rng.integers(0, 10 ** 8)),
                               "rel_db": 0.0, **self._envelope_params(bed_env, rng)})
        for pools_, env in scn.get("events", []):
            pool = self._pick_pool(pools_, rng)
            plan["layers"].append({**self._draw_file(pool, rng), "offset": int(rng.integers(0, 10 ** 8)),
                                   "rel_db": -sample(cfg["mix"]["layers_secondary_below_db"], rng),
                                   **self._envelope_params(env, rng)})
        # second rougher bed of the same class
        b2 = cfg["mix"]["bed2"]
        plan["bed2"] = {**self._draw_file(bed_pool, rng), "offset": int(rng.integers(0, 10 ** 8)),
                        "rel_db": sample(b2["rel_db"], rng), "kind": "gust", "env_seed": int(rng.integers(2 ** 31)),
                        "depth_db": sample(b2["gust_depth_db"], rng), "rate_hz": sample(b2["gust_rate_hz"], rng)}
        # texture hits
        tx = cfg["mix"]["texture"]
        for _ in range(int(sample(tx["count"], rng))):
            pool = self._pick_pool(tx["pools"], rng)
            plan["texture"].append({**self._draw_file(pool, rng), "offset": int(rng.integers(0, 10 ** 7)),
                                    "dur_s": sample(tx["dur_s"], rng), "t0_s": float(rng.uniform(0.0, seg - 0.3)),
                                    "rel_db": sample(tx["rel_db"], rng)})
        # capsule buffets
        bf = cfg["mix"]["buffets"]
        for _ in range(int(sample(bf["count"], rng))):
            plan["buffets"].append({"t0_s": float(rng.uniform(0.0, seg - 0.4)), "dur_s": sample(bf["dur_s"], rng),
                                    "rel_db": sample(bf["rel_db"], rng), "seed": int(rng.integers(2 ** 31))})
        # wind at the capsule
        w = cfg["wind"]
        if rng.random() < float(scn.get("wind_p", 0.0)):
            pool = self._pick_pool(w["pools"], rng)
            plan["wind"] = {**self._draw_file(pool, rng), "offset_boom": int(rng.integers(0, 10 ** 8)),
                            "offset_ref": int(rng.integers(0, 10 ** 8)), "rel_db": sample(w["rel_db"], rng),
                            "kind": "gust", "env_seed": int(rng.integers(2 ** 31)),
                            "depth_db": sample(w["gust_depth_db"], rng), "rate_hz": sample(w["gust_rate_hz"], rng)}
        # gunfire
        g = cfg["gunfire"]
        p_gun = min(1.0, float(g["p"]) * float(scn.get("gunfire_scale", 1.0)))
        distant_only = bool(scn.get("distant_only", False))
        if rng.random() < p_gun:
            n_bursts = 0 if distant_only else int(sample(g["bursts"], rng))
            src_names = list(g["source_p"])
            src_p = np.array([float(g["source_p"][k]) for k in src_names])
            for _ in range(n_bursts):
                auto = rng.random() < float(g["auto_p"])
                src = src_names[int(rng.choice(len(src_names), p=src_p / src_p.sum()))]
                b: dict[str, Any] = {"t0_s": float(rng.uniform(0.2, seg - 1.2)), "auto": bool(auto),
                                     "n_rounds": int(sample(g["rounds_auto"] if auto else g["rounds_semi"], rng)),
                                     "spacing_s": sample(g["spacing_auto_s"] if auto else g["spacing_semi_s"], rng),
                                     "source": src, "ratio_db": sample(g["peak_ratio_db"], rng),
                                     "seed": int(rng.integers(2 ** 31)), "tail_tau_s": sample(g["tail_tau_s"], rng),
                                     "tail_db": sample(g["tail_db"], rng), "distant": False}
                if src == "field":
                    b.update(self._draw_file("field_gunshot", rng))
                elif src == "mad":
                    b.update(self._draw_file(self._pick_pool(["mad_gunshot"], rng), rng))
                plan["bursts"].append(b)
            if distant_only or rng.random() < float(g["distant_p"]):
                plan["bursts"].append({"t0_s": float(rng.uniform(0.2, seg - 1.5)), "auto": True,
                                       "n_rounds": int(sample(g["distant_rounds"], rng)),
                                       "spacing_s": sample(g["distant_spacing_s"], rng), "source": "physics",
                                       "ratio_db": sample(g["distant_peak_ratio_db"], rng),
                                       "seed": int(rng.integers(2 ** 31)),
                                       "tail_tau_s": sample(g["distant_tail_tau_s"], rng),
                                       "tail_db": sample(g["distant_tail_db"], rng), "distant": True})
        # blasts
        bl = cfg["blasts"]
        p_bl = min(1.0, float(bl["p"]) * float(scn.get("blast_scale", 1.0)))
        if rng.random() < p_bl:
            src_names = list(bl["source_p"])
            src_p = np.array([float(bl["source_p"][k]) for k in src_names])
            for _ in range(int(sample(bl["count"], rng))):
                src = src_names[int(rng.choice(len(src_names), p=src_p / src_p.sum()))]
                d: dict[str, Any] = {"t0_s": float(rng.uniform(0.1, seg - 1.0)), "source": src,
                                     "ratio_db": sample(bl["peak_ratio_db"], rng), "seed": int(rng.integers(2 ** 31))}
                if src == "mad":
                    d.update(self._draw_file(self._pick_pool(["mad_shelling"], rng), rng))
                plan["blasts"].append(d)
        # electronics
        r = cfg["ref"]
        plan["ref"] = {"speech_db": sample(r["speech_db"], rng), "noise_db": sample(r["noise_db"], rng)}
        plan["boom_seed"] = int(rng.integers(2 ** 31))
        plan["ref_seed"] = int(rng.integers(2 ** 31))
        return plan

    # ---- render --------------------------------------------------------------
    def _layer_audio(self, L: dict[str, Any], offset_key: str = "offset") -> np.ndarray:
        x = self.pools.load(L)
        n = self.n
        if L.get("kind") == "real_passby":
            seg = _fade(x[:n].copy(), int(0.05 * self.sr))
            out = np.zeros(n, dtype=np.float32)
            _place(out, seg, min(int(L["t0_s"] * self.sr), n - len(seg)))
            return out / (db_to_lin(active_rms_db(seg)) + EPS)     # normalised by the clip, not the frame
        y = _crop_at(x, n, L[offset_key])
        kind = L.get("kind", "steady")
        if kind != "steady":
            e = self.cfg["mix"]["envelopes"]
            r = np.random.default_rng(int(L["env_seed"]))
            spec = {"gust_depth_db": L["depth_db"], "gust_rate_hz": L["rate_hz"], "ramp_db": L["ramp_db"],
                    "passby_peak_db": L["peak_db"], "doppler_max_pct": L["doppler_pct"]} if "ramp_db" in L else \
                   {"gust_depth_db": L["depth_db"], "gust_rate_hz": L["rate_hz"]}
            y, _ = apply_envelope(y, kind, r, spec)
        return _unit(y)

    def render(self, plan: dict[str, Any], speech: np.ndarray | None = None) -> Pair:
        """`speech` (16 kHz mono float32) replaces the plan's snippet -- used by the live
        demo to put a microphone recording through the same scene / channel as the data."""
        cfg, n, sr = self.cfg, self.n, self.sr
        # 1. speech
        from .audio import load_mono
        sp = speech if speech is not None else load_mono(data_root() / cfg["speech"]["snippets"] / plan["snippet"]["path"], sr)
        sp = sp[:n] if len(sp) >= n else np.pad(sp, (0, n - len(sp)))
        s = (sp * db_to_lin(plan["speech_level_db"] - active_rms_db(sp))).astype(np.float32)
        # 2-3. scene
        scene = np.zeros(n, dtype=np.float32)
        for L in plan["layers"]:
            scene += self._layer_audio(L) * db_to_lin(L["rel_db"])
        b2 = plan["bed2"]
        scene += self._layer_audio(b2) * db_to_lin(b2["rel_db"])
        for tx in plan["texture"]:
            x = self.pools.load(tx)
            seg = _fade(_crop_at(x, int(tx["dur_s"] * sr), tx["offset"]).copy(), int(0.02 * sr))
            _place(scene, _unit(seg) * db_to_lin(tx["rel_db"]), int(tx["t0_s"] * sr))
        buf = np.zeros(n, dtype=np.float32)
        for bf in plan["buffets"]:
            r = np.random.default_rng(bf["seed"])
            m = int(bf["dur_s"] * sr)
            seg = lowpass1(lowpass1(r.standard_normal(m).astype(np.float32), 150.0, sr), 150.0, sr)
            seg *= np.hanning(m).astype(np.float32) ** 0.5
            _place(buf, seg / (rms(seg) + EPS) * db_to_lin(bf["rel_db"]), int(bf["t0_s"] * sr))
        wind_boom = wind_ref = None
        if plan["wind"]:
            w = plan["wind"]
            wind_boom = self._layer_audio(w, "offset_boom") * db_to_lin(w["rel_db"])
            wind_ref = self._layer_audio({**w, "env_seed": w["env_seed"] + 1}, "offset_ref") * db_to_lin(w["rel_db"])
        scene_boom = scene + buf + (wind_boom if wind_boom is not None else 0.0)
        scene_ref = scene + 0.5 * buf + (wind_ref if wind_ref is not None else 0.0)
        # 4. SNR on K-weighted loudness
        g = gain_for_snr_lufs(s, scene_boom, plan["snr_db"], sr)
        noise_boom = (scene_boom * g).astype(np.float32)
        noise_ref = (scene_ref * g).astype(np.float32)
        # 5. transients at a peak ratio vs the speech peak
        sp_peak = float(np.abs(s).max()) + EPS
        imp = np.zeros(n, dtype=np.float32)
        events: list[Event] = []
        for b in plan["bursts"]:
            spec = dict(b)
            if b["source"] == "field":
                spec["file"] = str(data_root() / b["path"])
                spec["t_shot"] = b["shot_time"]
            elif b["source"] == "mad":
                spec["file"] = str(data_root() / b["path"])
            w, params = render_burst(spec, np.random.default_rng(b["seed"]), sr)
            t0 = int(b["t0_s"] * sr)
            ev = w * sp_peak * db_to_lin(b["ratio_db"])
            _place(imp, ev, t0)
            a, e = audible_span(ev[: n - t0], t0)
            cat = "gunfire_distant" if b["distant"] else ("gunfire_auto" if b["auto"] else "gunfire_semi")
            events.append(Event(a, e, cat, float(b["ratio_db"]), 0.0,
                                {"source": b["source"], "n_rounds": float(b["n_rounds"]), "spacing_s": float(b["spacing_s"]),
                                 **{k: float(v) for k, v in params.items()}}))
        for d in plan["blasts"]:
            spec = dict(d)
            if d["source"] == "mad":
                spec["file"] = str(data_root() / d["path"])
            w, params = render_blast(spec, np.random.default_rng(d["seed"]), sr)
            t0 = int(d["t0_s"] * sr)
            ev = w * sp_peak * db_to_lin(d["ratio_db"])
            _place(imp, ev, t0)
            a, e = audible_span(ev[: n - t0], t0)
            events.append(Event(a, e, "blast", float(d["ratio_db"]), 0.0,
                                {"source": d["source"], **{k: float(v) for k, v in params.items()}}))
        for tx in plan["texture"]:
            a = int(tx["t0_s"] * sr)
            e = min(n, a + int(tx["dur_s"] * sr))
            if e > a:
                events.append(Event(a, e, "hard_negative", float(tx["rel_db"]), 0.0, {"pool": tx["pool"]}))
        noise_total = noise_boom + imp
        for ev in events:
            ev.local_snr_db = _local_snr_db(s, noise_total, ev.start, ev.end)
        # 6-7. boom channel; target = dry speech x AGC gain
        boom, gain, st = capsule_channel(s + noise_total, np.random.default_rng(plan["boom_seed"]), cfg["channel"], sr)
        clean = (s * gain).astype(np.float32)
        # 8. reference mic
        rp = plan["ref"]
        ref_in = s * db_to_lin(rp["speech_db"]) + (noise_ref + imp) * db_to_lin(rp["noise_db"])
        ref, st_ref = ref_channel(ref_in, np.random.default_rng(plan["ref_seed"]), cfg["ref"]["channel"], sr)
        meta = {
            "index": plan["index"], "seed": plan["seed"], "split": plan["split"], "scenario": plan["scenario"],
            **{f"speech_{k}": v for k, v in plan["snippet"].items()},
            "speech_level_db": plan["speech_level_db"], "snr_lufs_db": plan["snr_db"],
            "snr_effective_db": float(active_rms_db(s) - active_rms_db(noise_boom)),
            "layers": [{k: L[k] for k in ("pool", "path", "start_s", "end_s", "group", "kind", "rel_db")} for L in plan["layers"]],
            "bed2": {k: b2[k] for k in ("pool", "path", "rel_db")},
            "wind": None if not plan["wind"] else {k: plan["wind"][k] for k in ("pool", "path", "rel_db")},
            "n_texture": len(plan["texture"]), "n_buffets": len(plan["buffets"]),
            "gunfire": any(not b["distant"] for b in plan["bursts"]), "n_bursts": len(plan["bursts"]),
            "n_blasts": len(plan["blasts"]),
            "agc_min_gain_db": st["agc_min_gain_db"], "clip_pct": st["clip_pct"],
            "ref_speech_db": rp["speech_db"], "ref_noise_db": rp["noise_db"], "ref_clip_pct": st_ref["clip_pct"],
        }
        if cfg.get("debug_stems"):
            meta["_stems"] = {"speech_dry": s, "scene": noise_boom, "transients": imp, "agc_gain": gain}
        pair = Pair(np.stack([boom, ref]).astype(np.float32), clean, events, meta)
        assert_contract(pair)
        return pair

    def generate(self, index: int) -> Pair:
        return self.render(self.plan(index))


def build_battlefield_chain(cfg: dict[str, Any] | str | Path, split: str, cache_mb: float | None = None) -> BattlefieldChain:
    if not isinstance(cfg, dict):
        cfg = load_battlefield_config(cfg)
    if cache_mb is not None:
        cfg = copy.deepcopy(cfg)
        cfg["pools"]["cache_mb"] = cache_mb
    return BattlefieldChain(cfg, split)
