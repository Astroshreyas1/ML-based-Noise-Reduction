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

# clip-level scene labels (v3.1 meta): pool -> label, first matching key wins
POOL_LABEL_EXACT = {"esc50_train": "vehicle_heavy", "esc50_sea_waves": "ambience_rural", "esc50_crackling_fire": "fire_crackle",
                    "esc50_car_horn": "vehicle_light", "esc50_glass_breaking": "debris_impact", "fsd_debris": "debris_impact",
                    "esc50_fireworks": "hard_negative", "fsd_fireworks": "hard_negative", "fsd_fire": "fire_crackle",
                    "fsd_nature": "ambience_rural", "fsd_water": "ambience_rural", "fsd_traffic": "ambience_urban",
                    "fsd_fan": "engine_machinery", "fsd_gear": "footsteps_gear", "demand_tbus": "vehicle_interior",
                    "fsd_interferer": "speech_interferer", "mad_communication": "speech_interferer"}
POOL_LABELS = (("helicopter", "helicopter"), ("fighter", "jet_aircraft"), ("aircraft", "jet_aircraft"),
               ("airplane", "jet_aircraft"), ("drone", "drone_uav"), ("siren", "siren_alarm"), ("thunder", "thunder"),
               ("rain", "rain"), ("wind", "wind_ambient"), ("shelling", "blast_distant"), ("explosion", "blast_distant"),
               ("chainsaw", "engine_machinery"), ("jackhammer", "engine_machinery"), ("engine", "engine_machinery"),
               ("idling", "engine_machinery"), ("truck", "vehicle_heavy"), ("bus", "vehicle_heavy"),
               ("mad_vehicle", "vehicle_heavy"), ("idmt_car", "vehicle_light"), ("motorcycle", "vehicle_light"),
               ("tcar", "vehicle_interior"), ("tmetro", "vehicle_interior"), ("nfield", "ambience_rural"),
               ("npark", "ambience_rural"), ("nriver", "ambience_rural"), ("straffic", "ambience_urban"),
               ("spsquare", "ambience_urban"), ("footsteps", "footsteps_gear"), ("gear", "footsteps_gear"))


def audible_fraction(sig: np.ndarray, rest: np.ndarray, sr: int = SR, frame_s: float = 0.05, within_db: float = -10.0) -> float:
    """Fraction of 50 ms frames (where sig is active) in which sig is within `within_db` of everything
    else. Energy over the whole clip would let one gunshot 'mask' a helicopter that is plainly audible."""
    m = int(frame_s * sr)
    k = len(sig) // m
    es = np.sum(sig[: k * m].reshape(k, m) ** 2, axis=1)
    er = np.sum(rest[: k * m].reshape(k, m) ** 2, axis=1)
    act = es > es.max() * 1e-4 if es.max() > 0 else np.zeros(k, bool)
    if not act.any():
        return 0.0
    return float(np.mean(10 * np.log10((es[act] + EPS) / (er[act] + EPS)) >= within_db))


def pool_label(pool: str) -> str:
    if pool in POOL_LABEL_EXACT:
        return POOL_LABEL_EXACT[pool]
    for key, lab in POOL_LABELS:
        if key in pool:
            return lab
    return "hard_negative"

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
        df = df.assign(root=cfg["speech"]["snippets"])
        extra = cfg["speech"].get("extra_sets") or []
        if extra:                                  # v3.1: radio-procedure speech mixed in at fixed corpus shares
            df = self._mix_extra(df, extra, split, int(cfg["seed"]))
        # fixed seeded permutation: any contiguous index range mixes corpora, speakers and efforts
        perm = np.random.default_rng([int(cfg["seed"]), 7]).permutation(len(df))
        self.snippets = df.iloc[perm].reset_index(drop=True)
        self.pools = pools or Pools(split, cache_mb=float(cfg["pools"].get("cache_mb", 512)),
                                    allow_voice=bool(cfg["pools"].get("allow_voice", False)),
                                    require_audit=bool(cfg["pools"].get("require_audit", False)))
        w = np.array([float(s.get("weight", 1.0)) for s in cfg["scenarios"]])
        self._scn_p = w / w.sum()
        self._pool_ok = {p: self.pools.n(p) > 0 for p in self.pools.pools()}

    @staticmethod
    def _mix_extra(base: pd.DataFrame, extra: list[dict[str, Any]], split: str, seed: int) -> pd.DataFrame:
        """Each extra corpus gets `share` of the final snippet list (base keeps the rest). A corpus with
        fewer snippets than its quota is re-used (different scenes per use); one with more is subsampled."""
        from .pools import split_mask
        shares = {c: float(s) for e in extra for c, s in e["share"].items()}
        base_share = 1.0 - sum(shares.values())
        frames = [base]
        for i, e in enumerate(extra):
            root = data_root() / e["root"]
            if not (root / "meta.parquet").exists():
                continue
            d = pd.read_parquet(root / "meta.parquet")
            d = d[split_mask(d.split, split)].assign(root=e["root"]).sort_values("id").reset_index(drop=True)
            for c, s in e["share"].items():
                dc = d[d.corpus == c]
                if len(dc) == 0:
                    continue
                want = int(round(len(base) * s / base_share))
                r = np.random.default_rng([seed, 11, i, sum(map(ord, c))])
                idx = r.choice(len(dc), size=want, replace=want > len(dc))
                frames.append(dc.iloc[np.sort(idx)])
        cols = list(base.columns)
        out = pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)
        out["id"] = out["id"].astype(str)
        # repeated snippets need distinct sort keys so the permutation stays deterministic
        out["_k"] = out.groupby("id").cumcount()
        out = out.sort_values(["id", "_k"]).drop(columns="_k").reset_index(drop=True)
        return out

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
            "snippet": {"id": snip.id, "path": snip.path, "root": str(getattr(snip, "root", cfg["speech"]["snippets"])),
                        "corpus": snip.corpus, "speaker_id": snip.speaker_id,
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
        if cfg.get("logic"):
            self._logic_pass(plan, scn, index)
        # electronics
        r = cfg["ref"]
        plan["ref"] = {"speech_db": sample(r["speech_db"], rng), "noise_db": sample(r["noise_db"], rng)}
        plan["boom_seed"] = int(rng.integers(2 ** 31))
        plan["ref_seed"] = int(rng.integers(2 ** 31))
        if cfg.get("radio"):                    # own stream: v3 / v3.1 draws above are untouched
            rr = np.random.default_rng([int(cfg["seed"]), int(index), 53])
            plan["radio"] = {"on": bool(rr.random() < float(cfg["radio"].get("p", 0.0))), "seed": int(rr.integers(2 ** 31))}
        return plan

    # ---- v3.1 layering logic ---------------------------------------------------
    def _extra_pieces(self, L: dict[str, Any], rng: np.random.Generator, seg: float) -> None:
        total = L["end_s"] - L["start_s"]
        extra: list[dict[str, Any]] = []
        while total < seg + 1.0 and len(extra) < 6:
            e = self._draw_file(L["pool"], rng)
            extra.append({k: e[k] for k in ("path", "start_s", "end_s", "duration_s")})
            total += e["end_s"] - e["start_s"] - 0.6
        if extra:
            L["extra"] = extra

    def _logic_pass(self, plan: dict[str, Any], scn: dict[str, Any], index: int) -> None:
        """v3.1: same scene recipe and density as v3 (the sound the listening trials chose);
        fixes the *logic* of how layers are cut, timed and levelled. Own RNG stream, so the
        v3 draws before it are untouched."""
        lg = self.cfg["logic"]
        rng = np.random.default_rng([int(self.cfg["seed"]), int(index), 31])
        seg = float(self.cfg["segment_seconds"])
        # 1. no loops: short beds / events / wind get extra distinct segments; bed 2 is a different file
        b2 = plan["bed2"]
        for _ in range(5):
            if b2["path"] != plan["layers"][0]["path"]:
                break
            b2.update(self._draw_file(b2["pool"], rng))
        for L in plan["layers"] + [b2] + ([plan["wind"]] if plan["wind"] else []):
            if L.get("kind") != "real_passby":
                self._extra_pieces(L, rng, seg)
        # 2. foreground events are foreground: a pass-by / approach is not 2-12 dB under the bed
        lo, hi = lg["foreground_event_db"]
        for L in plan["layers"][1:]:
            if L.get("kind") in ("passby", "approach", "real_passby"):
                L["rel_db"] = float(rng.uniform(lo, hi))
        # 3. impulses keep a minimum onset gap (two bursts never start on top of each other)
        imps = plan["bursts"] + plan["blasts"]
        gap = float(lg["impulse_gap_s"])
        placed: list[float] = []
        for ev in imps:
            for _ in range(20):
                if all(abs(ev["t0_s"] - t) >= gap for t in placed):
                    break
                ev["t0_s"] = float(rng.uniform(0.2, seg - 1.2))
            placed.append(ev["t0_s"])
        # 4. one outdoor "space" per clip: every tail shares a decay scale and is frequency-dependent
        tau = float(rng.uniform(*lg["tail_tau_scale"]))
        plan["space"] = {"tau_scale": tau, "banded": True}
        for b in plan["bursts"]:
            b["tail_tau_s"] *= tau
            b["banded"] = True
        for d in plan["blasts"]:
            d["tau_scale"], d["banded"] = tau, True
        # 5. textures: onset-aligned (render), not in the 0.4 s after a shot / blast, >= 0.15 s apart,
        #    and at most one hard negative per clip
        hn = lg["hardneg"]
        if rng.random() < float(hn["p"]):
            pool = self._pick_pool(hn["pools"], rng)
            plan["texture"].append({**self._draw_file(pool, rng), "offset": int(rng.integers(0, 10 ** 7)),
                                    "dur_s": float(rng.uniform(0.2, 0.8)), "t0_s": 0.0,
                                    "rel_db": float(rng.uniform(*hn["rel_db"])), "hardneg": True})
        quiet = [(t, t + float(lg["quiet_after_impulse_s"])) for t in placed]
        done: list[float] = []
        for tx in plan["texture"]:
            t0 = float(rng.uniform(0.0, seg - 0.3)) if tx.get("hardneg") else float(tx["t0_s"])
            for _ in range(30):
                if all(not (a <= t0 <= b) for a, b in quiet) and all(abs(t0 - d) >= float(lg["texture_gap_s"]) for d in done):
                    break
                t0 = float(rng.uniform(0.0, seg - 0.3))
            tx["t0_s"] = t0
            done.append(t0)
        # 6. buffets are wind on the capsule: none in a vehicle, at most `buffets_without_wind` handling bumps
        if float(scn.get("wind_p", 0.0)) == 0.0:
            plan["buffets"] = []
        elif not plan["wind"]:
            plan["buffets"] = plan["buffets"][: int(lg["buffets_without_wind"])]
        # 7. inside a vehicle, outside sound comes through the hull
        plan["interior"] = scn["name"] in lg["interior_scenarios"]

    # ---- render --------------------------------------------------------------
    def _layer_audio(self, L: dict[str, Any], offset_key: str = "offset") -> np.ndarray:
        x = self.pools.load(L)
        n = self.n
        if L.get("extra"):                      # v3.1: distinct segments crossfaded, never a tiled loop
            from .acoustics import concat_crossfade
            pieces = [x] + [self.pools.load(e) for e in L["extra"]]
            x = concat_crossfade(pieces, sum(len(q) for q in pieces) - int(0.6 * self.sr) * (len(pieces) - 1), 0.6, self.sr)
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
        sp = speech if speech is not None else load_mono(
            data_root() / plan["snippet"].get("root", cfg["speech"]["snippets"]) / plan["snippet"]["path"], sr)
        sp = sp[:n] if len(sp) >= n else np.pad(sp, (0, n - len(sp)))
        s = (sp * db_to_lin(plan["speech_level_db"] - active_rms_db(sp))).astype(np.float32)
        # 2-3. scene
        scene = np.zeros(n, dtype=np.float32)
        interior = bool(plan.get("interior"))
        from .acoustics import expander, hull_transmission, onset_crop
        layer_sigs: list[tuple[str, np.ndarray]] = []           # (label, signal before the SNR gain)
        for i, L in enumerate(plan["layers"]):
            y = self._layer_audio(L)
            if interior and i > 0:
                y = hull_transmission(y, sr)
                y = y / (db_to_lin(active_rms_db(y)) + EPS)
            scene += y * db_to_lin(L["rel_db"])
            layer_sigs.append((pool_label(L["pool"]), y * db_to_lin(L["rel_db"])))
        b2 = plan["bed2"]
        y = self._layer_audio(b2) * db_to_lin(b2["rel_db"])
        scene += y
        layer_sigs.append((pool_label(b2["pool"]), y))
        for tx in plan["texture"]:
            x = self.pools.load(tx)
            if cfg.get("logic"):                 # v3.1: start at a real onset, natural decay, own room tone removed
                seg = expander(onset_crop(x, tx["dur_s"], sr, search_from=int(tx["offset"]) % max(1, len(x) - sr // 5)), sr)
            else:
                seg = _fade(_crop_at(x, int(tx["dur_s"] * sr), tx["offset"]).copy(), int(0.02 * sr))
            tx_sig = np.zeros(n, dtype=np.float32)
            _place(tx_sig, _unit(seg) * db_to_lin(tx["rel_db"]), int(tx["t0_s"] * sr))
            scene += tx_sig
            tx["_sig"] = tx_sig
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
            if interior:
                w = hull_transmission(w, sr)
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
            if interior:
                w = hull_transmission(w, sr)
            t0 = int(d["t0_s"] * sr)
            ev = w * sp_peak * db_to_lin(d["ratio_db"])
            _place(imp, ev, t0)
            a, e = audible_span(ev[: n - t0], t0)
            events.append(Event(a, e, "blast", float(d["ratio_db"]), 0.0,
                                {"source": d["source"], **{k: float(v) for k, v in params.items()}}))
        for i_tx, tx in enumerate(plan["texture"]):
            a = int(tx["t0_s"] * sr)
            e = min(n, a + int(tx["dur_s"] * sr))
            if e > a:
                cat = "hard_negative" if not cfg.get("logic") or tx.get("hardneg") else pool_label(tx["pool"])
                if cat not in ("hard_negative", "footsteps_gear"):
                    cat = "hard_negative"
                events.append(Event(a, e, cat, float(tx["rel_db"]), 0.0, {"pool": tx["pool"], "tx_i": float(i_tx)}))
        noise_total = noise_boom + imp
        for ev in events:
            ev.local_snr_db = _local_snr_db(s, noise_total, ev.start, ev.end)
        if cfg.get("logic"):                        # audibility: an event > 20 dB under the rest of the mix is `masked`
            total = s + noise_total
            for ev in events:
                if "tx_i" in ev.params:
                    sig = plan["texture"][int(ev.params["tx_i"])]["_sig"] * g
                else:                                    # transient: the impulse track (overlaps are rare)
                    sig = imp
                a, e = ev.start, ev.end
                frac = audible_fraction(sig[a:e], total[a:e] - sig[a:e], sr)
                ev.params["audible_frac"] = frac
                ev.params["masked"] = bool(frac < float(cfg["logic"].get("audible_frac", 0.15)))
            for tx in plan["texture"]:
                tx.pop("_sig", None)
        # 6-7. boom channel; target = dry speech x AGC gain
        boom, gain, st = capsule_channel(s + noise_total, np.random.default_rng(plan["boom_seed"]), cfg["channel"], sr)
        clean = (s * gain).astype(np.float32)
        radio_info = None
        if plan.get("radio", {}).get("on"):     # tactical radio link on the input; target = link's linear filters only
            from .radio import radio_link
            rp_ = {k: v for k, v in cfg["radio"].items() if k != "p"}
            boom, clean, rmask, radio_info = radio_link(boom, clean, np.random.default_rng(plan["radio"]["seed"]), rp_)
            muted = np.where(rmask == 0)[0]
            radio_info["mute_span"] = [int(muted[0]), int(muted[-1]) + 1] if len(muted) else None
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
            "radio": radio_info,
        }
        if cfg.get("logic"):
            total = s + noise_total
            labs = set()
            layer_frac = {}
            for lab, sig in layer_sigs:                  # a layer counts if audible in >= 15 % of its active frames
                frac = audible_fraction(sig * g, total - sig * g, sr)
                layer_frac[lab] = max(layer_frac.get(lab, 0.0), frac)
                if frac >= float(cfg["logic"].get("audible_frac", 0.15)):
                    labs.add(lab)
            labs |= {ev.category for ev in events if not ev.params.get("masked")}
            meta["layer_audible_frac"] = layer_frac
            if plan["wind"]:
                labs.add("wind_mic")
            meta["labels"] = sorted(labs)
            meta["interior"] = interior
            meta["tail_tau_scale"] = plan["space"]["tau_scale"]
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
