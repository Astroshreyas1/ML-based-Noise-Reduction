"""Battlefield chain v4: "one space, few objects" (docs/NOISE_LAYERING.md section 8).

v3 sounded pasted on: up to ~20 dry objects from different recordings in 6 s,
each with its own room tone, mid-event crops, looped short beds and a
frequency-flat gunfire tail. v4 keeps the v3 speech, SNR, capsule channel and
target rule, and changes the scene:

     1. speech        as v3 (same seed: index k has the same snippet in v3 and v4)
     2. scenario      weighted draw -> environment (open_field / forest / urban / cabin)
     3. beds          1 bed + a different second bed with p 0.4; distinct segments crossfaded,
                      never looped; dry (they are already ambient recordings)
     4. foreground    count ~ P(0..3) = .15/.40/.35/.10; gunfire first with the scenario's p,
                      then labels from the scenario menu; each at a distance class (near / mid / far)
                      that sets level, DRR and air absorption together, propagated through
                      this clip's environment (acoustics.outdoor_ir, one position per source)
     5. schedule      <= 2 foreground events sounding at once, >= 0.25 s between onsets
     6. textures      0-2 (footsteps / gear, radio squelch, hard negatives, debris after a blast),
                      onset-aligned, own room tone expanded away, near-field IR, never within
                      0.4 s after a shot / blast
     7. budget        <= 6 distinct labels per clip (drop textures, then bed 2)
     8. wind          capsule wind (dry, independent per mic) + 0-3 buffets only with wind
     9. SNR, transients at a peak ratio, boom channel, target, ref, contract -- as v3

Every event is labelled from what was rendered (fine label, coarse group, role,
distance class, DRR) and marked `masked` when it is > 20 dB under the rest of
the mix in its own span.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .acoustics import (DISTANCE_CLASSES, air_absorb, concat_crossfade, draw_distance, draw_environment, expander,
                        hull_transmission, onset_crop, outdoor_ir, propagate, radio_squelch, random_biquad)
from .audio import EPS, active_rms_db, db_to_lin, load_mono, rms
from .battlefield import BattlefieldChain, _crop_at, _local_snr_db, _place
from .chain import Event, Pair, assert_contract
from .channel import capsule_channel, ref_channel
from .config import SR, child_rng, sample
from .loudness import gain_for_snr_lufs
from .paths import data_root
from .scene import apply_envelope
from .transients import audible_span, lowpass1, render_blast, render_burst

REQUIRED = ("name", "seed", "sample_rate", "segment_seconds", "channels", "target", "speech", "mix", "budget",
            "gunfire", "blasts", "labels", "levels", "wind", "scenarios", "channel", "ref", "pools", "build")
IMPULSE_COARSE = {"gunfire_semi": "weapons", "gunfire_auto": "weapons", "gunfire_distant": "weapons",
                  "blast_near": "explosion", "blast_distant": "explosion"}


def load_v4_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if int(cfg["sample_rate"]) != SR or list(cfg["channels"]) != ["boom", "ref"] or cfg["target"] != "dry_x_agc":
        raise ValueError(f"{path}: sample_rate / channels / target are locked (16000, [boom, ref], dry_x_agc)")
    for s in cfg["scenarios"]:
        for key in ("beds", "bed2"):
            for lab in (s[key] if isinstance(s[key], list) else list(s[key])):
                if cfg["labels"].get(lab, {}).get("role") != "bed":
                    raise ValueError(f"{path}: scenario {s['name']}: {lab!r} is not a bed label")
        for lab in list(s.get("foreground", {})) + list(s.get("texture", {})):
            if lab != "blast" and lab not in cfg["labels"]:
                raise ValueError(f"{path}: scenario {s['name']}: unknown label {lab!r}")
    if abs(sum(cfg["budget"]["foreground_count_p"]) - 1) > 1e-6 or abs(sum(cfg["budget"]["texture_count_p"]) - 1) > 1e-6:
        raise ValueError(f"{path}: budget probabilities must sum to 1")
    cfg["_path"] = str(path)
    return cfg


def _pick(weights: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(weights)
    p = np.array([float(weights[k]) for k in keys])
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _fade_rc(x: np.ndarray, n_in: int, n_out: int) -> np.ndarray:
    n_in, n_out = min(n_in, len(x) // 2), min(n_out, len(x) // 2)
    if n_in > 0:
        x[:n_in] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_in))).astype(np.float32)
    if n_out > 0:
        x[-n_out:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n_out))).astype(np.float32)
    return x


class BattlefieldChainV4(BattlefieldChain):
    """Same constructor, sizes and snippet order as v3; new plan / render."""

    # ---- plan helpers -----------------------------------------------------------
    def _usable(self, label: str) -> bool:
        spec = self.cfg["labels"].get(label, {})
        return bool(spec.get("synth")) or any(self._pool_ok.get(p) for p in spec.get("pools", []))

    def _file(self, pool: str, rng: np.random.Generator) -> dict[str, Any]:
        pp = pool in self.cfg["levels"].get("pp_only_pools", [])
        row = self.pools.draw(pool, rng, pp_only=pp)
        return {"pool": pool, "path": row["path"], "start_s": float(row["start_s"]), "end_s": float(row["end_s"]),
                "duration_s": float(row["duration_s"]), "shot_time": float(row["shot_time"]), "group": str(row["group"])}

    def _pieces(self, pools: list[str], seconds: float, rng: np.random.Generator, first: dict[str, Any] | None = None,
                cap: int = 8) -> list[dict[str, Any]]:
        """Enough *different* segments to cover `seconds` without looping one clip."""
        out = [first or self._file(self._pick_pool(pools, rng), rng)]
        pool = out[0]["pool"]
        total = out[0]["end_s"] - out[0]["start_s"]
        while total < seconds + 1.0 and len(out) < cap:
            f = self._file(pool, rng)
            out.append(f)
            total += f["end_s"] - f["start_s"] - 0.6
        return out

    def _bed(self, label: str, rel_db: float, rng: np.random.Generator) -> dict[str, Any]:
        spec = self.cfg["labels"][label]
        seg = float(self.cfg["segment_seconds"])
        pieces = self._pieces(spec["pools"], seg, rng)
        e = self.cfg["mix"]["bed_envelope"]
        return {"label": label, "role": "bed", "pieces": pieces, "rel_db": float(rel_db),
                "offset": int(rng.integers(0, 10 ** 8)), "env_seed": int(rng.integers(2 ** 31)),
                "gust_depth_db": sample(e["gust_depth_db"], rng), "gust_rate_hz": sample(e["gust_rate_hz"], rng),
                "bq_seed": int(rng.integers(2 ** 31))}

    def _dist(self, probs: dict[str, float], rng: np.random.Generator, far_only: bool) -> dict[str, Any]:
        cls = "far" if far_only and "far" in probs else _pick(probs, rng)
        return draw_distance(cls, rng)

    def _recurrent(self, label: str, rng: np.random.Generator, far_only: bool) -> dict[str, Any]:
        spec = self.cfg["labels"][label]
        pool = self._pick_pool(spec["pools"], rng)
        f = self._file(pool, rng)
        dist = self._dist(spec["distance_p"], rng, far_only)
        ev: dict[str, Any] = {"label": label, "role": "recurrent", "dist": dist,
                              "rel_db": sample(self.cfg["levels"]["recurrent"][dist["cls"]], rng),
                              "ir_seed": int(rng.integers(2 ** 31)), "bq_seed": int(rng.integers(2 ** 31))}
        if spec.get("real_passby"):
            ev.update(kind="real_passby", file=f, dur_s=min(2.0, f["end_s"] - f["start_s"]))
        elif spec.get("onset"):
            ev.update(kind="onset", file=f, dur_s=float(rng.uniform(*spec["dur_s"])), search_frac=float(rng.uniform(0, 0.6)))
        else:
            dur = float(rng.uniform(*spec["dur_s"]))
            ev.update(kind="segment", pieces=self._pieces([pool], dur, rng, first=f), dur_s=dur,
                      offset=int(rng.integers(0, 10 ** 8)), envelope=str(rng.choice(spec["envelopes"])),
                      env_seed=int(rng.integers(2 ** 31)))
        return ev

    def _gunfire(self, rng: np.random.Generator, far_only: bool) -> dict[str, Any]:
        g = self.cfg["gunfire"]
        auto = bool(rng.random() < float(g["auto_p"]))
        src = _pick(g["source_p"], rng)
        dist = self._dist(g["distance_p"], rng, far_only)
        if src == "mad" and dist["cls"] == "near":           # lossy multi-source YouTube: never a near source
            src = "physics"
        b: dict[str, Any] = {"role": "impulse", "kind": "gunfire", "auto": auto, "source": src, "dist": dist,
                             "n_rounds": int(sample(g["rounds_auto"] if auto else g["rounds_semi"], rng)),
                             "spacing_s": sample(g["spacing_auto_s"] if auto else g["spacing_semi_s"], rng),
                             "ratio_db": sample(g["peak_ratio_db"][dist["cls"]], rng),
                             "seed": int(rng.integers(2 ** 31)), "ir_seed": int(rng.integers(2 ** 31))}
        b["label"] = "gunfire_distant" if dist["cls"] == "far" else ("gunfire_auto" if auto else "gunfire_semi")
        if src == "field":
            b["file"] = self._file("field_gunshot", rng)
        elif src == "mad":
            b["file"] = self._file(self._pick_pool(["mad_gunshot"], rng), rng)
        b["dur_s"] = (b["n_rounds"] - 1) * b["spacing_s"] + 0.35 if src != "mad" else 1.5
        return b

    def _blast(self, rng: np.random.Generator, far_only: bool) -> dict[str, Any]:
        bl = self.cfg["blasts"]
        src = _pick(bl["source_p"], rng)
        cls = "far" if far_only else _pick(bl["distance_p"], rng)
        if src == "mad" and cls != "far":
            src = "physics"
        lo, hi = bl["standoff_m"][cls]
        dist = {"cls": cls, "m": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
                "drr_db": float(rng.uniform(*DISTANCE_CLASSES[cls]["drr_db"]))}
        d: dict[str, Any] = {"role": "impulse", "kind": "blast", "source": src, "dist": dist,
                             "label": "blast_near" if cls == "mid" else "blast_distant",
                             "ratio_db": sample(bl["peak_ratio_db"][cls], rng), "seed": int(rng.integers(2 ** 31)),
                             "ir_seed": int(rng.integers(2 ** 31)), "dur_s": 1.5}
        if src == "fsd":
            d["file"] = self._file(self._pick_pool(["fsd_explosion"], rng), rng)
        elif src == "mad":
            d["file"] = self._file(self._pick_pool(["mad_shelling"], rng), rng)
        return d

    def _texture(self, label: str, rng: np.random.Generator) -> dict[str, Any]:
        spec = self.cfg["labels"][label]
        t: dict[str, Any] = {"label": label, "role": "texture", "rel_db": sample(self.cfg["levels"]["texture"], rng),
                             "dist": {"cls": "near", "m": float(rng.uniform(1.0, 6.0)), "drr_db": float(rng.uniform(12, 20))},
                             "ir_seed": int(rng.integers(2 ** 31)), "bq_seed": int(rng.integers(2 ** 31)),
                             "seed": int(rng.integers(2 ** 31))}
        if spec.get("synth") == "squelch":
            t.update(kind="squelch", dur_s=0.4)
        else:
            t.update(kind="onset", file=self._file(self._pick_pool(spec["pools"], rng), rng),
                     dur_s=float(rng.uniform(*spec["dur_s"])), search_frac=float(rng.uniform(0, 0.6)))
        return t

    @staticmethod
    def _overlap_ok(t0: float, dur: float, placed: list[tuple[float, float]], max_overlap: int, gap: float, seg: float) -> bool:
        a, b = max(0.0, t0), min(seg, t0 + dur)
        for s, e in placed:
            if t0 >= 0 and s >= 0 and abs(s - t0) < gap:
                return False
        for t in np.arange(a, b, 0.05):
            if 1 + sum(1 for s, e in placed if s <= t < s + e) > max_overlap:
                return False
        return True

    # ---- plan -----------------------------------------------------------------
    def plan(self, index: int) -> dict[str, Any]:
        cfg = self.cfg
        bud = cfg["budget"]
        rng = child_rng(cfg["seed"], index)
        snip = self.snippets.iloc[index % self.n_snippets]
        scn = copy.deepcopy(cfg["scenarios"][int(rng.choice(len(self._scn_p), p=self._scn_p))])
        seg = float(cfg["segment_seconds"])
        far_only = bool(scn.get("far_only", False))
        plan: dict[str, Any] = {
            "version": 4, "index": int(index), "seed": int(cfg["seed"]), "split": self.split, "scenario": scn["name"],
            "snippet": {"id": snip.id, "path": snip.path, "corpus": snip.corpus, "speaker_id": snip.speaker_id,
                        "effort": snip.effort, "gender": snip.gender, "transcript": snip.transcript,
                        "speech_fraction": float(snip.speech_fraction)},
            "speech_level_db": sample(cfg["speech"]["level_db"], rng), "snr_db": sample(cfg["mix"]["snr_db"], rng),
            "env": draw_environment(_pick(scn["env"], rng), rng),
            "beds": [], "fg": [], "texture": [], "wind": None, "buffets": [],
        }
        plan["env"]["outside_rt60_s"] = float(rng.uniform(0.15, 0.4))      # cabin: what the exterior path sees
        # beds
        beds = {k: v for k, v in scn["beds"].items() if self._usable(k)}
        bed = _pick(beds, rng)
        plan["beds"].append(self._bed(bed, 0.0, rng))
        if rng.random() < float(cfg["mix"]["bed2_p"]):
            opts = [b for b in scn["bed2"] if b != bed and self._usable(b)]
            if opts:
                plan["beds"].append(self._bed(opts[int(rng.integers(len(opts)))], sample(cfg["mix"]["bed2_rel_db"], rng), rng))
        # foreground: gunfire first, then the scenario menu (blast may repeat once)
        k = int(rng.choice(len(bud["foreground_count_p"]), p=bud["foreground_count_p"]))
        kinds: list[str] = []
        if rng.random() < float(scn.get("gunfire_p", 0.0)):
            kinds += ["gunfire"] * (2 if rng.random() < float(cfg["gunfire"]["second_burst_p"]) else 1)
        k = max(k, len(kinds))
        menu = {lab: w for lab, w in scn.get("foreground", {}).items() if lab == "blast" or self._usable(lab)}
        while len(kinds) < k and menu:
            lab = _pick(menu, rng)
            kinds.append(lab)
            if lab != "blast" or kinds.count("blast") >= 2:
                menu.pop(lab)
        placed: list[tuple[float, float]] = []
        for kind in kinds:
            ev = (self._gunfire(rng, far_only) if kind == "gunfire" else
                  self._blast(rng, far_only) if kind == "blast" else self._recurrent(kind, rng, far_only))
            dur = float(ev["dur_s"])
            for _ in range(int(bud["place_tries"])):
                if ev["role"] == "impulse" or ev.get("kind") == "onset":
                    t0 = float(rng.uniform(0.1, max(0.2, seg - min(dur, seg * 0.5) - 0.3)))
                else:
                    t0 = float(rng.uniform(-0.4 * dur, max(-0.4 * dur + 0.1, seg - 0.6 * dur)))
                if self._overlap_ok(t0, dur, placed, int(bud["max_overlap"]), float(bud["min_onset_gap_s"]), seg):
                    ev["t0_s"] = t0
                    placed.append((t0, dur))
                    plan["fg"].append(ev)
                    break
        # causal follow-on: debris after a near blast
        bl = cfg["blasts"]
        for ev in list(plan["fg"]):
            if ev["kind"] == "blast" and ev["label"] == "blast_near" and self._usable("debris_impact") \
                    and rng.random() < float(bl["debris_p"]):
                t = self._texture("debris_impact", rng)
                t["t0_s"] = min(seg - 0.3, ev["t0_s"] + sample(bl["debris_delay_s"], rng))
                t["rel_db"] = sample(bl["debris_rel_db"], rng)
                t["follows"] = ev["label"]
                plan["texture"].append(t)
        # textures, away from impulse onsets and from each other
        n_tx = int(rng.choice(len(bud["texture_count_p"]), p=bud["texture_count_p"]))
        tmenu = {lab: w for lab, w in scn.get("texture", {}).items() if self._usable(lab)}
        quiet = [(ev["t0_s"], ev["t0_s"] + float(bud["quiet_after_impulse_s"])) for ev in plan["fg"] if ev["role"] == "impulse"]
        for _ in range(n_tx if tmenu else 0):
            t = self._texture(_pick(tmenu, rng), rng)
            for _ in range(int(bud["place_tries"])):
                t0 = float(rng.uniform(0.0, seg - 0.3))
                busy = quiet + [(x["t0_s"], x["t0_s"] + x["dur_s"]) for x in plan["texture"]]
                if all(not (a - 0.05 <= t0 <= b) for a, b in busy):
                    t["t0_s"] = t0
                    plan["texture"].append(t)
                    break
        # wind at the capsule (+ buffets only with wind)
        w = cfg["wind"]
        if rng.random() < float(scn.get("wind_p", 0.0)):
            pool = self._pick_pool(w["pools"], rng)
            plan["wind"] = {**self._file(pool, rng), "offset_boom": int(rng.integers(0, 10 ** 8)),
                            "offset_ref": int(rng.integers(0, 10 ** 8)), "rel_db": sample(w["rel_db"], rng),
                            "kind": "gust", "env_seed": int(rng.integers(2 ** 31)),
                            "depth_db": sample(w["gust_depth_db"], rng), "rate_hz": sample(w["gust_rate_hz"], rng)}
            bf = w["buffets"]
            for _ in range(int(sample(bf["count"], rng))):
                plan["buffets"].append({"t0_s": float(rng.uniform(0.0, seg - 0.4)), "dur_s": sample(bf["dur_s"], rng),
                                        "rel_db": sample(bf["rel_db"], rng), "seed": int(rng.integers(2 ** 31))})
        # label budget: drop plain textures, then bed 2, then the last non-gunfire foreground event
        while len(self.plan_labels(plan)) > int(bud["max_labels"]):
            plain = [i for i, t in enumerate(plan["texture"]) if "follows" not in t]
            if plain:
                plan["texture"].pop(plain[-1])
            elif len(plan["beds"]) > 1:
                plan["beds"].pop()
            else:
                other = [i for i, e in enumerate(plan["fg"]) if e.get("kind") != "gunfire"]
                if not other:
                    break
                gone = plan["fg"].pop(other[-1])
                plan["texture"] = [t for t in plan["texture"] if not (t.get("follows") and gone.get("kind") == "blast")]
        r = cfg["ref"]
        plan["ref"] = {"speech_db": sample(r["speech_db"], rng), "noise_db": sample(r["noise_db"], rng)}
        plan["boom_seed"] = int(rng.integers(2 ** 31))
        plan["ref_seed"] = int(rng.integers(2 ** 31))
        return plan

    @staticmethod
    def plan_labels(plan: dict[str, Any]) -> list[str]:
        labs = [b["label"] for b in plan["beds"]] + [e["label"] for e in plan["fg"]] + [t["label"] for t in plan["texture"]]
        if plan.get("wind"):
            labs.append("wind_mic")
        return sorted(set(labs))

    # ---- render helpers ------------------------------------------------------
    def _env_path(self, x: np.ndarray, ev: dict[str, Any], env: dict[str, Any], rng: np.random.Generator,
                  absorb: bool = True, ground: bool | None = None) -> np.ndarray:
        """Propagate one source to the listener. In a cabin, sources outside (everything but
        textures) cross an open-field path and the hull; textures sit inside the cabin."""
        if env["kind"] == "cabin" and ev["role"] != "texture":
            outside = {"kind": "open_field", "rt60_s": env["outside_rt60_s"]}
            y = propagate(x, outside, ev["dist"], rng, self.sr, absorb, ground)
            return hull_transmission(y, self.sr)
        return propagate(x, env, ev["dist"], rng, self.sr, absorb, ground)

    def _bed_audio(self, b: dict[str, Any]) -> np.ndarray:
        pieces = [self.pools.load(p) for p in b["pieces"]]
        r = np.random.default_rng(b["offset"] % (2 ** 31))
        pieces = [p[int(r.integers(0, max(1, len(p) - self.n))):] if len(p) > self.n else p for p in pieces]
        y = concat_crossfade(pieces, self.n, float(self.cfg["mix"]["bed_xfade_s"]), self.sr)
        y, _ = apply_envelope(y, "gust", np.random.default_rng(b["env_seed"]),
                              {"gust_depth_db": b["gust_depth_db"], "gust_rate_hz": b["gust_rate_hz"]})
        y = random_biquad(y, np.random.default_rng(b["bq_seed"]), float(self.cfg["augment"]["biquad_r"]))
        return (y / (db_to_lin(active_rms_db(y)) + EPS)).astype(np.float32)

    def _recurrent_audio(self, ev: dict[str, Any], env: dict[str, Any]) -> np.ndarray:
        sr = self.sr
        m = int(ev["dur_s"] * sr)
        if ev["kind"] == "real_passby":
            y = self.pools.load(ev["file"])[:m].copy()
            y = _fade_rc(y, int(0.05 * sr), int(0.05 * sr))
        elif ev["kind"] == "onset":
            x = self.pools.load(ev["file"])
            y = onset_crop(x, ev["dur_s"], sr, search_from=int(ev["search_frac"] * len(x)), release_s=0.2)
        else:
            pieces = [self.pools.load(p) for p in ev["pieces"]]
            long = concat_crossfade(pieces, m + int(0.5 * sr), 0.3, sr)
            y = _crop_at(long, m, ev["offset"] % max(1, int(0.5 * sr)))
            y, _ = apply_envelope(y, ev["envelope"], np.random.default_rng(ev["env_seed"]),
                                  {"ramp_db": {"dist": "uniform", "low": 6, "high": 15},
                                   "passby_peak_db": {"dist": "uniform", "low": 6, "high": 12}, "doppler_max_pct": 2.0})
            fade = int(min(0.8, 0.2 * ev["dur_s"]) * sr)
            y = _fade_rc(np.asarray(y, dtype=np.float32).copy(), fade, fade)
        y = random_biquad(y, np.random.default_rng(ev["bq_seed"]), float(self.cfg["augment"]["biquad_r"]))
        return self._env_path(y, ev, env, np.random.default_rng(ev["ir_seed"]))

    def _texture_audio(self, t: dict[str, Any], env: dict[str, Any]) -> np.ndarray:
        sr = self.sr
        if t["kind"] == "squelch":
            y = radio_squelch(np.random.default_rng(t["seed"]), sr)
        else:
            x = self.pools.load(t["file"])
            y = expander(onset_crop(x, t["dur_s"], sr, search_from=int(t["search_frac"] * len(x))), sr)
        y = random_biquad(y, np.random.default_rng(t["bq_seed"]), float(self.cfg["augment"]["biquad_r"]))
        return self._env_path(y, t, env, np.random.default_rng(t["ir_seed"]))

    def _impulse_audio(self, ev: dict[str, Any], env: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
        sr = self.sr
        rng = np.random.default_rng(ev["seed"])
        if ev["kind"] == "gunfire":
            spec = {k: ev[k] for k in ("source", "n_rounds", "spacing_s")}
            if ev["source"] == "physics":
                spec["standoff_m"] = ev["dist"]["m"]
            else:
                spec["file"] = str(data_root() / ev["file"]["path"])
                spec["t_shot"] = ev["file"]["shot_time"]
            w, params = render_burst(spec, rng, sr)
        else:
            spec = {"source": "physics" if ev["source"] == "physics" else "mad", "tail": False}
            if ev["source"] == "physics":
                spec["standoff_m"] = ev["dist"]["m"]
            else:
                spec["file"] = str(data_root() / ev["file"]["path"])
            w, params = render_blast(spec, rng, sr)
        physics = ev["source"] == "physics"
        if not physics:
            w = expander(w, sr)
        w = self._env_path(w, ev, env, np.random.default_rng(ev["ir_seed"]), absorb=not physics,
                           ground=None if not physics else False)
        return (w / (np.abs(w).max() + EPS)).astype(np.float32), params

    # ---- render ---------------------------------------------------------------
    def render(self, plan: dict[str, Any], speech: np.ndarray | None = None) -> Pair:
        cfg, n, sr = self.cfg, self.n, self.sr
        env = plan["env"]
        sp = speech if speech is not None else load_mono(data_root() / cfg["speech"]["snippets"] / plan["snippet"]["path"], sr)
        sp = sp[:n] if len(sp) >= n else np.pad(sp, (0, n - len(sp)))
        s = (sp * db_to_lin(plan["speech_level_db"] - active_rms_db(sp))).astype(np.float32)
        parts: list[tuple[dict[str, Any], np.ndarray]] = []          # (plan item, placed full-length signal)
        # beds
        for b in plan["beds"]:
            parts.append((b, self._bed_audio(b) * db_to_lin(b["rel_db"])))
        # recurrent foreground and textures (continuous-scene loudness includes them)
        for ev in plan["fg"]:
            if ev["role"] != "recurrent":
                continue
            y = self._recurrent_audio(ev, env)
            y = y / (db_to_lin(active_rms_db(y)) + EPS) * db_to_lin(ev["rel_db"])
            out = np.zeros(n, dtype=np.float32)
            t0 = int(ev["t0_s"] * sr)
            if t0 < 0:
                y, t0 = y[-t0:], 0
            _place(out, y, t0)
            parts.append((ev, out))
        for t in plan["texture"]:
            y = self._texture_audio(t, env)
            y = y / (db_to_lin(active_rms_db(y)) + EPS) * db_to_lin(t["rel_db"])
            out = np.zeros(n, dtype=np.float32)
            _place(out, y, int(t["t0_s"] * sr))
            parts.append((t, out))
        scene = np.sum([p for _, p in parts], axis=0).astype(np.float32)
        buf = np.zeros(n, dtype=np.float32)
        for bf in plan["buffets"]:
            r = np.random.default_rng(bf["seed"])
            m = int(bf["dur_s"] * sr)
            seg = lowpass1(lowpass1(r.standard_normal(m).astype(np.float32), 150.0, sr), 150.0, sr)
            seg *= np.hanning(m).astype(np.float32) ** 0.5
            _place(buf, seg / (rms(seg) + EPS) * db_to_lin(bf["rel_db"]), int(bf["t0_s"] * sr))
        wind_boom = wind_ref = np.zeros(n, dtype=np.float32)
        if plan["wind"]:
            w = plan["wind"]
            wind_boom = self._layer_audio(w, "offset_boom") * db_to_lin(w["rel_db"])
            wind_ref = self._layer_audio({**w, "env_seed": w["env_seed"] + 1}, "offset_ref") * db_to_lin(w["rel_db"])
        scene_boom = scene + buf + wind_boom
        scene_ref = scene + 0.5 * buf + wind_ref
        g = gain_for_snr_lufs(s, scene_boom, plan["snr_db"], sr)
        noise_boom = (scene_boom * g).astype(np.float32)
        noise_ref = (scene_ref * g).astype(np.float32)
        # impulses at a peak ratio vs the speech peak
        sp_peak = float(np.abs(s).max()) + EPS
        imp = np.zeros(n, dtype=np.float32)
        imp_parts: list[tuple[dict[str, Any], np.ndarray, dict[str, float]]] = []
        for ev in plan["fg"]:
            if ev["role"] != "impulse":
                continue
            w, params = self._impulse_audio(ev, env)
            out = np.zeros(n, dtype=np.float32)
            _place(out, w * sp_peak * db_to_lin(ev["ratio_db"]), int(ev["t0_s"] * sr))
            imp += out
            imp_parts.append((ev, out, params))
        noise_total = noise_boom + imp
        # labels, from what was rendered
        events: list[Event] = []
        thr = float(cfg.get("label_audible_snr_db", -20.0))
        total = s + noise_total

        def add(item: dict[str, Any], sig: np.ndarray, ratio: float, extra: dict[str, Any]) -> None:
            if item["role"] == "bed":
                a, e = 0, n
            else:
                t0 = max(0, int(item.get("t0_s", 0.0) * sr))
                if not np.any(sig[t0:]):
                    return
                a, e = audible_span(sig[t0:], t0, -40.0 if item["role"] != "recurrent" else -30.0)
            if e <= a:
                return
            rest = total[a:e] - sig[a:e]
            local = float(10 * np.log10((np.sum(sig[a:e] ** 2) + EPS) / (np.sum(rest ** 2) + EPS)))
            lab = item["label"]
            coarse = IMPULSE_COARSE.get(lab) or cfg["labels"].get(lab, {}).get("coarse", "other")
            d = item.get("dist") or {}
            events.append(Event(a, e, lab, float(ratio), _local_snr_db(s, noise_total, a, e),
                                {"coarse": coarse, "role": item["role"], "masked": bool(local < thr),
                                 "event_to_rest_db": local, "dist_cls": d.get("cls", "dry"), "dist_m": float(d.get("m", 0.0)),
                                 "drr_db": float(d.get("drr_db", 99.0)), "env": env["kind"],
                                 "source": item.get("source") or (item.get("file") or {}).get("pool")
                                 or (item["pieces"][0]["pool"] if item.get("pieces") else item.get("kind", "")),
                                 **extra}))

        for item, sig in parts:
            add(item, sig * g, float(item.get("rel_db", 0.0)), {})
        for item, sig, params in imp_parts:
            add(item, sig, float(item["ratio_db"]), {"n_rounds": float(item.get("n_rounds", 1)),
                                                     **{k: float(v) for k, v in params.items()}})
        if plan["wind"]:
            events.append(Event(0, n, "wind_mic", float(plan["wind"]["rel_db"]), _local_snr_db(s, noise_total, 0, n),
                                {"coarse": "channel", "role": "channel", "masked": False, "env": env["kind"]}))
        # boom channel; target = dry speech x AGC gain; ref
        boom, gain, st = capsule_channel(s + noise_total, np.random.default_rng(plan["boom_seed"]), cfg["channel"], sr)
        clean = (s * gain).astype(np.float32)
        rp = plan["ref"]
        ref_in = s * db_to_lin(rp["speech_db"]) + (noise_ref + imp) * db_to_lin(rp["noise_db"])
        ref, st_ref = ref_channel(ref_in, np.random.default_rng(plan["ref_seed"]), cfg["ref"]["channel"], sr)
        labels = sorted({e.category for e in events if not e.params.get("masked")})
        meta = {
            "version": 4, "index": plan["index"], "seed": plan["seed"], "split": plan["split"], "scenario": plan["scenario"],
            **{f"speech_{k}": v for k, v in plan["snippet"].items()},
            "speech_level_db": plan["speech_level_db"], "snr_lufs_db": plan["snr_db"],
            "snr_effective_db": float(active_rms_db(s) - active_rms_db(noise_boom)),
            "environment": env["kind"], "rt60_s": env["rt60_s"], "labels": labels, "n_labels": len(labels),
            "n_masked": sum(1 for e in events if e.params.get("masked")),
            "beds": [{"label": b["label"], "pools": [p["pool"] for p in b["pieces"]], "paths": [p["path"] for p in b["pieces"]],
                      "rel_db": b["rel_db"]} for b in plan["beds"]],
            "foreground": [{"label": e["label"], "role": e["role"], "dist_cls": e["dist"]["cls"], "t0_s": e["t0_s"],
                            "path": (e.get("file") or (e.get("pieces") or [{}])[0]).get("path", "")} for e in plan["fg"]],
            "textures": [{"label": t["label"], "t0_s": t["t0_s"], "path": (t.get("file") or {}).get("path", "")}
                         for t in plan["texture"]],
            "wind": None if not plan["wind"] else {k: plan["wind"][k] for k in ("pool", "path", "rel_db")},
            "n_buffets": len(plan["buffets"]),
            "gunfire": any(e.get("kind") == "gunfire" and e["label"] != "gunfire_distant" for e in plan["fg"]),
            "any_gunfire": any(e.get("kind") == "gunfire" for e in plan["fg"]),
            "n_bursts": sum(1 for e in plan["fg"] if e.get("kind") == "gunfire"),
            "n_blasts": sum(1 for e in plan["fg"] if e.get("kind") == "blast"),
            "agc_min_gain_db": st["agc_min_gain_db"], "clip_pct": st["clip_pct"],
            "ref_speech_db": rp["speech_db"], "ref_noise_db": rp["noise_db"], "ref_clip_pct": st_ref["clip_pct"],
        }
        if cfg.get("debug_stems"):
            meta["_stems"] = {"speech_dry": s, "scene": noise_boom, "transients": imp, "agc_gain": gain}
        pair = Pair(np.stack([boom, ref]).astype(np.float32), clean, events, meta)
        assert_contract(pair)
        return pair


def build_battlefield_v4_chain(cfg: dict[str, Any] | str | Path, split: str, cache_mb: float | None = None) -> BattlefieldChainV4:
    if not isinstance(cfg, dict):
        cfg = load_v4_config(cfg)
    if cache_mb is not None:
        cfg = copy.deepcopy(cfg)
        cfg["pools"]["cache_mb"] = cache_mb
    return BattlefieldChainV4(cfg, split)
