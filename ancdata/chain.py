"""The generation chain: one (noisy, clean) pair per call, fully seeded.

Order of operations is physics, not preference::

     1. draw clean speech segment              reject if near-silent; short utterances padded, never looped
     2. draw the scene SNR; Lombard strength   s = dose-response of the noise level (louder scene, more effort)
        Lombard transform (WORLD: F0, tilt, F1, duration, periodicity) on the speech itself
     3. >>> TARGET = dry segment <<<           level U(speech_level_db), captured BEFORE room, mic, noise
     4. draw a room from the RIR bank          four RIRs: speech/noise x boom/ref, direct path at t=0
     5. speech -> boom & ref channels          boom nearly dry (+ proximity low shelf), ref far-field
     6. noise SCENE: 1-3 concurrent layers     each with its own level and time envelope (gust / approach /
        through the SAME room                  recede / pass-by / steady); scene scaled to the SNR vs TARGET
     7. capsule-local layers (wind) stay dry   uncorrelated between the two mics
     8. impulsive events, scaled by PEAK ratio vs speech peak (global SNR is meaningless
        for a 200 ms event in a 4 s clip); spans recorded as labels
     9. hard negatives, same scaling            so "loud broadband" alone is not the cue
    10. ADC: random gain -> hard clip -> 16-bit  applied to input only, same gain both channels
    11. radio codec on the boom channel         optional
    12. contract asserts

Step 3 defines the task: target = the dry speech the talker produced. Lombard
sits BEFORE the target on purpose: time-stretch and F0 shift on the input
alone would leave a target the network cannot reach with a mask (a mask cannot
move harmonics or time), so the transform is speech diversity, not a distortion
to undo. Room, microphone, noise, transients, ADC and codec are the input-side
conditions. Any linear gain the ADC / safety stage applies to the mixture is
applied to the target too, so target level always tracks the input speech
level. There is no post-mix normalisation (it would undo the clipping).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, fftconvolve, sosfilt

from .adc import radio_codec, simulate_adc
from .audio import EPS, active_rms_db, normalise_rms, peak, place_or_crop, rms, tile_or_crop, trim_silence
from .config import SEG_SAMPLES, SR, child_rng, sample
from .lombard import load_stats, lombard
from .physics import synth_blast, synth_rotor
from .realcheck import apply_mic_ir, load_mic_ir
from .scene import compose_scene
from .rir_gen import RirBank
from .sampler import Sampler

TARGET_RMS_DB = -25.0
MAX_SPEECH_TRIES = 50


@dataclass
class Event:
    start: int
    end: int
    category: str
    ratio_db: float
    local_snr_db: float
    params: dict[str, float] = field(default_factory=dict)

    def as_tuple(self) -> tuple[int, int, str, float]:
        return (self.start, self.end, self.category, self.ratio_db)


@dataclass
class Pair:
    noisy: np.ndarray            # (2, T) float32: [boom, ref]
    clean: np.ndarray            # (T,) float32, dry target, sample-aligned with noisy[0]
    events: list[Event]
    meta: dict[str, Any]


def assert_contract(pair: Pair) -> None:
    n = pair.noisy
    c = pair.clean
    assert n.ndim == 2 and n.shape[0] == 2, f"noisy shape {n.shape}"
    assert c.ndim == 1 and c.shape[0] == n.shape[1], f"clean shape {c.shape} vs noisy {n.shape}"
    assert n.dtype == np.float32 and c.dtype == np.float32, (n.dtype, c.dtype)
    assert np.isfinite(n).all() and np.isfinite(c).all(), "non-finite samples"
    assert np.abs(c).max() > 1e-4, "target is silence"
    assert np.abs(n).max() <= 1.0 + 1e-6, f"noisy exceeds full scale: {np.abs(n).max()}"
    for ev in pair.events:
        assert 0 <= ev.start < ev.end <= n.shape[1], f"bad event span {ev}"


def _conv_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal convolution truncated to len(x): output sample i depends only on x[:i+1]."""
    y = fftconvolve(x, h, mode="full")[: len(x)]
    return y.astype(np.float32)


def _low_shelf(x: np.ndarray, gain_db: float, f_hz: float, sr: int = SR) -> np.ndarray:
    """First-order low shelf: +gain_db below f_hz (proximity effect)."""
    if abs(gain_db) < 1e-3:
        return x
    sos = butter(1, f_hz, btype="low", fs=sr, output="sos")
    lo = sosfilt(sos, x)
    return (x + (10.0 ** (gain_db / 20.0) - 1.0) * lo).astype(np.float32)


def _local_snr_db(speech: np.ndarray, noise_added: np.ndarray, start: int, end: int) -> float:
    s = speech[start:end]
    n = noise_added[start:end]
    return float(10.0 * np.log10((np.sum(s ** 2) + EPS) / (np.sum(n ** 2) + EPS)))


def snr_spec_for_index(mix_cfg: dict[str, Any], index: int) -> Any:
    """SNR curriculum. ``mix_ambience.schedule`` is an ordered list of
    ``{until: <example index or null>, snr_db: <spec>}``; the first stage whose
    ``until`` exceeds the example index is used. Without a schedule the plain
    ``snr_db`` spec applies. Because the stage is a function of the example
    index, the curriculum is reproducible and worker-independent."""
    sched = mix_cfg.get("schedule")
    if not sched:
        return mix_cfg["snr_db"]
    for stage in sched:
        until = stage.get("until")
        if until is None or index < int(until):
            return stage["snr_db"]
    return sched[-1]["snr_db"]


class Chain:
    def __init__(self, cfg: dict[str, Any], sampler: Sampler, bank: RirBank,
                 rir_split: str | None = None):
        self.cfg = cfg
        self.sampler = sampler
        self.bank = bank
        self.rir_split = rir_split or ("test" if sampler.split == "test" else "train")
        self.lombard_stats = load_stats(cfg["lombard"].get("stats_file"))
        self.dry_categories = set(cfg["reverb"].get("dry_categories", ["wind"]))
        self.mic_ir = load_mic_ir(cfg.get("mic", {}).get("ir_file"))  # measured boom-mic response, optional

    # -------------------------------------------------------------- stages
    def _draw_speech(self, rng: np.random.Generator) -> tuple[np.ndarray, pd.Series]:
        for _ in range(MAX_SPEECH_TRIES):
            _, row = self.sampler.draw_row("speech", rng)
            audio = self.sampler.audio(row)
            if len(audio) < SEG_SAMPLES // 2:
                continue
            seg = place_or_crop(audio, SEG_SAMPLES, rng)
            if active_rms_db(seg) > float(self.cfg.get("speech_min_active_db", -35.0)):
                return seg.astype(np.float32), row
        raise RuntimeError("could not draw an active speech segment; check the speech pile")

    def _draw_ambience(self, rng: np.random.Generator) -> tuple[np.ndarray, str, dict[str, Any]]:
        entry, row = self.sampler.draw_row("ambience", rng)
        if row is None:
            gen = entry["generator"]
            if gen == "rotor":
                audio, meta = synth_rotor(rng, SEG_SAMPLES, params=entry.get("params"))
                return audio, "synthetic_rotor", meta
            raise ValueError(f"unknown ambience generator {gen!r}")
        audio = trim_silence(self.sampler.audio(row))
        return audio.astype(np.float32), str(row.category), {"path": str(row.path), "pile": str(row.pile)}

    def _draw_impulsive(self, rng: np.random.Generator) -> tuple[np.ndarray, str, dict[str, float]]:
        entry, row = self.sampler.draw_row("impulsive", rng)
        if row is None:
            if entry["generator"] != "friedlander":
                raise ValueError(f"unknown impulsive generator {entry['generator']!r}")
            audio, meta = synth_blast(rng, params=entry.get("params"))
            return audio, "blast_synthetic", meta
        audio = trim_silence(self.sampler.audio(row))
        audio = audio / (peak(audio))
        return audio.astype(np.float32), str(row.category), {}

    def _inject(self, boom: np.ndarray, ref: np.ndarray, rirs: np.ndarray, ev: np.ndarray,
                category: str, ratio_db: float, speech_boom: np.ndarray, rng: np.random.Generator,
                through_room: bool, params: dict[str, float]) -> Event:
        """Add one transient to both channels at a peak ratio relative to the
        speech peak on the boom channel. Returns the span as a label."""
        n = len(boom)
        if len(ev) >= n:
            ev = ev[: n // 2]
        if through_room:
            ev_b = fftconvolve(ev, rirs[2], mode="full").astype(np.float32)
            ev_r = fftconvolve(ev, rirs[3], mode="full").astype(np.float32)
            keep = min(len(ev_b), n)
            ev_b, ev_r = ev_b[:keep], ev_r[:keep]
        else:
            ev_b, ev_r = ev.astype(np.float32), ev.astype(np.float32)
        # trim leading silence and the reverberant tail so the span label covers audible energy only
        pk_b = peak(ev_b)
        idx = np.where(np.abs(ev_b) > 0.01 * pk_b)[0]
        first, last = int(idx[0]), int(idx[-1]) + 1
        ev_b, ev_r = ev_b[first:last], ev_r[first:last]
        target_peak = peak(speech_boom) * (10.0 ** (ratio_db / 20.0))
        g = target_peak / pk_b
        ev_b, ev_r = ev_b * g, ev_r * g
        start = int(rng.integers(0, n - len(ev_b) + 1))
        end = start + len(ev_b)
        boom[start:end] += ev_b
        ref[start:end] += ev_r
        local = _local_snr_db(speech_boom, np.pad(ev_b, (start, n - end)), start, end)
        return Event(start, end, category, float(ratio_db), local, params)

    # ---------------------------------------------------------------- main
    def generate(self, index: int) -> Pair:
        cfg = self.cfg
        rng = child_rng(cfg["seed"], index)
        meta: dict[str, Any] = {"index": int(index), "seed": int(cfg["seed"]), "split": self.sampler.split}

        # 1. speech
        seg, srow = self._draw_speech(rng)
        meta["speaker_id"] = str(srow.speaker_id)
        meta["speech_path"] = str(srow.path)

        # 2. scene SNR first, then Lombard strength as a dose-response of it (Lombard slope:
        #    lower SNR <=> louder ambient <=> more vocal effort), then the transform itself
        snr_db = sample(snr_spec_for_index(cfg["mix_ambience"], index), rng)
        lom = cfg["lombard"]
        if rng.random() < float(lom["probability"]):
            c = lom.get("coupling", {})
            strength = float(c.get("base", 0.3)) + (float(c.get("snr_ref_db", 5.0)) - snr_db) / float(c.get("snr_span_db", 25.0))
            strength += float(rng.normal(0.0, float(c.get("jitter", 0.15))))
            strength = float(np.clip(strength, float(c.get("min", 0.15)), 1.0))
            seg, lp = lombard(seg, rng, self.lombard_stats, strength=strength)
            seg = place_or_crop(seg, SEG_SAMPLES, rng) if len(seg) != SEG_SAMPLES else seg
            meta["lombard"] = lp

        # 3. TARGET: dry speech at a random level
        level_db = sample(cfg.get("speech_level_db", TARGET_RMS_DB), rng)
        clean = normalise_rms(seg, level_db)
        meta["speech_level_db"] = float(level_db)
        x = clean

        # 4-5. room and speech channels
        room_idx, rirs, rmeta = self.bank.draw(rng, self.rir_split)
        meta["room"] = {"index": room_idx, **rmeta}
        use_reverb = rng.random() < float(cfg["reverb"]["probability"])
        meta["reverb"] = bool(use_reverb)
        if use_reverb:
            sp_boom = _conv_same(x, rirs[0])
            sp_ref = _conv_same(x, rirs[1])
        else:
            sp_boom = x.copy()
            sp_ref = x * float(10.0 ** (-sample(cfg["reverb"].get("dry_ref_atten_db", 12.0), rng) / 20.0))
        # proximity effect: a directional boom mic 2-3 cm from the mouth boosts the lows of the
        # NEAR source only (far-field noise is unaffected); skipped when a measured IR exists
        prox = cfg.get("mic", {}).get("proximity_boost_db")
        if prox is not None and self.mic_ir is None:
            boost = sample(prox, rng)
            sp_boom = _low_shelf(sp_boom, boost, float(cfg["mic"].get("proximity_hz", 150.0)))
            meta["proximity_boost_db"] = float(boost)
        boom = sp_boom.copy()
        ref = sp_ref.copy()

        # 6-7. noise scene: 1-3 layers, each enveloped, through the room unless capsule-local
        layers = compose_scene(self._draw_ambience, rng, cfg.get("scene", {}), SEG_SAMPLES, self.dry_categories)
        n_boom = np.zeros(SEG_SAMPLES, dtype=np.float32)
        n_ref = np.zeros(SEG_SAMPLES, dtype=np.float32)
        stems_boom: list[np.ndarray] = []
        for L in layers:
            if L.dry or not use_reverb:
                lb, lr = L.audio_boom, L.audio_ref
            else:
                lb, lr = _conv_same(L.audio_boom, rirs[2]), _conv_same(L.audio_ref, rirs[3])
            n_boom += lb
            n_ref += lr
            stems_boom.append(lb)
        scale = (rms(clean) / (rms(n_boom) + EPS)) * (10.0 ** (-snr_db / 20.0))
        boom += n_boom * scale
        ref += n_ref * scale
        meta.update({
            "snr_db": float(snr_db),
            "ambience": layers[0].category,                       # dominant layer (kept for per-category tables)
            "ambience_meta": layers[0].meta,
            "ambience_dry": bool(layers[0].dry),
            "scene": [{"category": L.category, "envelope": L.envelope, "rel_db": L.rel_db, "dry": L.dry,
                       **{k: v for k, v in L.meta.items() if k not in ("path", "pile")}} for L in layers],
        })
        if cfg.get("debug_stems"):
            meta["_stems"] = {"speech_boom": sp_boom.copy(),
                              **{f"layer{i}_{L.category}": st * scale for i, (L, st) in enumerate(zip(layers, stems_boom))}}

        # 8. impulsive
        events: list[Event] = []
        pre_events = boom.copy() if cfg.get("debug_stems") else None
        imp = cfg["impulsive"]
        if self.sampler.has("impulsive") and rng.random() < float(imp["probability"]):
            count = int(sample(imp.get("count", 1), rng))
            for _ in range(count):
                ev, cat, p = self._draw_impulsive(rng)
                ratio = sample(imp["peak_ratio_db"], rng)
                events.append(self._inject(boom, ref, rirs, ev, cat, ratio, sp_boom, rng,
                                           bool(imp.get("through_room", True)) and use_reverb, p))

        # 9. hard negatives
        hn = cfg["hard_negative"]
        if self.sampler.has("hard_negative") and rng.random() < float(hn["probability"]):
            _, row = self.sampler.draw_row("hard_negative", rng)
            ev = self.sampler.audio(row)
            ev = ev / peak(ev)
            ratio = sample(hn["peak_ratio_db"], rng)
            events.append(self._inject(boom, ref, rirs, ev, f"hardneg_{row.category}", ratio, sp_boom, rng,
                                       bool(hn.get("through_room", True)) and use_reverb, {}))

        if pre_events is not None:
            meta["_stems"]["events"] = boom - pre_events

        # 9b. measured microphone colouration on the boom channel (speech AND noise pass through the capsule)
        if self.mic_ir is not None:
            boom = apply_mic_ir(boom, self.mic_ir)
            meta["mic_ir"] = True

        # 10. ADC
        noisy = np.stack([boom, ref]).astype(np.float32)
        adc = cfg["adc"]
        if rng.random() < float(adc["probability"]):
            headroom = sample(adc["headroom_db"], rng)
            gain = float(10.0 ** (-headroom / 20.0) / (np.abs(noisy).max() + 1e-9))
            noisy = simulate_adc(noisy, headroom, int(adc.get("bits", 16)))
            meta["adc"] = {"headroom_db": float(headroom), "gain": gain}
        else:
            # never exceed full scale even without the ADC stage; a fixed safety gain, NOT normalisation
            pk = float(np.abs(noisy).max())
            gain = 0.99 / pk if pk > 0.99 else 1.0
            if gain != 1.0:
                noisy = (noisy * gain).astype(np.float32)
                meta["safety_gain"] = gain
        # the same LINEAR gain goes on the target so its level tracks the input speech level
        # (clipping and quantisation stay input-only)
        clean = (clean * gain).astype(np.float32)

        # 11. codec (boom / radio path only)
        codec = cfg["codec"]
        if rng.random() < float(codec["probability"]):
            noisy[0] = radio_codec(noisy[0], rng, dropout_p=float(codec.get("dropout_p", 0.02)))
            meta["codec"] = True

        pair = Pair(noisy=noisy.astype(np.float32), clean=clean.astype(np.float32),
                    events=events, meta=meta)
        assert_contract(pair)
        return pair
