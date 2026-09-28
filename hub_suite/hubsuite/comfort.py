"""Comfort bed for the officer's headset (feature 1: "soothing frequency").

Evidence (docs/RESEARCH_comfort_and_field_learning.md, section A): no frequency calms people by a special property
(432 Hz, solfeggio, binaural-beat "entrainment": unsupported; binaural beats need stereo and a
400-500 Hz carrier inside the speech band). Changing-state sound during listening harms serial recall
(callsigns, grids). What has evidence: a steady, unobtrusive bed, and SLOW PACED BREATHING (~6/min,
resonance-frequency HRV biofeedback, d ~ 0.8) when the listener actually breathes along.

Design, therefore:
    * officer headset only, NEVER on transmit; opt-in, default OFF; kill switch; auto-off after 2 h
    * steady brown noise, low-passed < 250 Hz (no overlap with the 300-3400 Hz radio speech), level
      >= 25 dB below the speech active level, < 1 dB slow drift
    * optional breath pacer: the same bed swelling +-6 dB, 4 s "in" / 6 s "out", only when idle > 3 s
    * DUCKED TO SILENCE whenever the squelch is open, the input VAD fires, the officer keys PTT, or an
      alarm sounds. The sidechain reads HubNet's INPUT, which is 18 ms ahead of HubNet's output, so the
      bed is gone before the enhanced speech arrives: zero added latency, zero overlap with speech.

    bed = ComfortBed(mode="line_alive")          # or "pacer" / "off"
    y_out = bed.process(y_hub_hop, x_in_hop, squelch_open=..., ptt=...)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi

SR = 16000


@dataclass
class ComfortConfig:
    mode: str = "off"                 # off | line_alive | pacer
    level_below_speech_db: float = 25.0
    speech_ref_dbfs: float = -26.0    # running estimate replaces this once speech has been heard
    lp_hz: float = 250.0
    attack_ms: float = 10.0
    hold_ms: float = 1000.0
    release_ms: float = 500.0
    pacer_in_s: float = 4.0
    pacer_out_s: float = 6.0
    pacer_depth_db: float = 6.0
    pacer_idle_s: float = 3.0
    vad_threshold_dbfs: float = -45.0
    max_on_s: float = 2 * 3600.0


class ComfortBed:
    def __init__(self, cfg: ComfortConfig | None = None, seed: int = 0):
        self.cfg = cfg or ComfortConfig()
        self.rng = np.random.default_rng(seed)
        self.sos = butter(4, self.cfg.lp_hz, "lp", fs=SR, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.brown = 0.0
        self.g = 0.0
        self.hold = 0.0
        self.idle_s = 0.0
        self.t = 0.0
        self.drift = 0.0
        self.speech_db = self.cfg.speech_ref_dbfs
        self.on_since = time.monotonic()
        self.killed = False
        self.log: list[tuple[float, str]] = []

    # ---- controls -------------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        assert mode in ("off", "line_alive", "pacer")
        self.cfg.mode = mode
        self.on_since = time.monotonic()
        self.log.append((time.time(), f"mode={mode}"))

    def kill(self) -> None:
        self.killed = True
        self.log.append((time.time(), "kill"))

    # ---- signal -----------------------------------------------------------------------------
    def _brown(self, n: int) -> np.ndarray:
        w = self.rng.standard_normal(n) * 0.02
        out, zb = lfilter([1.0], [1.0, -0.998], w, zi=[self.brown * 0.998])   # leaky integrator: brown noise
        self.brown = float(out[-1])
        y, self.zi = sosfilt(self.sos, out, zi=self.zi)
        return y.astype(np.float32)

    def _pacer_gain_db(self) -> float:
        c = self.cfg
        per = c.pacer_in_s + c.pacer_out_s
        ph = self.t % per
        if ph < c.pacer_in_s:                  # smooth rise
            x = 0.5 - 0.5 * np.cos(np.pi * ph / c.pacer_in_s)
        else:                                  # smooth fall
            x = 0.5 + 0.5 * np.cos(np.pi * (ph - c.pacer_in_s) / c.pacer_out_s)
        return float(c.pacer_depth_db * (2 * x - 1))

    def process(self, y_hub: np.ndarray, x_in: np.ndarray, squelch_open: bool = False, ptt: bool = False,
                alarm: bool = False) -> np.ndarray:
        """One hop: y_hub = HubNet output hop, x_in = the hop of HubNet INPUT that is 18 ms newer."""
        c = self.cfg
        n = len(y_hub)
        dt = n / SR
        self.t += dt
        lvl = 10 * np.log10(np.mean(np.square(x_in, dtype=np.float64)) + 1e-12)
        speech = lvl > c.vad_threshold_dbfs
        out_lvl = 10 * np.log10(np.mean(np.square(y_hub, dtype=np.float64)) + 1e-12)
        if out_lvl > c.vad_threshold_dbfs + 10:          # track the speech level the bed sits under
            self.speech_db = 0.99 * self.speech_db + 0.01 * out_lvl
        if c.mode != "off" and time.monotonic() - self.on_since > c.max_on_s:
            self.set_mode("off")
            self.log.append((time.time(), "auto-off after max_on_s"))
        duck = squelch_open or speech or ptt or alarm or self.killed or c.mode == "off" or np.abs(y_hub).max() > 0.99
        if duck:
            self.hold = c.hold_ms / 1000
            self.idle_s = 0.0
            target = 0.0
        else:
            self.idle_s += dt
            self.hold = max(0.0, self.hold - dt)
            target = 0.0 if self.hold > 0 else 1.0
        tau = (c.attack_ms if target < self.g else c.release_ms) / 1000
        self.g += (target - self.g) * (1 - np.exp(-dt / tau))
        if self.g < 1e-4 and target == 0.0:
            self._brown(n)                                  # keep the generator running, output silence
            return y_hub
        bed = self._brown(n)
        self.drift = 0.999 * self.drift + 0.001 * self.rng.standard_normal()
        level_db = self.speech_db - c.level_below_speech_db + float(np.clip(self.drift, -1, 1))
        if c.mode == "pacer" and self.idle_s > c.pacer_idle_s:
            level_db += self._pacer_gain_db()
        bed_rms = float(np.sqrt(np.mean(bed ** 2)) + 1e-9)
        bed = bed / bed_rms * 10 ** (level_db / 20)
        return (y_hub + self.g * bed).astype(np.float32)


def evaluate_bed(clean: np.ndarray, enhanced: np.ndarray, noisy: np.ndarray, mode: str = "line_alive") -> dict:
    """Objective gate (research A4): STOI with the bed ON vs OFF on a stream, and the bed energy that
    leaks into speech frames (must be < -40 dB re speech)."""
    from pystoi import stoi
    hop = 96
    delay = 288                                             # HubNet output lags its input by 18 ms
    bed = ComfortBed(ComfortConfig(mode=mode))
    out = np.zeros_like(enhanced)
    for s in range(0, len(enhanced) - hop, hop):
        xi = noisy[min(len(noisy) - hop, s + delay): min(len(noisy), s + delay + hop)]
        out[s: s + hop] = bed.process(enhanced[s: s + hop], xi)
    leak = out - enhanced
    sp = np.abs(clean) > 0.05 * np.abs(clean).max()
    leak_db = 10 * np.log10((np.mean(leak[sp] ** 2) + 1e-12) / (np.mean(clean[sp] ** 2) + 1e-12)) if sp.any() else -99
    return {"stoi_off": stoi(clean, enhanced, SR), "stoi_on": stoi(clean, out, SR), "leak_in_speech_db": float(leak_db),
            "bed_on_fraction": float(np.mean(np.abs(leak) > 1e-6))}
