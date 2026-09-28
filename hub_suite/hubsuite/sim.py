"""Sandbox: a scripted scenario that generates radio traffic with ground truth (the "elaborate construction
to play with"). Units move inside the sandbox square; every few seconds someone transmits a contact
report, SITREP, MEDEVAC, fire mission, check fire, movement or radio check that names TRUE grids.
Audio: Piper TTS (904 voices) -> battlefield noise at the talker's mic -> tactical radio link.

    sc = Scenario(seed=1); for ev in sc.events(n=20): ev.text, ev.truth, ev.audio (16 kHz radio audio)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from .radio_link import radio_link
from .radiotext import grid_to_latlon

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
NATO = ["Alfa", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Kilo", "Lima", "Mike",
        "November", "Oscar", "Papa", "Romeo", "Sierra", "Tango", "Victor", "Whiskey", "Yankee", "Zulu"]
SAY = {"0": "zero", "1": "one", "2": "two", "3": "tree", "4": "fower", "5": "fife", "6": "six", "7": "seven",
       "8": "eight", "9": "niner"}


def say_digits(s: str, strict: float, rng: np.random.Generator) -> str:
    plain = {"3": "three", "4": "four", "5": "five"}
    return " ".join(SAY[c] if (c not in plain or rng.random() < strict) else plain[c] for c in s)


@dataclass
class Unit:
    callsign: str
    spoken: str
    e: float           # metres inside the square (0..100000)
    n: float
    vx: float = 0.0
    vy: float = 0.0
    side: str = "friendly"


@dataclass
class Event:
    kind: str
    text: str
    truth: dict
    audio: np.ndarray | None = None
    clean: np.ndarray | None = None
    info: dict = field(default_factory=dict)


class Scenario:
    def __init__(self, seed: int = 1, square: str = "33UVP", box=(40000, 60000)):
        self.rng = np.random.default_rng(seed)
        self.square = square
        r = self.rng
        lo, hi = box
        self.hq = Unit("Z0", "Zulu zero", (lo + hi) / 2, (lo + hi) / 2)
        self.friendly = []
        used = set()
        for _ in range(4):
            while True:
                a, d = NATO[int(r.integers(len(NATO)))], int(r.integers(1, 10))
                if a + str(d) not in used:
                    used.add(a + str(d))
                    break
            self.friendly.append(Unit(f"{a[0]}{d}", f"{a} {SAY[str(d)]}", r.uniform(lo, hi), r.uniform(lo, hi),
                                      r.uniform(-2, 2), r.uniform(-2, 2)))
        self.enemy = [Unit(f"E{i}", "", r.uniform(lo, hi), r.uniform(lo, hi), r.uniform(-1, 1), r.uniform(-1, 1), "enemy")
                      for i in range(3)]
        self.t = 0.0
        self._voice = None
        self.voices = {u.callsign: int(r.integers(904)) for u in self.friendly}

    # ---- world ----------------------------------------------------------------------------------
    def advance(self, dt: float) -> None:
        for u in self.friendly + self.enemy:
            u.e = float(np.clip(u.e + u.vx * dt, 1000, 99000))
            u.n = float(np.clip(u.n + u.vy * dt, 1000, 99000))
        self.t += dt

    @staticmethod
    def grid(u: Unit, digits: int = 6) -> str:
        k = digits // 2
        return f"{int(u.e):05d}"[:k] + f"{int(u.n):05d}"[:k]

    # ---- traffic --------------------------------------------------------------------------------
    def next_event(self) -> Event:
        r = self.rng
        me = self.friendly[int(r.integers(len(self.friendly)))]
        to = "Zulu zero"
        strict = float(r.uniform(0.3, 1.0))
        kind = str(r.choice(["contact", "contact", "sitrep", "medevac", "fire", "move", "radio_check", "check_fire"]))
        g_me = self.grid(me)
        if kind == "contact":
            en = min(self.enemy, key=lambda e: (e.e - me.e) ** 2 + (e.n - me.n) ** 2)
            g = self.grid(en)
            size = str(r.choice(["squad size", "platoon size", "two vehicles", "four"]))
            text = (f"{to}, this is {me.spoken}, contact {r.choice(['front', 'north', 'east', 'left'])}, {size} enemy "
                    f"{r.choice(['moving', 'stationary', 'digging in'])}, grid {say_digits(g, strict, r)}, "
                    f"{r.choice(['engaging', 'observing'])}, over.")
            truth = {"kind": "enemy", "grid": g, "from": me.callsign}
        elif kind == "sitrep":
            text = (f"{to}, this is {me.spoken}, sitrep, grid {say_digits(g_me, strict, r)}, no contact, "
                    f"ammo {r.choice(['green', 'amber'])}, over.")
            truth = {"kind": "friendly", "grid": g_me, "from": me.callsign}
        elif kind == "medevac":
            n = int(r.integers(1, 4))
            text = (f"{to}, this is {me.spoken}, nine line medevac, line one, grid {say_digits(g_me, strict, r)}, "
                    f"line three, {SAY[str(n)]} urgent, over.")
            truth = {"kind": "casualty", "grid": g_me, "from": me.callsign, "casualties": n}
        elif kind == "fire":
            en = self.enemy[int(r.integers(len(self.enemy)))]
            g = self.grid(en)
            text = (f"{to}, this is {me.spoken}, adjust fire, grid {say_digits(g, strict, r)}, "
                    f"infantry in the open, fire for effect, over.")
            truth = {"kind": "fire_mission", "grid": g, "from": me.callsign}
        elif kind == "move":
            text = (f"{to}, this is {me.spoken}, moving to rally point {r.choice(NATO)}, "
                    f"grid {say_digits(g_me, strict, r)}, ETA {say_digits(str(int(r.integers(10, 60))), strict, r)} minutes, over.")
            truth = {"kind": "friendly", "grid": g_me, "from": me.callsign}
        elif kind == "check_fire":
            g = self.grid(me, 4)
            text = f"All stations, this is {me.spoken}, check fire, check fire, grid {say_digits(g, strict, r)}, out."
            truth = {"kind": "check_fire", "grid": g, "from": me.callsign}
        else:
            text = f"{to}, this is {me.spoken}, radio check, over."
            truth = {"kind": None, "grid": None, "from": me.callsign}
        truth["t"] = self.t
        return Event(kind, text, truth, info={"voice": self.voices[me.callsign]})

    # ---- audio --------------------------------------------------------------------------------------
    def speak(self, ev: Event, snr_db: float | None = None, link: str | None = None) -> Event:
        from piper import PiperVoice, SynthesisConfig
        if self._voice is None:
            self._voice = PiperVoice.load(str(ROOT / "artefacts" / "tts" / "en_US-libritts_r-medium.onnx"))
        r = self.rng
        chunks = list(self._voice.synthesize(ev.text, SynthesisConfig(speaker_id=ev.info["voice"],
                                                                     length_scale=float(r.uniform(0.85, 1.0)))))
        x = np.concatenate([c.audio_float_array for c in chunks])
        g = np.gcd(chunks[0].sample_rate, SR)
        x = resample_poly(x, SR // g, chunks[0].sample_rate // g).astype(np.float32)
        x = np.concatenate([np.zeros(int(0.25 * SR), np.float32), x, np.zeros(int(0.3 * SR), np.float32)])
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.05
        snr = float(r.uniform(0, 15)) if snr_db is None else snr_db
        noise = battlefield_noise(len(x), r, gunfire=ev.kind in ("contact", "fire"))
        noise *= np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(noise ** 2)) + 1e-9) * 10 ** (-snr / 20)
        mic = np.clip(x + noise, -1, 1).astype(np.float32)
        params = {"link_p": {link: 1.0}} if link else {"link_p": {"cvsd16": 0.55, "fm": 0.35, "cvsd32": 0.10}}
        y, tgt, _, info = radio_link(mic, x, np.random.default_rng(int(r.integers(2 ** 31))), params)
        ev.audio, ev.clean = y, tgt
        ev.info.update(info, snr_db=snr)
        return ev


def battlefield_noise(n: int, r: np.random.Generator, gunfire: bool = False) -> np.ndarray:
    """Synthetic scene for the sandbox (the real pools stay with the dataset): gusting wind / engine rumble
    (filtered noise with a slow envelope) + optional bursts of rifle fire (Friedlander pulses + crack)."""
    w = r.standard_normal(n)
    kind = r.integers(3)
    if kind == 0:                                                    # wind: low-passed, gusting
        y = sosfilt(butter(2, 400, "lp", fs=SR, output="sos"), w)
        env = np.interp(np.arange(n), np.linspace(0, n, 8), 10 ** (r.uniform(-8, 4, 8) / 20))
    elif kind == 1:                                                  # engine: harmonic rumble
        f0 = r.uniform(25, 45)
        t = np.arange(n) / SR
        y = sum(np.sin(2 * np.pi * f0 * k * t + r.uniform(0, 6)) / k for k in range(1, 8)) + 0.3 * \
            sosfilt(butter(2, 1500, "lp", fs=SR, output="sos"), w)
        env = np.ones(n)
    else:                                                            # broadband ambience
        y = sosfilt(butter(2, [100, 4000], "bp", fs=SR, output="sos"), w)
        env = np.ones(n)
    y = y * env
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9)
    if gunfire:
        for _ in range(int(r.integers(1, 3))):
            t0 = int(r.uniform(0.2, 0.8) * n)
            for k in range(int(r.integers(3, 9))):
                a = t0 + k * int(r.uniform(0.07, 0.1) * SR)
                L = int(0.004 * SR)
                if a + 3 * L >= n:
                    break
                tt = np.arange(3 * L) / SR
                pulse = (1 - tt / 0.002) * np.exp(-tt / 0.002)
                y[a: a + 3 * L] += pulse * r.uniform(8, 20)
    return y.astype(np.float32)


def position_error_m(truth_grid: str, marker_lat: float, marker_lon: float, square: str = "33UVP") -> float:
    lat, lon, _ = grid_to_latlon(truth_grid, square)
    k = 111320.0
    return float(np.hypot((marker_lat - lat) * k, (marker_lon - lon) * k * np.cos(np.radians(lat))))
