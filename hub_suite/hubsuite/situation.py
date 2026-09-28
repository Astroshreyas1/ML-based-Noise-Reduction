"""The situation picture (feature 3): transmissions in, markers out.

System 1 (hot path, deterministic, cannot invent): radiotext.extract (callsigns, grids, sizes, types) and
optionally Laya typed decisions (message type / urgency / enemy contact) -> markers immediately.
System 2 (background): a local LLM proposes extra fields; each is kept only if `radiotext.grounded()` finds
it in the transcript; it never creates a marker from nothing.
Every marker keeps the id of the transmission it came from (click -> hear it).
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import queue
import threading
import time
from dataclasses import asdict, dataclass, field

from .radiotext import extract, grid_to_latlon

SQUARE = "33UVP"
CONFIRM_P = 0.75          # calibrated on 60 sandbox transmissions: no wrong grid confirmed (5 wrong all <= 0.72)


def _grid_confidence(grids: list[str], words: list[tuple[str, float]]) -> dict[str, float]:
    """min ASR word probability over the digit words that make up each grid (in order of appearance)."""
    from .radiotext import AMBIG, DIGITS
    digs: list[tuple[str, float]] = []
    for w, p in words:
        t = w.lower().strip(".,?!")
        if t in DIGITS:
            digs.append((DIGITS[t], p))
        elif t in AMBIG:
            digs.append((AMBIG[t], p))
        elif t.isdigit():
            digs += [(c, p) for c in t]
    s = "".join(d for d, _ in digs)
    out = {}
    for g in grids:
        k = s.find(g)
        if k >= 0:
            out[g] = float(np.mean([p for _, p in digs[k:k + len(g)]]))
    return out          # sandbox 100 km square (config); grids in traffic are relative to it


@dataclass
class Marker:
    id: str
    kind: str                     # friendly | enemy | casualty | fire_mission | check_fire | note
    label: str
    grid: str
    lat: float
    lon: float
    half_m: float                 # uncertainty (half side of the grid square)
    t: float
    tx: str                       # transmission id
    source: str                   # rules | laya | llm | operator
    details: dict = field(default_factory=dict)


@dataclass
class Transmission:
    id: str
    t: float
    text: str
    partial: str = ""
    extraction: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    audio_path: str | None = None


class Situation:
    def __init__(self, square: str = SQUARE, laya=None):
        self.square = square
        self.laya = laya
        self.markers: dict[str, Marker] = {}
        self.tx: dict[str, Transmission] = {}
        self.units: dict[str, str] = {}        # callsign -> marker id (last known)
        self._ids = itertools.count(1)
        self.lock = threading.Lock()
        self.subscribers: list[queue.Queue] = []

    # ---- pub/sub (SSE) ------------------------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        self.subscribers.append(q)
        return q

    def publish(self, kind: str, obj: dict) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait({"type": kind, "data": obj})
            except queue.Full:
                pass

    def snapshot(self) -> dict:
        with self.lock:
            return {"square": self.square, "markers": [asdict(m) for m in self.markers.values()],
                    "transmissions": [asdict(t) for t in sorted(self.tx.values(), key=lambda x: x.t)][-200:]}

    # ---- ingest -------------------------------------------------------------------------------
    def partial(self, tx_id: str, text: str) -> None:
        self.publish("partial", {"id": tx_id, "text": text})

    def ingest(self, tx_id: str, text: str, t_end: float | None = None, audio_path: str | None = None,
               extra_latency: dict | None = None, words: list[tuple[str, float]] | None = None) -> Transmission:
        t0 = time.perf_counter()
        ex = extract(text)
        if "[unreliable" in text:                          # looped ASR: never plot its digits
            ex["grids"] = []
            ex["unreliable"] = True
        ex["grid_conf"] = _grid_confidence(ex.get("grids", []), words) if words else {}
        t_rules = time.perf_counter()
        dec = self._laya(text) if self.laya is not None and text.strip() else {}
        t_laya = time.perf_counter()
        tx = Transmission(tx_id, time.time(), text, extraction={k: v for k, v in ex.items() if k != "raw"},
                          decisions=dec, audio_path=audio_path)
        tx.latency = {**(extra_latency or {}), "rules_ms": (t_rules - t0) * 1000, "laya_ms": (t_laya - t_rules) * 1000}
        with self.lock:
            self.tx[tx_id] = tx
            new = self._apply(tx, ex, dec)
        tx.latency["map_ms"] = (time.perf_counter() - t0) * 1000
        self.publish("transmission", asdict(tx))
        for m in new:
            self.publish("marker", asdict(m))
        return tx

    def _laya(self, text: str) -> dict:
        from .decisions import decide
        return decide(self.laya, text)

    def _mk(self, kind: str, label: str, grid: str, tx: Transmission, source: str, **details) -> Marker | None:
        try:
            lat, lon, half = grid_to_latlon(grid, self.square)
        except Exception:
            return None
        m = Marker(f"m{next(self._ids)}", kind, label, grid, lat, lon, half, time.time(), tx.id, source, details)
        self.markers[m.id] = m
        return m

    def _apply(self, tx: Transmission, ex: dict, dec: dict) -> list[Marker]:
        new: list[Marker] = []
        types = set(ex.get("types", []))
        if dec.get("msg_type") and dec.get("msg_type_conf", 0) > 0.6:
            types.add(dec["msg_type"])
        enemy = ex.get("contact") or dec.get("enemy_contact", 0) > 0.7
        sender = ex.get("from")
        for g in ex.get("grids", []):
            if "medevac" in types:
                m = self._mk("casualty", f"CASEVAC {ex.get('casualties', '?')} urgent", g, tx, "rules", casualties=ex.get("casualties"))
            elif "check_fire" in types:
                m = self._mk("check_fire", "CHECK FIRE", g, tx, "rules")
            elif "fire_mission" in types:
                m = self._mk("fire_mission", "Fire mission", g, tx, "rules")
            elif enemy:
                m = self._mk("enemy", f"Enemy {ex.get('size', '')}".strip(), g, tx, "rules",
                             size=ex.get("size"), direction=ex.get("direction"), reported_by=sender)
            elif sender:
                old = self.units.get(sender)
                m = self._mk("friendly", sender, g, tx, "rules")
                if m is not None:
                    if old and old in self.markers:
                        self.markers[old].details["superseded_by"] = m.id
                    self.units[sender] = m.id
            else:
                m = self._mk("note", "Position", g, tx, "rules")
            if m is not None:
                conf = ex.get("grid_conf", {}).get(g)
                m.details["confidence"] = conf
                m.details["confirmed"] = conf is None or conf >= CONFIRM_P
                if not m.details["confirmed"]:
                    m.label = "UNCONFIRMED " + m.label
                new.append(m)
        return new

    def enrich(self, tx_id: str, fields: dict, source: str = "llm") -> dict:
        """System-2 output: keep a field only if it is grounded in the transmission's transcript."""
        from .radiotext import grounded
        tx = self.tx.get(tx_id)
        if tx is None:
            return {}
        kept = {k: v for k, v in fields.items() if v not in (None, "", []) and grounded(v if not isinstance(v, list) else " ".join(map(str, v)), tx.text)}
        # System 1 is authoritative where both speak: a System-2 value that CONTRADICTS the rules is dropped
        from .radiotext import extract as _ex, normalise
        ex = tx.extraction
        conflicts = []
        for fld, rule in (("sender", "from"), ("receiver", "to")):
            if fld in kept and ex.get(rule):
                cs = _ex(f"{kept[fld]}, this is x").get("to") or ""
                if cs and cs != ex[rule]:
                    conflicts.append(fld)
        if "grid_digits" in kept and ex.get("grids"):
            if re.sub(r"\D", "", normalise(str(kept["grid_digits"]))) not in ex["grids"]:
                conflicts.append("grid_digits")
        from .radiotext import DIRS
        if "direction" in kept and str(kept["direction"]).lower().strip() not in DIRS:
            conflicts.append("direction")
        enemy_ok = ex.get("contact") or tx.decisions.get("enemy_contact", 0) > 0.5
        if not enemy_ok:
            conflicts += [k for k in ("enemy_size", "enemy_activity") if k in kept]
        for k in conflicts:
            kept.pop(k, None)
        dropped = sorted(set(fields) - set(kept))
        with self.lock:
            tx.extraction.setdefault("llm", {}).update(kept)
            for m in self.markers.values():
                if m.tx == tx_id:
                    m.details.setdefault("llm", {}).update(kept)
        self.publish("enrich", {"id": tx_id, "kept": kept, "dropped": dropped, "source": source})
        return {"kept": kept, "dropped": dropped}

    def operator_marker(self, grid: str, label: str, kind: str = "note") -> Marker | None:
        tx = Transmission(f"op{next(self._ids)}", time.time(), f"operator: {label} grid {grid}")
        with self.lock:
            self.tx[tx.id] = tx
            m = self._mk(kind, label, grid, tx, "operator")
        if m:
            self.publish("marker", asdict(m))
        return m
