"""Radio-procedure transmissions with ground truth, for fine-tuning the ASR that feeds the map.

Wider than the sandbox's seven templates (hubsuite/sim.py) so whisper learns radio numerals and
procedure, not seven sentences: 4/6/8-digit grids, corrections, "I spell", say-again, 9-line MEDEVAC,
call for fire with direction and shot/splash, SALUTE spot reports, bearings, distances, times,
frequencies, every NATO letter in callsigns. Digits are uniform random, so no grid can be guessed
from the language model.

    t = make(rng) -> {"text": "...", "grids": ["447559"], "kind": "contact"}
"""
from __future__ import annotations

import numpy as np

NATO = ["Alfa", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliett", "Kilo", "Lima",
        "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey",
        "X-ray", "Yankee", "Zulu"]
RADIO = {"0": "zero", "1": "one", "2": "two", "3": "tree", "4": "fower", "5": "fife", "6": "six", "7": "seven",
         "8": "eight", "9": "niner"}
PLAIN = {"3": "three", "4": "four", "5": "five", "9": "nine"}


class Say:
    """Numerals in radio style with a per-transmission strictness (how often tree/fower/fife/niner are used)."""

    def __init__(self, rng: np.random.Generator):
        self.r = rng
        self.strict = float(rng.uniform(0.2, 1.0))

    def digits(self, s: str) -> str:
        return " ".join(RADIO[c] if (c not in PLAIN or self.r.random() < self.strict) else PLAIN[c] for c in s)

    def num(self, lo: int, hi: int) -> str:
        return self.digits(str(int(self.r.integers(lo, hi))))

    def callsign(self) -> str:
        return f"{NATO[int(self.r.integers(len(NATO)))]} {self.digits(str(int(self.r.integers(1, 100))))}"


def _grid(r: np.random.Generator) -> str:
    n = int(r.choice([4, 6, 8], p=[0.15, 0.6, 0.25]))
    return "".join(str(int(d)) for d in r.integers(0, 10, n))


def make(r: np.random.Generator) -> dict:
    s = Say(r)
    to, me = (s.callsign() if r.random() < 0.8 else "Zulu zero"), s.callsign()
    head = f"{to}, this is {me}"
    g = _grid(r)
    G = s.digits(g)
    grids = [g]
    kind = str(r.choice(["contact", "contact", "spot", "sitrep", "medevac", "fire", "fire_adjust", "move", "check_fire",
                         "radio_check", "say_again", "correction", "i_spell", "freq"]))
    pick = lambda xs: str(r.choice(xs))  # noqa: E731
    over = pick(["over.", "over.", "out."])
    if kind == "contact":
        text = (f"{head}, contact {pick(['front', 'rear', 'left', 'right', 'north', 'south', 'east', 'west'])}, "
                f"{pick(['squad size', 'platoon size', 'section size', 'two vehicles', 'one BMP', 'fower pax', 'sniper'])} "
                f"enemy {pick(['moving east', 'moving west', 'stationary', 'digging in', 'in the treeline'])}, grid {G}, "
                f"{pick(['engaging', 'observing', 'breaking contact', 'request fire'])}, {over}")
    elif kind == "spot":
        text = (f"{head}, spot report, size {s.num(2, 40)} personnel, activity {pick(['moving south', 'setting up mortars', 'patrolling', 'refuelling'])}, "
                f"location grid {G}, unit {pick(['unknown', 'mechanised', 'light infantry'])}, time {s.digits(f'{int(r.integers(0, 24)):02d}{int(r.integers(0, 60)):02d}')}, "
                f"equipment {pick(['small arms', 'RPG', 'two trucks', 'heavy machine gun'])}, {over}")
    elif kind == "sitrep":
        text = (f"{head}, sitrep, location grid {G}, {pick(['no contact', 'quiet', 'contact earlier, now clear'])}, "
                f"ammo {pick(['green', 'amber', 'red'])}, fuel {pick(['green', 'amber'])}, {s.num(0, 4)} casualties, {over}")
    elif kind == "medevac":
        n = s.digits(str(int(r.integers(1, 5))))
        text = (f"{head}, nine line medevac, line one, grid {G}, line two, {s.num(30, 88)} decimal {s.num(0, 10)}, "
                f"{s.callsign()}, line tree, {n} urgent, line fower, {pick(['none', 'hoist', 'extraction equipment'])}, "
                f"line fife, {n} litter, line six, {pick(['no enemy', 'possible enemy', 'enemy in area, armed escort required'])}, {over}")
    elif kind == "fire":
        text = (f"{head}, fire mission, grid {G}, {pick(['infantry in the open', 'vehicles', 'mortar position', 'bunker'])}, "
                f"{pick(['fire for effect', 'adjust fire', 'danger close, adjust fire'])}, {over}")
    elif kind == "fire_adjust":
        text = (f"{head}, adjust fire, polar, direction {s.digits(f'{int(r.integers(0, 6400)):04d}')} mils, "
                f"distance {s.num(1, 9)} hundred, {pick(['add', 'drop'])} {s.num(1, 5)} hundred, "
                f"{pick(['left', 'right'])} {s.num(1, 9)} zero, grid {G}, fire for effect, {over}")
    elif kind == "move":
        text = (f"{head}, moving to {pick(['rally point', 'checkpoint', 'phase line'])} {NATO[int(r.integers(len(NATO)))]}, "
                f"grid {G}, ETA {s.num(5, 60)} minutes, {over}")
    elif kind == "check_fire":
        text = f"All stations, this is {me}, check fire, check fire, grid {G}, {pick(['out.', 'I say again, check fire, out.'])}"
    elif kind == "radio_check":
        text = f"{head}, radio check, {over} {pick(['', 'This is ' + to + ', loud and clear, over.', 'This is ' + to + ', weak but readable, over.'])}"
        grids = []
    elif kind == "say_again":
        text = f"{head}, say again grid, over. This is {to}, I say again, grid {G}, {over}"
    elif kind == "correction":
        wrong = _grid(r)[: len(g)]
        text = f"{head}, {pick(['contact front', 'sitrep', 'moving'])}, grid {s.digits(wrong)}, correction, grid {G}, {over}"
    elif kind == "i_spell":
        word = pick(["Kharkiv", "Novak", "Petrov", "Dalny", "Krasny", "Volkov"])
        text = f"{head}, target at village {word}, I spell, {', '.join(NATO[ord(c) - 65] for c in word.upper())}, grid {G}, {over}"
    else:  # freq
        text = (f"{head}, change to frequency {s.num(30, 88)} decimal {s.digits(f'{int(r.integers(0, 1000)):03d}')}, "
                f"{pick(['wilco', 'roger'])}, report at grid {G}, {over}")
    return {"text": " ".join(text.split()), "grids": grids, "kind": kind}
