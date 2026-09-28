"""Military / radio voice-procedure vocabulary and a phrase generator for TTS.

Vocabulary (docs/research/RADIO_VOCAB_2026-09-27.md): NATO spelling alphabet,
radio numerals, prowords (ACP 125(G), ATP 6-02.53), contact / MEDEVAC / call-for-
fire words, ATC and general radio words. `KEYWORDS` (single tokens, lower case)
is what the keyword spans, the keyword-weighted loss and the keyword-recall
metric look for.

`radio_phrase(rng)` writes one realistic transmission ("Bravo two six, this is
Alpha one, contact front, grid four niner seven two, over") from templates of
real procedure: radio checks, contact reports (SALUTE-style), SITREPs, 9-line
MEDEVAC lines, calls for fire, relay / say again / I spell exchanges. Digits are
spoken the radio way (niner, tree, fife in some voices, figures spoken singly).
"""
from __future__ import annotations

import re

import numpy as np

NATO = ["alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett", "kilo", "lima",
        "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
        "x-ray", "yankee", "zulu"]
DIGIT_RADIO = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "niner"}
DIGIT_ALT = {3: "tree", 4: "fower", 5: "fife"}

PROWORDS = ["roger", "wilco", "over", "out", "say again", "i say again", "break", "copy", "copy that", "affirmative",
            "negative", "wait", "wait out", "correction", "i spell", "read back", "radio check", "loud and clear",
            "stand by", "acknowledge", "all after", "all before", "disregard", "this is", "message", "more to follow",
            "speak slower", "verify", "authenticate", "nothing heard", "relay", "send your message", "go ahead",
            "execute", "flash", "immediate", "priority", "routine", "that is correct", "how do you read", "how copy",
            "weak but readable", "unreadable", "distorted", "all stations", "understood", "confirm", "mayday", "sitrep"]
TACTICAL = ["contact", "enemy", "hostile", "friendly", "friendlies", "troops", "infantry", "vehicle", "vehicles", "tank",
            "truck", "convoy", "patrol", "position", "location", "moving", "stationary", "casualty", "casualties",
            "wounded", "medevac", "casevac", "medic", "urgent", "litter", "ambulatory", "ied", "ambush", "sniper",
            "mortar", "artillery", "rocket", "rpg", "small arms", "machine gun", "fire mission", "adjust fire",
            "fire for effect", "danger close", "check fire", "cease fire", "shot", "splash", "rounds", "ammo",
            "under fire", "taking fire", "pinned down", "suppress", "engage", "target", "eta", "rally point",
            "objective", "checkpoint", "phase line", "landing zone", "lz", "extract", "reinforcements", "support",
            "requesting", "en route", "secure", "clear", "hold", "halt", "advance", "withdraw", "fall back",
            "helicopter", "drone", "uav", "airstrike", "bridge", "road", "building", "compound", "tree line", "ridge",
            "hill", "village", "platoon", "squad", "company", "commander", "actual", "headquarters", "smoke", "flare",
            "grenade", "mine", "grid", "north", "south", "east", "west", "left", "right", "front", "rear", "meters",
            "klicks", "bearing", "heading", "degrees", "mils", "hours", "minutes", "figures"]
ATC = ["tower", "ground", "approach", "runway", "taxi", "climb", "descend", "maintain", "flight level", "turn left",
       "turn right", "frequency", "squawk", "traffic", "wind", "knots", "report", "final", "go around", "unable",
       "emergency", "monitor", "radar", "direct", "do you copy", "all units", "responding", "stand down", "repeat"]

KEYWORDS: set[str] = set()
for _w in NATO + list(DIGIT_RADIO.values()) + list(DIGIT_ALT.values()) + PROWORDS + TACTICAL + ATC + \
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "hundred", "thousand"]:
    KEYWORDS.update(re.findall(r"[a-z]+(?:-[a-z]+)?", _w.lower()))
KEYWORDS -= {"i", "is", "the", "a", "to", "all", "that", "do", "you", "but", "and", "for", "this", "out", "over"}
KEYWORDS |= {"over", "out", "this"}         # prowords that are also function words: keep, they carry procedure


def _pick(rng: np.random.Generator, xs):
    return xs[int(rng.integers(len(xs)))]


def _digits(rng: np.random.Generator, n: int, alt_p: float) -> str:
    out = []
    for _ in range(n):
        d = int(rng.integers(10))
        out.append(DIGIT_ALT[d] if d in DIGIT_ALT and rng.random() < alt_p else DIGIT_RADIO[d])
    return " ".join(out)


def callsign(rng: np.random.Generator, alt_p: float = 0.3) -> str:
    return f"{_pick(rng, NATO).title()} {_digits(rng, int(rng.integers(1, 3)), alt_p)}"


def grid_ref(rng: np.random.Generator, alt_p: float) -> str:
    return f"grid {_digits(rng, int(_pick(rng, [4, 6, 6, 8])), alt_p)}"


def radio_phrase(rng: np.random.Generator) -> str:
    """One transmission, 4-10 s when spoken."""
    alt = float(rng.uniform(0.0, 0.8))          # how strictly this talker uses tree / fower / fife
    me, you = callsign(rng, alt), callsign(rng, alt)
    t = int(rng.integers(12))
    dist = f"{_digits(rng, int(rng.integers(2, 4)), alt)} {_pick(rng, ['meters', 'meters', 'klicks'])}"
    dirn = _pick(rng, ["north", "south", "east", "west", "north east", "south west", "left", "right", "front", "rear"])
    end = _pick(rng, ["over", "over", "over", "out", "over"])
    if t == 0:
        return f"{you}, this is {me}, radio check, over."
    if t == 1:
        return f"{me}, this is {you}, I read you {_pick(rng, ['loud and clear', 'weak but readable', 'broken and distorted'])}, over."
    if t == 2:
        size = _pick(rng, ["two", "three", "four", "squad size", "platoon size", "one vehicle", "two vehicles"])
        act = _pick(rng, ["moving", "stationary", "digging in", "setting up a machine gun", "moving toward the bridge"])
        return (f"{you}, this is {me}, contact {dirn}, {size} enemy {act}, {grid_ref(rng, alt)}, "
                f"{_pick(rng, ['engaging', 'observing', 'taking fire', 'request support'])}, {end}.")
    if t == 3:
        return (f"{you}, this is {me}, sitrep, {grid_ref(rng, alt)}, {_pick(rng, ['no contact', 'contact at', 'all clear'])}, "
                f"{_digits(rng, 1, alt)} casualties, ammo {_pick(rng, ['green', 'amber', 'red'])}, {end}.")
    if t == 4:
        return (f"{you}, this is {me}, nine line medevac, line one, {grid_ref(rng, alt)}, line two, "
                f"{_pick(rng, NATO).title()} {_digits(rng, 2, alt)}, line three, {_digits(rng, 1, alt)} urgent, {end}.")
    if t == 5:
        return (f"{you}, this is {me}, adjust fire, {grid_ref(rng, alt)}, direction {_digits(rng, 4, alt)} mils, "
                f"{_pick(rng, ['infantry in the open', 'enemy mortar', 'vehicle in the tree line'])}, "
                f"{_pick(rng, ['fire for effect', 'danger close', 'at my command'])}, {end}.")
    if t == 6:
        return f"{me}, this is {you}, shot, over. {you}, shot out. Splash, over. Splash out."
    if t == 7:
        word = _pick(rng, ["bridge", "ridge", "compound", "village", "checkpoint"])
        spelled = " ".join(_pick(rng, NATO).title() for _ in range(int(rng.integers(3, 6))))
        return f"{you}, this is {me}, say again all after {word}, I spell, {spelled}, {end}."
    if t == 8:
        return (f"{me}, this is {you}, roger, {_pick(rng, ['wilco', 'moving to rally point', 'holding at phase line'])} "
                f"{_pick(rng, NATO).title()}, ETA {_digits(rng, 2, alt)} minutes, {end}.")
    if t == 9:
        return (f"{you}, {me}, taking fire from the {_pick(rng, ['tree line', 'ridge', 'building', 'compound'])}, "
                f"{dirn} {dist}, request {_pick(rng, ['casevac', 'air support', 'smoke', 'reinforcements'])}, {end}.")
    if t == 10:
        return (f"All stations, this is {me}, {_pick(rng, ['check fire, check fire', 'cease fire, cease fire', 'IED on the road', 'drone overhead'])}, "
                f"{grid_ref(rng, alt)}, acknowledge, {end}.")
    return (f"{_pick(rng, ['Tower', 'Ground', 'Approach'])}, {me}, {_pick(rng, ['request taxi', 'ready for departure', 'on final', 'request descent'])}, "
            f"{_pick(rng, ['runway', 'flight level', 'heading'])} {_digits(rng, 3, alt)}, {_pick(rng, ['roger', 'wilco', 'copy'])}.")


def keyword_hits(words: list[str]) -> list[bool]:
    return [re.sub(r"[^a-z\-]", "", w.lower()) in KEYWORDS for w in words]
