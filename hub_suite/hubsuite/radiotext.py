"""Radio-procedure text: normalisation and rule extraction (the hot path of feature 3 -- no LLM).

    normalise("grid four niner seven two, fife meters")   -> "grid 4972, 5 meters"
    extract("Bravo two six, this is Alpha one, contact front, squad size enemy, grid four niner seven two, over")
      -> {"to": "B26", "from": "A1", "grids": ["4972"], "contact": True, "size": "squad", "direction": "front", ...}

Every value extracted here is a literal span of the transcript, so nothing can be invented. The
LLM (System 2) only adds fields; its output is checked against the transcript by `grounded()`.
"""
from __future__ import annotations

import re

import mgrs as _mgrs

NATO = {"alfa": "A", "alpha": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E", "foxtrot": "F", "golf": "G",
        "hotel": "H", "india": "I", "juliett": "J", "juliet": "J", "kilo": "K", "lima": "L", "mike": "M", "november": "N",
        "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T", "uniform": "U", "victor": "V",
        "whiskey": "W", "whisky": "W", "x-ray": "X", "xray": "X", "yankee": "Y", "zulu": "Z"}
DIGITS = {"zero": "0", "oh": "0", "one": "1", "wun": "1", "two": "2", "too": "2", "three": "3", "tree": "3", "four": "4",
          "fower": "4", "for": "4", "five": "5", "fife": "5", "six": "6", "seven": "7", "eight": "8", "ait": "8",
          "nine": "9", "niner": "9"}
SIZE = ["platoon", "squad", "section", "company", "fire team", "team"]
DIRS = ["north east", "north west", "south east", "south west", "northeast", "northwest", "southeast", "southwest",
        "north", "south", "east", "west", "front", "rear", "left", "right"]
TYPES = {
    "medevac": ["medevac", "nine line", "9 line", "casevac", "casualt", "wounded", "urgent surgical"],
    "fire_mission": ["fire mission", "adjust fire", "fire for effect", "danger close", "shot over", "splash"],
    "contact_report": ["contact", "taking fire", "under fire", "engaging", "enemy", "hostile", "ambush", "sniper", "ied"],
    "radio_check": ["radio check", "how do you read", "loud and clear", "weak but readable"],
    "sitrep": ["sitrep", "situation report", "no contact", "all clear", "ammo"],
    "movement": ["moving to", "rally point", "phase line", "en route", "eta", "holding at", "wilco"],
    "check_fire": ["check fire", "cease fire"],
}


# words ASR puts in place of radio numerals: only read as digits NEXT TO other digits ("grid five power two")
AMBIG = {"power": "4", "for": "4", "four's": "4", "to": "2", "too": "2", "won": "1", "free": "3", "fight": "5",
         "nicer": "9", "nina": "9", "ate": "8", "sex": "6", "fire": "5"}
GRID_WORDS = {"grid", "grade", "great", "grit", "greed", "greet", "grids"}


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\-']+|[,.?!]", text.lower())


def normalise(text: str) -> str:
    """Spoken radio numerals -> digits (runs of digit words become one number), NATO kept as words."""
    out: list[str] = []
    run: list[str] = []
    toks = _tok(text) + ["."]

    def digitish(j: int) -> bool:
        return 0 <= j < len(toks) and (toks[j] in DIGITS or toks[j].isdigit())
    for i, t in enumerate(toks):
        if t in (",", ".") and run and digitish(i + 1):        # "grid four, five niner" -> one number
            continue
        if t in GRID_WORDS and (digitish(i + 1) or (toks[i + 1:i + 2] and toks[i + 1] in AMBIG and digitish(i + 2))):
            t = "grid"
        if t in AMBIG and (run or digitish(i + 1)) and not (t == "fire" and not run):
            run.append(AMBIG[t])
            continue
        if t in DIGITS:
            run.append(DIGITS[t])
            continue
        if t.isdigit():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    s = " ".join(out[:-1])
    return re.sub(r" ([,.?!])", r"\1", s)


def _callsign(words: list[str], i: int) -> tuple[str, int] | None:
    if i < len(words) and words[i] in NATO:
        cs = NATO[words[i]]
        j = i + 1
        while j < len(words) and j < i + 3 and (words[j] in DIGITS or words[j].isdigit()):   # callsign: <= 2 digits
            cs += DIGITS.get(words[j], words[j])
            j += 1
        if j > i + 1:
            return cs, j
    return None


def extract(text: str) -> dict:
    words = [w for w in _tok(text) if w not in ",.?!"]
    low = " ".join(words)
    norm = normalise(text)
    r: dict = {"raw": text, "norm": norm}
    # "<to>, this is <from>"
    m = re.search(r"this is", low)
    if m:
        k = len(low[: m.start()].split())
        for i in range(max(0, k - 4), k):
            c = _callsign(words, i)
            if c and c[1] <= k:
                r["to"] = c[0]
                break
        c = _callsign(words, k + 2)
        if c:
            r["from"] = c[0]
    elif words:
        c = _callsign(words, 0)
        if c:
            r["to"] = c[0]
    # grids: "grid" followed by an even run of 4/6/8/10 digits
    grids = []
    for g in re.finditer(r"grid (\d+)", norm.replace(",", " ")):
        d = g.group(1)
        if len(d) in (4, 6, 8, 10):
            grids.append(d)
    r["grids"] = grids
    r["types"] = [t for t, keys in TYPES.items() if any(k in low for k in keys)]
    r["contact"] = "contact_report" in r["types"] and "no contact" not in low
    for s in SIZE:
        if s in low:
            r["size"] = s
            break
    else:
        m2 = re.search(r"(\d+) (?:enemy|hostiles?|troops|vehicles?|pax)", norm)
        if m2:
            r["size"] = m2.group(1)
    for d in DIRS:
        if re.search(rf"\b{d}\b", low):
            r["direction"] = d
            break
    m3 = re.search(r"(\d+) (meters|metres|klicks|kilometers|km)", norm)
    if m3:
        r["distance"] = f"{m3.group(1)} {m3.group(2)}"
    m4 = re.search(r"(\d+) (?:urgent|casualt\w*|wounded|kia|wia)", norm)
    if m4:
        r["casualties"] = int(m4.group(1))
    r["proword_end"] = "out" if re.search(r"\bout\b\W*$", low) else ("over" if re.search(r"\bover\b\W*$", low) else None)
    return r


# --------------------------------------------------------------------------------------------------
# Grid -> position inside the configured 100 km MGRS square (sandbox default: 33U VP, central Europe)
# --------------------------------------------------------------------------------------------------
_M = _mgrs.MGRS()


def grid_to_latlon(grid: str, square: str = "33UVP") -> tuple[float, float, float]:
    """(lat, lon, half-size of the uncertainty box in metres). 4 digits = 1 km square (centre), 6 = 100 m,
    8 = 10 m, 10 = 1 m."""
    n = len(grid) // 2
    e, no = grid[:n], grid[n:]
    res = 10 ** (5 - n)
    e_m = int(e) * res + res // 2
    n_m = int(no) * res + res // 2
    lat, lon = _M.toLatLon(f"{square}{e_m:05d}{n_m:05d}")
    return float(lat), float(lon), res / 2


def latlon_to_grid(lat: float, lon: float, digits: int = 8) -> str:
    s = _M.toMGRS(lat, lon, MGRSPrecision=5)
    body = s[5:]
    e, n = body[:5], body[5:]
    k = digits // 2
    return e[:k] + n[:k]


def grounded(value: str, transcript: str) -> bool:
    """Anti-hallucination check for System-2 output: every digit run / word of `value` must appear in the
    normalised transcript."""
    norm = normalise(transcript)
    for tok in re.findall(r"[a-z]+|\d+", str(value).lower()):
        if tok not in norm and tok not in transcript.lower():
            return False
    return True
