"""Labelled radio-transmission cases for fine-tuning Laya (System-1 typed decisions).

Format = LocalLLaMA/typed-decisions (what the Laya fine-tuning notebook consumes):
    {"id", "workflow": "radio_hub", "state": json({"transcript": ...}), "questions": json(QUESTIONS),
     "gold": json({qid: {"label": ..., "probabilities": {...}}})}

Text sources, all with ground truth from the sandbox scenario (hubsuite/sim.py):
    clean      the scripted transmission text
    asr_sim    the same text through an ASR-error model (radio numerals misheard, "grid" -> "grade",
               dropped / inserted words, numerals written as digits, punctuation lost)
    asr_real   real transcripts from the full chain (TTS -> noise -> radio link -> HubNet -> whisper), from
               sandbox run reports (runs/*/report.json) -- the most realistic, used for eval first

    python finetune/make_dataset.py --n 12000 --out finetune/data
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hubsuite.decisions import QUESTIONS  # noqa: E402
from hubsuite.sim import Scenario  # noqa: E402

KIND_TO = {  # sandbox event kind -> (msg_type, urgency level 0..2, enemy_contact, casualties)
    "contact": ("contact_report", 2, True, False),
    "medevac": ("medevac", 2, False, True),
    "fire": ("fire_mission", 1, True, False),
    "check_fire": ("check_fire", 2, False, False),
    "sitrep": ("sitrep", 0, False, False),
    "move": ("movement", 0, False, False),
    "radio_check": ("radio_check", 0, False, False),
}
CONFUSE = {"grid": ["grade", "great", "grit", "grid"], "four": ["for", "power", "four"], "fower": ["power", "four", "flower"],
           "niner": ["nine", "nina", "niner", "nicer"], "fife": ["five", "fight", "fife", "fyfe"], "tree": ["three", "free", "tree"],
           "contact": ["contact", "contacts", "context"], "medevac": ["medevac", "medivac", "med evac"], "over": ["over", "over.", ""],
           "sitrep": ["sitrep", "sit rep", "set rep"], "zulu": ["zulu", "julie", "zooloo"], "whiskey": ["whiskey", "whisky", "risky"]}
DIG = {"zero": "0", "one": "1", "two": "2", "three": "3", "tree": "3", "four": "4", "fower": "4", "five": "5", "fife": "5",
       "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9"}


def asr_noise(text: str, rng: random.Random, p: float) -> str:
    words = re.findall(r"[A-Za-z\-']+|\d+", text.lower())
    out = []
    for w in words:
        r = rng.random()
        if r < p * 0.3:                     # deletion
            continue
        if w in CONFUSE and r < p * 1.6:
            w = rng.choice(CONFUSE[w])
        elif w in DIG and r < p * 1.2:      # whisper writes numerals
            w = DIG[w]
        out.append(w)
        if rng.random() < p * 0.1:          # insertion
            out.append(rng.choice(["the", "a", "uh", "and", "is"]))
    s = " ".join(x for x in out if x)
    return re.sub(r"(\d) (?=\d)", r"\1", s) if rng.random() < 0.5 else s


def gold_for(kind: str, rng: random.Random) -> dict:
    mt, urg, enemy, cas = KIND_TO[kind]
    keys = list(QUESTIONS["msg_type"]["criteria"])
    soft = 0.9
    probs = {k: (soft if k == mt else (1 - soft) / (len(keys) - 1)) for k in keys}
    lv = np.full(3, 0.05)
    lv[urg] = 0.9
    lv = lv / lv.sum()
    return {"msg_type": {"label": mt, "probabilities": probs},
            "urgency": {"label": urg, "score": float(urg), "probabilities": {str(i): float(v) for i, v in enumerate(lv)}},
            "enemy_contact": {"label": "true" if enemy else "false", "noul": 0.92 if enemy else 0.08,
                              "probabilities": {"true": 0.92 if enemy else 0.08, "false": 0.08 if enemy else 0.92}},
            "casualties": {"label": "true" if cas else "false", "noul": 0.92 if cas else 0.08,
                           "probabilities": {"true": 0.92 if cas else 0.08, "false": 0.08 if cas else 0.92}}}


def case(i: int, text: str, kind: str, source: str, rng: random.Random) -> dict:
    return {"id": f"radio-{i:06d}", "workflow": "radio_hub", "source": source, "kind": kind,
            "state": json.dumps({"transcript": text}), "questions": json.dumps(QUESTIONS),
            "gold": json.dumps(gold_for(kind, rng))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--out", type=Path, default=ROOT / "finetune" / "data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    rows = {"train": [], "val": []}
    k = 0
    for split, n, seeds in (("train", a.n, range(1000, 1000 + 400)), ("val", a.n // 10, range(9000, 9040))):
        per = max(1, n // len(seeds))
        for s in seeds:
            sc = Scenario(seed=s)
            for _ in range(per):
                sc.advance(rng.uniform(10, 60))
                ev = sc.next_event()
                src = rng.choices(["clean", "asr_sim"], [0.3, 0.7])[0]
                txt = ev.text if src == "clean" else asr_noise(ev.text, rng, rng.uniform(0.05, 0.3))
                rows[split].append(case(k, txt, ev.kind, src, rng))
                k += 1
    # real full-chain transcripts from sandbox runs -> held-out test
    test = []
    for rep in sorted((ROOT / "runs").glob("*/report.json")):
        for r in json.loads(rep.read_text(encoding="utf-8")):
            if r.get("asr_text") and r.get("kind") in KIND_TO:
                test.append(case(k, r["asr_text"], r["kind"], "asr_real", rng))
                k += 1
    rows["test"] = test
    for split, rs in rows.items():
        with (a.out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
        kinds = {}
        for r in rs:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(split, len(rs), kinds)


if __name__ == "__main__":
    main()
