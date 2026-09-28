"""Base vs fine-tuned Laya on the radio-hub cases: accuracy per question, per text source, latency.

    python finetune/eval_laya.py --data finetune/data [--model artefacts/laya_radio] [--split val]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def run(agent, rows):
    acc: dict[str, list[float]] = {}
    lat = []
    for r in rows:
        st, q, g = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
        t = time.perf_counter()
        res = agent.predict(st, q)
        lat.append((time.perf_counter() - t) * 1000)
        a = res["answers"]
        for qid, qd in q.items():
            if qd["type"] == "choice":
                ok = a[qid]["choice"] == g[qid]["label"]
            elif qd["type"] == "noul":
                ok = (a[qid]["noul"] >= 0.5) == (g[qid]["label"] == "true")
            else:
                probs = [a[qid]["probabilities"].get(str(i), 0.0) for i in range(len(qd["criteria"]))]
                ok = int(np.argmax(probs)) == int(g[qid]["label"])
            acc.setdefault(qid, []).append(float(ok))
            acc.setdefault(f"{qid}@{r['source']}", []).append(float(ok))
    return {k: round(float(np.mean(v)), 3) for k, v in sorted(acc.items())}, float(np.median(lat))


def main() -> None:
    import laya
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (a.data / f"{a.split}.jsonl").open(encoding="utf-8")][: a.n]
    out = {}
    for name, agent in (("base", laya.load("convaiinnovations/laya", device="cuda")),) + \
                       ((("finetuned", laya.Agent(str(a.model), device="cuda")),) if a.model else ()):
        acc, lat = run(agent, rows)
        out[name] = {"accuracy": acc, "median_ms": lat}
        print(name, json.dumps(out[name]), flush=True)
    if a.model:
        (a.model / f"eval_{a.split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
