"""Listening sheet for a built or streamed battlefield set: one row per pair
with boom, ref and target players, loudness-matched playback copies so the
ear judges realism rather than level. ``ancdata battlefield-listen --n 10``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .audio import EPS, db_to_lin, write_wav
from .battlefield import build_battlefield_chain
from .loudness import loudness_lufs


def _playback(x: np.ndarray, lufs: float = -23.0) -> np.ndarray:
    g = min(db_to_lin(lufs - loudness_lufs(x)), 0.99 / (float(np.abs(x).max()) + EPS))
    return (x * g).astype(np.float32)


def listen_sheet(cfg_path: Path, split: str, n: int, out: Path, start: int = 0, embed: bool = True) -> Path:
    chain = build_battlefield_chain(cfg_path, split)
    out = Path(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(start, start + n):
        pair = chain.generate(k)
        m = pair.meta
        files = {}
        for name, x in (("boom", pair.noisy[0]), ("ref", pair.noisy[1]), ("target", pair.clean)):
            rel = f"audio/{k:06d}_{name}.wav"
            write_wav(out / rel, _playback(x), fmt="pcm16")
            files[name] = rel
        ev = ", ".join(f"{e.category}@{e.start / 16000:.1f}s({e.ratio_db:+.0f}dB)" for e in pair.events
                       if e.category != "hard_negative")
        rows.append(dict(k=k, scenario=m["scenario"], snippet=m["speech_id"], effort=m["speech_effort"],
                         snr=m["snr_lufs_db"], eff=m["snr_effective_db"], clip=m["clip_pct"], events=ev, files=files))
    (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    def page_html(embed_audio: bool) -> str:
        def src(rel: str) -> str:
            if not embed_audio:
                return rel
            return "data:audio/wav;base64," + base64.b64encode((out / rel).read_bytes()).decode("ascii")
        trs = []
        for r in rows:
            cells = "".join(f'<td><audio controls preload="none" src="{src(r["files"][c])}"></audio></td>'
                            for c in ("boom", "ref", "target"))
            trs.append(f'<tr><td class="k">{r["k"]:06d}</td><td class="sc">{r["scenario"]}<br><small>{r["snippet"]} · '
                       f'{r["effort"]} · SNR {r["snr"]:.0f} dB (eff {r["eff"]:.0f})<br>{r["events"] or "no transients"}'
                       f'</small></td>{cells}</tr>')
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>battlefield_v3 {split}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}}
table{{border-collapse:collapse}}td,th{{padding:6px 10px;border-bottom:1px solid #ddd;vertical-align:middle}}
th{{text-align:left;background:#eee}}td.k{{font-weight:600}}td.sc{{min-width:260px}}audio{{width:230px}}</style></head>
<body><h2>battlefield_v3 — {split} pairs {start}..{start + n - 1}</h2>
<p>Final recipe (S2 + reduced gunfire, 50 % of clips with small-arms fire). boom = model input, ref = ear-cup
reference mic, target = dry speech x AGC gain. Playback loudness-matched to -23 LUFS.</p>
<table><tr><th>#</th><th>scene</th><th>boom</th><th>ref</th><th>target</th></tr>{''.join(trs)}</table></body></html>"""

    (out / "listen.html").write_text(page_html(False), encoding="utf-8")
    page = out / "listen.html"
    if embed:
        page = out / "listen_embedded.html"
        page.write_text(page_html(True), encoding="utf-8")
    return page
