"""v3 vs v4 listening sheet: one row per scenario, same speech snippet in both
(same seed and index), loudness-matched playback. Players per row: v4 boom,
v4 scene only (the noise through the same capsule channel, no speech), v3 boom.
Label chips are coloured by coarse group; masked labels are struck through.

    ancdata battlefield-compare --v4 configs/battlefield_v4.yaml --v3 configs/battlefield.yaml
"""
from __future__ import annotations

import base64
import copy
import html
import json
from pathlib import Path

import numpy as np

from .audio import write_wav
from .battlefield import build_battlefield_chain
from .battlefield_v4 import BattlefieldChainV4, load_v4_config
from .channel import capsule_channel
from .listen import _playback

COARSE_COLOURS = {
    "weapons": "#e5484d", "explosion": "#f76b15", "ground_vehicle": "#8e6c3a", "aircraft": "#3e63dd",
    "weather": "#0d9488", "ambience": "#46a758", "human": "#d6409f", "alarm": "#f5a524", "channel": "#8e4ec6",
    "impact": "#6b7280", "other": "#6b7280",
}


def compare_sheet(v4_cfg: Path, v3_cfg: Path | None, split: str, out: Path, per_scenario: int = 1,
                  search: int = 400, embed: bool = True) -> Path:
    cfg = load_v4_config(v4_cfg)
    cfg = copy.deepcopy(cfg)
    cfg["debug_stems"] = True
    ch4 = BattlefieldChainV4(cfg, split)
    ch3 = build_battlefield_chain(v3_cfg, split) if v3_cfg else None
    out = Path(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    want = {s["name"]: per_scenario for s in cfg["scenarios"]}
    picks: list[int] = []
    for k in range(search):
        name = ch4.plan(k)["scenario"]
        if want.get(name, 0) > 0:
            want[name] -= 1
            picks.append(k)
        if not any(want.values()):
            break
    rows = []
    for k in picks:
        pair = ch4.generate(k)
        m = pair.meta
        st = m["_stems"]
        noise_only, _, _ = capsule_channel(st["scene"] + st["transients"], np.random.default_rng(0), cfg["channel"])
        files = {}
        clips = [("v4_boom", pair.noisy[0]), ("noise_only", noise_only), ("target", pair.clean)]
        env = np.abs(pair.clean)
        on = np.where(env > 0.05 * env.max())[0]
        speech_span = f"speech {on[0] / 16000:.1f}-{on[-1] / 16000:.1f} s" if len(on) else "speech ?"
        if ch3 is not None:
            clips.append(("v3_boom", ch3.generate(k).noisy[0]))
        for name, x in clips:
            rel = f"audio/{k:06d}_{name}.wav"
            write_wav(out / rel, _playback(x), fmt="pcm16")
            files[name] = rel
        chips = []
        for e in sorted(pair.events, key=lambda e: (e.start, e.category)):
            p = e.params
            col = COARSE_COLOURS.get(str(p.get("coarse", "other")), "#6b7280")
            span = "whole clip" if (e.start == 0 and e.end == pair.clean.shape[0]) else f"{e.start / 16000:.1f}-{e.end / 16000:.1f}s"
            dist = "" if p.get("dist_cls") in (None, "dry") else f" · {p['dist_cls']} {p.get('dist_m', 0):.0f} m"
            cls = "chip masked" if p.get("masked") else "chip"
            chips.append(f'<span class="{cls}" style="--c:{col}" title="{html.escape(str(p.get("source", "")))}">'
                         f'{html.escape(e.category)}<small>{span}{dist}</small></span>')
        rows.append(dict(k=k, scenario=m["scenario"], env=m["environment"], rt60=m["rt60_s"], snr=m["snr_lufs_db"],
                         effort=m["speech_effort"], snippet=m["speech_id"], span=speech_span, n_labels=m["n_labels"], chips="".join(chips),
                         files=files))
    (out / "rows.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "chips"} for r in rows], indent=1),
                                   encoding="utf-8")

    def page(embed_audio: bool) -> str:
        def src(rel: str) -> str:
            if not embed_audio:
                return rel
            return "data:audio/wav;base64," + base64.b64encode((out / rel).read_bytes()).decode("ascii")
        cols = ["v4_boom", "v3_boom", "target", "noise_only"] if ch3 is not None else ["v4_boom", "target", "noise_only"]
        names = {"v4_boom": "v4 boom (speech + noise)", "v3_boom": "v3 boom (speech + noise)", "target": "target (clean speech)",
                 "noise_only": "v4 noise only &mdash; no speech, by design"}
        head = "".join(f"<th>{names[c]}</th>" for c in cols)
        trs = []
        for i, r in enumerate(rows):
            cells = "".join(f'<td><audio controls preload="none" src="{src(r["files"][c])}"></audio></td>' for c in cols)
            trs.append(f'<tr><td class="n">{i + 1}</td><td class="sc"><b>{html.escape(r["scenario"])}</b>'
                       f'<span class="env">{r["env"]} · RT60 {r["rt60"]:.2f} s</span>'
                       f'<div class="meta">#{r["k"]:06d} · {html.escape(r["snippet"])} · {r["effort"]} · SNR {r["snr"]:.0f} dB · '
                       f'{r["n_labels"]} labels · <b>{r["span"]}</b></div><div class="chips">{r["chips"]}</div></td>{cells}</tr>')
        legend = "".join(f'<span class="chip" style="--c:{c}">{g.replace("_", " ")}</span>' for g, c in COARSE_COLOURS.items()
                         if g != "other")
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Battlefield v4 listening</title><style>
:root{{--bg:#f6f7fb;--card:#fff;--ink:#1c2024;--mute:#60646c;--line:#e4e6ee;--acc1:#3e63dd;--acc2:#e5484d;--acc3:#f5a524}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111318;--card:#1a1d24;--ink:#edeef0;--mute:#9ba1a6;--line:#2b2f38}}}}
:root[data-theme="dark"]{{--bg:#111318;--card:#1a1d24;--ink:#edeef0;--mute:#9ba1a6;--line:#2b2f38}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}
header{{padding:28px 24px 22px;background:linear-gradient(120deg,var(--acc1),#8e4ec6 45%,var(--acc2) 80%,var(--acc3));color:#fff}}
header h1{{margin:0 0 6px;font-size:26px}}header p{{margin:0;max-width:980px;opacity:.95}}
main{{padding:18px 16px 40px;max-width:1500px;margin:auto}}
.legend{{margin:4px 0 16px;display:flex;flex-wrap:wrap;gap:6px}}
.wrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:middle;text-align:left}}
th{{position:sticky;top:0;background:var(--card);font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}}
tr:nth-child(even) td{{background:color-mix(in srgb,var(--acc1) 3%,transparent)}}
td.n{{font-weight:700;color:var(--acc1);font-size:18px}}td.sc{{min-width:330px}}
.env{{margin-left:8px;padding:1px 8px;border-radius:99px;background:color-mix(in srgb,var(--acc1) 14%,transparent);color:var(--acc1);font-size:12px}}
.meta{{color:var(--mute);font-size:12px;margin:3px 0 6px}}.chips{{display:flex;flex-wrap:wrap;gap:5px}}
.chip{{display:inline-flex;gap:6px;align-items:baseline;padding:2px 9px;border-radius:99px;font-size:12.5px;font-weight:600;
color:var(--c);background:color-mix(in srgb,var(--c) 14%,transparent);border:1px solid color-mix(in srgb,var(--c) 45%,transparent)}}
.chip small{{font-weight:400;opacity:.8}}.chip.masked{{text-decoration:line-through;opacity:.5}}
audio{{width:210px;height:36px}}footer{{color:var(--mute);font-size:13px;margin-top:14px}}
</style></head><body><header><h1>Battlefield v4 — one space, few objects</h1>
<p>Same speech snippet in every column. <b>v4 boom</b> = new layering (shared outdoor/urban/forest/cabin acoustics, distance classes,
≤ 6 labels and ≤ 2 overlapping foreground events per clip, onset-aligned events, no loops). <b>v4 noise only</b> (last column) = the same noise with the speech
removed, so you can judge the layering on its own &mdash; it has no speech on purpose. GRID sentences are ~2 s, so each clip has speech only in
the span shown per row. <b>v3 boom</b> = the current dataset's recipe for comparison. All players loudness-matched to −23 LUFS.</p></header>
<main><div class="legend">{legend}</div><div class="wrap"><table><tr><th>#</th><th>scene · labels</th>{head}</tr>{''.join(trs)}</table></div>
<footer>{split} split · {len(rows)} clips · chips show label, time span, distance class; struck-through = masked (inaudible under the mix).</footer></main></body></html>"""

    (out / "listen.html").write_text(page(False), encoding="utf-8")
    page_path = out / "listen.html"
    if embed:
        page_path = out / "listen_embedded.html"
        page_path.write_text(page(True), encoding="utf-8")
    return page_path


def ab_sheet(cfg_a: Path, cfg_b: Path, split: str, out: Path, per_scenario: int = 1, search: int = 400) -> Path:
    """Blind-free A/B for two v3-family configs (e.g. v3 vs v3.1): one row per scenario of B,
    same index (same snippet and same v3 draws), boom A / boom B / target / B noise only."""
    from .battlefield import BattlefieldChain, load_battlefield_config
    ca, cb = load_battlefield_config(cfg_a), copy.deepcopy(load_battlefield_config(cfg_b))
    cb["debug_stems"] = True
    cha, chb = BattlefieldChain(ca, split), BattlefieldChain(cb, split)
    out = Path(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    want = {s["name"]: per_scenario for s in cb["scenarios"]}
    picks = []
    for k in range(search):
        name = chb.plan(k)["scenario"]
        if want.get(name, 0) > 0:
            want[name] -= 1
            picks.append(k)
        if not any(want.values()):
            break
    rows = []
    for k in picks:
        pb = chb.generate(k)
        m = pb.meta
        st = m["_stems"]
        noise_only, _, _ = capsule_channel(st["scene"] + st["transients"], np.random.default_rng(0), cb["channel"])
        env = np.abs(pb.clean)
        on = np.where(env > 0.05 * env.max())[0]
        files = {}
        for name, x in (("a", cha.generate(k).noisy[0]), ("b", pb.noisy[0]), ("target", pb.clean), ("noise", noise_only)):
            rel = f"audio/{k:06d}_{name}.wav"
            write_wav(out / rel, _playback(x), fmt="pcm16")
            files[name] = "data:audio/wav;base64," + base64.b64encode((out / rel).read_bytes()).decode("ascii")
        chips = "".join(f'<span class="chip" style="--c:#3e63dd">{html.escape(l)}</span>' for l in m.get("labels", []))
        rows.append((k, m, f"{on[0] / 16000:.1f}-{on[-1] / 16000:.1f} s" if len(on) else "?", chips, files))
    trs = "".join(
        f'<tr><td class="n">{i + 1}</td><td><b>{html.escape(m["scenario"])}</b><div class="meta">#{k:06d} · {html.escape(m["speech_id"])} · '
        f'SNR {m["snr_lufs_db"]:.0f} dB · <b>speech {span}</b>{" · inside vehicle" if m.get("interior") else ""}</div>'
        f'<div class="chips">{chips}</div></td>'
        + "".join(f'<td><audio controls preload="none" src="{f[c]}"></audio></td>' for c in ("a", "b", "target", "noise")) + "</tr>"
        for i, (k, m, span, chips, f) in enumerate(rows))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v3 vs v3.1 layering</title><style>
:root{{--bg:#f6f7fb;--card:#fff;--ink:#1c2024;--mute:#60646c;--line:#e4e6ee}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111318;--card:#1a1d24;--ink:#edeef0;--mute:#9ba1a6;--line:#2b2f38}}}}
:root[data-theme="dark"]{{--bg:#111318;--card:#1a1d24;--ink:#edeef0;--mute:#9ba1a6;--line:#2b2f38}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}}
header{{padding:26px 24px;background:linear-gradient(120deg,#0d9488,#3e63dd 50%,#8e4ec6);color:#fff}}header h1{{margin:0 0 6px}}
main{{padding:16px;max-width:1400px;margin:auto}}.wrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}}td.n{{font-weight:700;color:#3e63dd;font-size:18px}}
.meta{{color:var(--mute);font-size:12px;margin:3px 0 6px}}.chips{{display:flex;flex-wrap:wrap;gap:5px}}
.chip{{padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600;color:var(--c);background:color-mix(in srgb,var(--c) 14%,transparent)}}
audio{{width:210px;height:36px}}</style></head><body><header><h1>v3 vs v3.1 — same sound, fixed layering logic</h1>
<p>Same speech and same scene draw per row. <b>v3</b> = current dataset. <b>v3.1</b> = no looped beds, textures start at real onsets without
their own room tone, pass-bys sit in front of the bed, bursts never start on top of each other, one shared tail space per clip,
vehicle-interior scenes hear outside sound through the hull. Last column = v3.1 noise only (no speech, on purpose). All −23 LUFS.</p></header>
<main><div class="wrap"><table><tr><th>#</th><th>scene · labels</th><th>v3 boom</th><th>v3.1 boom</th><th>target (clean speech)</th>
<th>v3.1 noise only</th></tr>{trs}</table></div></main></body></html>"""
    path = out / "listen_embedded.html"
    path.write_text(page, encoding="utf-8")
    return path


def radio_sheet(cfg_path: Path, split: str, out: Path, per_link: int = 4) -> Path:
    """Studio vs radio for the same clip: v3.1 boom without the link, the same boom over each link
    type, the radio target (band-limited clean) and the studio target."""
    from .battlefield import BattlefieldChain, load_battlefield_config
    from .radio import radio_link
    cfg = load_battlefield_config(cfg_path)
    ch = BattlefieldChain({**cfg, "radio": None}, split)
    rparams = {k: v for k, v in cfg["radio"].items() if k != "p"}
    out = Path(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    k = 0
    for link in rparams["link_p"]:
        for _ in range(per_link):
            pair = ch.generate(k)
            y, tgt, _, info = radio_link(pair.noisy[0], pair.clean, np.random.default_rng(1000 + k),
                                         {**rparams, "link_p": {link: 1.0}})
            files = {}
            for name, x in (("studio", pair.noisy[0]), ("radio", y), ("radio_target", tgt), ("studio_target", pair.clean)):
                rel = f"audio/{k:06d}_{name}.wav"
                write_wav(out / rel, _playback(x), fmt="pcm16")
                files[name] = "data:audio/wav;base64," + base64.b64encode((out / rel).read_bytes()).decode("ascii")
            desc = ", ".join(f"{a} {v:.3g}" if isinstance(v, float) else f"{a} {v}" for a, v in info.items() if a != "link")
            rows.append((k, link, pair.meta, desc, files))
            k += 1
    colour = {"cvsd16": "#e5484d", "fm": "#0d9488", "cvsd32": "#3e63dd"}
    trs = "".join(
        f'<tr><td class="n">{i + 1}</td><td><span class="chip" style="--c:{colour.get(link, "#8e4ec6")}">{link}</span> '
        f'<b>{html.escape(m["scenario"])}</b><div class="meta">SNR {m["snr_lufs_db"]:.0f} dB · {html.escape(desc)}</div></td>'
        + "".join(f'<td><audio controls preload="none" src="{f[c]}"></audio></td>' for c in ("studio", "radio", "radio_target", "studio_target"))
        + "</tr>" for i, (k, link, m, desc, f) in enumerate(rows))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Radio link listening</title><style>
:root{{--bg:#f6f7fb;--card:#fff;--ink:#1c2024;--mute:#60646c;--line:#e4e6ee}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111318;--card:#1a1d24;--ink:#edeef0;--mute:#9ba1a6;--line:#2b2f38}}}}
:root[data-theme="dark"]{{--bg:#111318;--card:#1a1d24;--ink:#edeef0;--mute:#9ba1a6;--line:#2b2f38}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}}
header{{padding:26px 24px;background:linear-gradient(120deg,#e5484d,#f76b15 40%,#0d9488 75%,#3e63dd);color:#fff}}header h1{{margin:0 0 6px}}
main{{padding:16px;max-width:1400px;margin:auto}}.wrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}}td.n{{font-weight:700;color:#e5484d;font-size:18px}}
.meta{{color:var(--mute);font-size:12px;margin-top:4px}}
.chip{{padding:2px 9px;border-radius:99px;font-size:12px;font-weight:700;color:var(--c);background:color-mix(in srgb,var(--c) 14%,transparent)}}
audio{{width:210px;height:36px}}</style></head><body><header><h1>Studio vs tactical radio</h1>
<p>Same v3.1 clip per row. <b>cvsd16</b> = 16 kbit/s CVSD secure voice (SINCGARS/VINSON) with burst bit errors; <b>fm</b> = analog narrowband
FM with hiss, fading, clicks and squelch tail; <b>cvsd32</b> = 32 kbit/s CVSD. Some rows lose the first syllable to VOX/PTT keying on purpose.
<b>radio target</b> = what the model is asked to output for a radio clip (clean speech, radio band only). All −23 LUFS.</p></header>
<main><div class="wrap"><table><tr><th>#</th><th>link · scene</th><th>studio input (v3.1)</th><th>radio input</th><th>radio target</th>
<th>studio target</th></tr>{trs}</table></div></main></body></html>"""
    path = out / "listen_embedded.html"
    path.write_text(page, encoding="utf-8")
    return path


def built_sheet(root: Path, out: Path, n: int = 16, seed: int = 0) -> Path:
    """Listening sheet straight from a BUILT split: the model's input (radio boom), the radio-band target,
    with scene labels, link type, speech corpus and transcript. Rows spread over links and corpora."""
    import soundfile as sf
    from .radio import link_filters
    root, out = Path(root), Path(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    meta = [json.loads(l) for l in (root / "meta.jsonl").open(encoding="utf-8") if l.strip()]
    rng = np.random.default_rng(seed)
    by: dict[tuple, list] = {}
    for m in meta:
        by.setdefault(((m.get("radio") or {}).get("link", "studio"), m["speech_corpus"]), []).append(m)
    keys = sorted(by)
    picks = []
    while len(picks) < n and keys:
        for k in keys:
            if len(picks) < n and by[k]:
                picks.append(by[k].pop(int(rng.integers(len(by[k])))))
    rows = []
    for m in picks:
        def rd(sub):
            for ext in (".wav", ".flac"):
                p = root / sub / f"{m['id']}{ext}"
                if p.exists():
                    return sf.read(str(p), dtype="float32", always_2d=True)[0][:, 0]
        x, c = rd("noisy"), rd("clean")
        if not m.get("radio"):
            c = link_filters(c, 3400.0)
        files = {}
        for name, s in (("input", x), ("target", c)):
            rel = f"audio/{m['id']}_{name}.wav"
            write_wav(out / rel, _playback(s), fmt="pcm16")
            files[name] = "data:audio/wav;base64," + base64.b64encode((out / rel).read_bytes()).decode("ascii")
        rows.append((m, files))
    colour = {"cvsd16": "#e5484d", "fm": "#0d9488", "cvsd32": "#3e63dd", "studio": "#8e4ec6"}
    trs = []
    for i, (m, f) in enumerate(rows):
        link = (m.get("radio") or {}).get("link", "studio")
        chips = "".join(f'<span class="chip" style="--c:#46a758">{html.escape(l)}</span>' for l in m.get("labels", []))
        trs.append(f'<tr><td class="n">{i + 1}</td><td><span class="chip" style="--c:{colour[link]}">{link}</span> '
                   f'<b>{html.escape(m["scenario"])}</b> · {html.escape(m["speech_corpus"])} · SNR {m["snr_lufs_db"]:.0f} dB'
                   f'<div class="meta">{html.escape(str(m.get("speech_transcript") or "")[:160])}</div><div class="chips">{chips}</div></td>'
                   f'<td><audio controls preload="none" src="{f["input"]}"></audio></td>'
                   f'<td><audio controls preload="none" src="{f["target"]}"></audio></td></tr>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Battlefield v3.1 dataset</title><style>
:root{{--bg:#f6f7fb;--card:#fff;--ink:#1c2024;--mute:#60646c;--line:#e4e6ee}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111318;--card:#1a1d24;--ink:#edeef0;--mute:#9ba1a6;--line:#2b2f38}}}}
:root[data-theme="dark"]{{--bg:#111318;--card:#1a1d24;--ink:#edeef0;--mute:#9ba1a6;--line:#2b2f38}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}}
header{{padding:26px 24px;background:linear-gradient(120deg,#46a758,#0d9488 35%,#3e63dd 70%,#8e4ec6);color:#fff}}header h1{{margin:0 0 6px}}
main{{padding:16px;max-width:1300px;margin:auto}}.wrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}}td.n{{font-weight:700;color:#0d9488;font-size:18px}}
.meta{{color:var(--mute);font-size:12px;margin:3px 0 6px}}.chips{{display:flex;flex-wrap:wrap;gap:5px}}
.chip{{padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600;color:var(--c);background:color-mix(in srgb,var(--c) 14%,transparent)}}
audio{{width:230px;height:36px}}</style></head><body><header><h1>Battlefield v3.1 — the built dataset</h1>
<p>Rows straight from <code>{html.escape(str(root))}</code>. <b>input</b> = what the hub model hears (battlefield noise at the talker's mic, then the
radio link); <b>target</b> = clean speech, radio band. Speech now includes air-traffic radio (atcosim), spoken figures / directions
(speechcommands) and generated radio procedure (tts, train/val only). Green chips = audible scene labels. All −23 LUFS.</p></header>
<main><div class="wrap"><table><tr><th>#</th><th>link · scene · speech · labels</th><th>input</th><th>target</th></tr>{''.join(trs)}</table></div></main></body></html>"""
    path = out / "listen_embedded.html"
    path.write_text(page, encoding="utf-8")
    return path
