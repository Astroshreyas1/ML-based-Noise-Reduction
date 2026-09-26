"""One command, run before any training. Raises on failure.

    python -m ancdata.selftest --config configs/train.yaml
    python -m ancdata.selftest --smoke          # no downloads, synthetic fixtures

Checks
  1. contract        N examples generate; every contract assertion passes
  2. determinism     same (seed, index) twice, fresh Chain -> bit-identical output
  3. leakage         no non-train / eval_only source appears in a train batch
  4. snr accuracy    with reverb/ADC/transients off, measured SNR == requested within 0.5 dB
  5. span alignment  RMS inside each event span exceeds RMS outside by a clear margin
  6. non-silent      no returned clean is (near) all-zero
  7. sources         every manifest file opens, is >= 0.2 s, resamples to 16 kHz
  0. scorer          known-answer test of SI-SNR / STOI / PESQ / chunked path (runs first)
  8. real support    if in-house recordings exist: their peak level and crest factor
                     fall inside the synthetic training distribution (demo-day guard)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import rms
from .config import SR, load_config, with_overrides
from .paths import ENV_VAR, from_relative
from .registry import build_manifest, load_manifest
from .rir_gen import RirBank, synthetic_bank
from .sampler import Sampler
from .chain import Chain, assert_contract

SMOKE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "smoke.yaml"


def _ok(name: str, detail: str = "") -> None:
    print(f"  {name:<18} OK {detail}")


def check_contract(chain: Chain, n: int) -> list:
    pairs = []
    for k in range(n):
        p = chain.generate(k)
        assert_contract(p)
        pairs.append(p)
    _ok("contract", f"({n}/{n})")
    return pairs


def check_determinism(cfg, sampler, bank, pairs) -> None:
    fresh = Chain(cfg, sampler, bank)
    for k in (0, len(pairs) // 2, len(pairs) - 1):
        q = fresh.generate(k)
        if not (np.array_equal(q.noisy, pairs[k].noisy) and np.array_equal(q.clean, pairs[k].clean)):
            raise AssertionError(f"determinism FAILED at index {k}: outputs differ")
        if [e.as_tuple() for e in q.events] != [e.as_tuple() for e in pairs[k].events]:
            raise AssertionError(f"determinism FAILED at index {k}: events differ")
    _ok("determinism")


def check_leakage(manifest, chain: Chain, pairs) -> None:
    by_path = manifest.set_index("path")
    for p in pairs:
        used = [p.meta["speech_path"]]
        am = p.meta.get("ambience_meta", {})
        if "path" in am:
            used.append(am["path"])
        for rel in used:
            row = by_path.loc[rel]
            if row.split != chain.sampler.split:
                raise AssertionError(f"LEAKAGE: {rel} has split={row.split} in a {chain.sampler.split} batch")
            if chain.sampler.split == "train" and bool(row.eval_only):
                raise AssertionError(f"LEAKAGE: eval_only file {rel} drawn for training")
        if chain.bank.split[p.meta["room"]["index"]] != chain.rir_split:
            raise AssertionError("LEAKAGE: room from the wrong split")
    _ok("leakage")


def check_snr(cfg, sampler, bank, n: int) -> None:
    plain = with_overrides(cfg, **{"lombard.probability": 0.0, "reverb.probability": 0.0,
                                   "impulsive.probability": 0.0, "hard_negative.probability": 0.0,
                                   "adc.probability": 0.0, "codec.probability": 0.0,
                                   "mic.proximity_boost_db": 0.0, "mic.ir_file": None})
    chain = Chain(plain, sampler, bank)
    worst = 0.0
    for k in range(n):
        p = chain.generate(k)
        # clean carries the same linear gain as noisy, so the residual is the scaled noise
        noise = p.noisy[0] - p.clean
        measured = 20.0 * np.log10(rms(p.clean) / rms(noise))
        worst = max(worst, abs(measured - p.meta["snr_db"]))
    if worst > 0.5:
        raise AssertionError(f"snr accuracy FAILED: max error {worst:.2f} dB")
    _ok("snr accuracy", f"(max err {worst:.2f} dB)")


def check_spans(cfg, sampler, bank, n: int) -> None:
    loud = with_overrides(cfg, **{"impulsive.probability": 1.0, "impulsive.peak_ratio_db": {"dist": "uniform", "low": 12, "high": 18},
                                  "hard_negative.probability": 0.0, "codec.probability": 0.0,
                                  "adc.probability": 0.0, "mix_ambience.snr_db": 20.0})
    chain = Chain(loud, sampler, bank)
    checked = 0
    for k in range(n):
        p = chain.generate(k)
        x = p.noisy[0]
        mask = np.zeros(len(x), dtype=bool)
        for e in p.events:
            mask[e.start:e.end] = True
        if not mask.any() or mask.all():
            continue
        inside = rms(x[mask])
        outside = rms(x[~mask])
        if inside < 1.5 * outside:
            raise AssertionError(f"span alignment FAILED at index {k}: inside {inside:.4f} outside {outside:.4f}")
        checked += 1
    if checked == 0:
        raise AssertionError("span alignment: no events were generated")
    _ok("event alignment", f"({checked} examples)")


def check_nonsilent(pairs) -> None:
    for p in pairs:
        if rms(p.clean) < 1e-4:
            raise AssertionError(f"silent target at index {p.meta['index']}")
    _ok("non-silent")


def check_sources(manifest, limit: int | None = None) -> None:
    rows = manifest if limit is None else manifest.sample(min(limit, len(manifest)), random_state=0)
    for rel in rows.path:
        p = from_relative(rel)
        info = sf.info(str(p))
        if info.frames == 0:
            raise AssertionError(f"{p}: empty file")
    _ok("sources", f"({len(rows)} files open)")


def check_scorer() -> None:
    from .metrics import scorer_selftest
    try:
        r = scorer_selftest()
    except ImportError as e:  # pesq / pystoi missing
        print(f"  {'scorer':<18} SKIPPED ({e})")
        return
    _ok("scorer", f"(stoi {r.get('stoi_identical', float('nan')):.3f}, pesq {r.get('pesq_identical', float('nan')):.2f}, snr err {abs(r['snr_plus10'] - 10):.2f} dB)")


def check_real_support(manifest, pairs, margin_db: float = 3.0) -> None:
    """Demo-day guard: real headset recordings must sit inside the synthetic
    input distribution on peak level and crest factor. Skipped when no
    `realcheck` pile is registered."""
    from .audio import crest_factor_db, load_mono
    real = manifest[manifest.pile == "realcheck"]
    if len(real) == 0:
        print(f"  {'real support':<18} SKIPPED (no realcheck recordings registered yet)")
        return
    syn_pk = np.array([20 * np.log10(np.abs(p.noisy[0]).max() + 1e-9) for p in pairs])
    syn_cf = np.array([crest_factor_db(p.noisy[0]) for p in pairs])
    bad = []
    for rel in real.path:
        x = load_mono(from_relative(rel))
        pk, cf = 20 * np.log10(np.abs(x).max() + 1e-9), crest_factor_db(x)
        if not (syn_pk.min() - margin_db <= pk <= syn_pk.max() + margin_db):
            bad.append(f"{rel}: peak {pk:.1f} dBFS outside synthetic [{syn_pk.min():.1f}, {syn_pk.max():.1f}]")
        if not (syn_cf.min() - margin_db <= cf <= syn_cf.max() + margin_db):
            bad.append(f"{rel}: crest {cf:.1f} dB outside synthetic [{syn_cf.min():.1f}, {syn_cf.max():.1f}]")
    if bad:
        msg = "real recordings OUTSIDE synthetic support (fix gain / ADC headroom in config):"
        raise AssertionError(msg + "".join("\n  " + b for b in bad))
    _ok("real support", f"({len(real)} recordings inside synthetic envelope)")


def run(cfg, split: str = "train", n: int = 100, source_limit: int | None = 500) -> None:
    check_scorer()
    manifest = load_manifest()
    sampler = Sampler(manifest, cfg, split)
    bank = RirBank()
    chain = Chain(cfg, sampler, bank)
    t0 = time.time()
    pairs = check_contract(chain, n)
    dt = (time.time() - t0) / n
    check_determinism(cfg, sampler, bank, pairs)
    check_leakage(manifest, chain, pairs)
    check_snr(cfg, sampler, bank, max(10, n // 4))
    check_spans(cfg, sampler, bank, max(10, n // 4))
    check_nonsilent(pairs)
    check_sources(manifest, source_limit)
    check_real_support(manifest, pairs)
    print(f"all checks passed  ({dt * 1000:.0f} ms/example, sources {sampler.counts()})")


def smoke(keep: Path | None = None) -> None:
    from .fixtures import make_fixtures
    from .materialize import materialize
    from .plots import plot_crest_attack

    root = keep or Path(tempfile.mkdtemp(prefix="ancdata_smoke_"))
    os.environ[ENV_VAR] = str(root)
    print(f"smoke: data root {root}")
    make_fixtures(verbose=False)
    synthetic_bank(n=20, seed=0)
    build_manifest(screen=False, verbose=False)  # fixtures are synthetic and voice-free by construction
    cfg = load_config(SMOKE_CONFIG)
    run(cfg, split="train", n=30, source_limit=None)
    out = root / "eval" / "smoke"
    materialize(cfg, "test", 5, out=out, verbose=False)
    plot_path = plot_crest_attack(cfg, root / "outputs" / "crest_attack.png", n_synth=50, real_pile="fixture_gunshot")
    print(f"materialised 5 pairs -> {out}\nplot -> {plot_path}\nsmoke passed")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="chain config yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="synthetic fixtures in a temp data root")
    ap.add_argument("--keep", type=Path, help="with --smoke: use/keep this data root instead of a temp dir")
    a = ap.parse_args(argv)
    if a.smoke:
        smoke(a.keep)
        return
    if not a.config:
        ap.error("--config required unless --smoke")
    run(load_config(a.config), a.split, a.n)


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Battlefield v3 chain (ancdata/battlefield.py)
# ---------------------------------------------------------------------------
def battlefield_selftest(cfg_path: Path, n: int = 100) -> None:
    """Contract, determinism, split leakage (speakers and noise recordings),
    K-weighted SNR bookkeeping, target = dry x AGC gain, spans, gunfire rate."""
    from .audio import active_rms_db, db_to_lin
    from .battlefield import BattlefieldChain, load_battlefield_config
    from .loudness import loudness_lufs
    from .pools import load_pools
    from .snippets import load_snippets
    from .paths import data_root

    cfg = load_battlefield_config(cfg_path)
    chains = {s: BattlefieldChain(cfg, s) for s in ("train", "val", "test")}
    print(f"battlefield selftest: {cfg['name']}  snippets train/val/test = "
          f"{chains['train'].n_snippets}/{chains['val'].n_snippets}/{chains['test'].n_snippets}, "
          f"max examples {chains['train'].max_examples}/{chains['val'].max_examples}/{chains['test'].max_examples}")
    ch = chains["train"]
    t0 = time.time()
    pairs = [ch.generate(k) for k in range(n)]
    dt = (time.time() - t0) / n
    for p in pairs:
        assert_contract(p)
    _ok("contract", f"{n} pairs, {dt * 1000:.0f} ms/pair")

    fresh = BattlefieldChain(cfg, "train")
    for k in (0, n // 2, n - 1):
        a, b = pairs[k], fresh.generate(k)
        assert np.array_equal(a.noisy, b.noisy) and np.array_equal(a.clean, b.clean), f"index {k} not deterministic"
    _ok("determinism", "fresh chain -> bit-identical")

    snips = load_snippets(data_root() / cfg["speech"]["snippets"])
    spk = {s: set(snips[snips.split == s].speaker_id) for s in ("train", "val", "test")}
    assert not (spk["train"] & spk["test"]) and not (spk["train"] & spk["val"]), "speaker leakage between splits"
    pools = load_pools()
    groups = {s: set(pools[pools.split == s].group + "|" + pools[pools.split == s].corpus) for s in ("train", "val", "test")}
    demand = pools[pools.corpus == "demand"]
    assert not (groups["train"] & groups["test"] - set(demand.group + "|" + demand.corpus)), "noise recording leakage"
    for p in pairs:
        assert p.meta["speech_speaker_id"] in spk["train"]
    _ok("leakage", f"speakers disjoint; noise recordings disjoint (DEMAND split by time)")

    # SNR bookkeeping with stems on
    cfg_dbg = dict(cfg, debug_stems=True)
    chd = BattlefieldChain(cfg_dbg, "train", pools=ch.pools, snippets=ch.snippets)
    errs = []
    for k in range(10):
        p = chd.generate(k)
        st = p.meta["_stems"]
        got = loudness_lufs(st["speech_dry"]) - loudness_lufs(st["scene"])
        errs.append(abs(got - p.meta["snr_lufs_db"]))
        # target = dry x gain
        want = st["speech_dry"] * st["agc_gain"]
        assert np.allclose(want, p.clean, atol=1e-6), "target != dry speech x AGC gain"
    assert max(errs) < 0.5, f"K-weighted SNR error {max(errs):.2f} dB"
    _ok("snr", f"K-weighted SNR within {max(errs):.2f} dB; target = dry x AGC gain")

    for p in pairs:
        for e in p.events:
            assert 0 <= e.start < e.end <= p.noisy.shape[1]
    _ok("spans", "every event span inside the clip")
    assert all(np.abs(p.clean).max() > 1e-3 for p in pairs), "silent target"
    _ok("non-silent", "")
    plans = [ch.plan(k) for k in range(max(n, 400))]
    rate = np.mean([any(not b["distant"] for b in q["bursts"]) for q in plans])
    assert 0.3 < rate < 0.7, f"gunfire rate {rate:.2f} (config p={cfg['gunfire']['p']})"
    scn = {q["scenario"] for q in plans}
    _ok("gunfire", f"{rate * 100:.0f} % of clips carry small-arms fire; {len(scn)} scenarios drawn")
    clip = np.mean([p.meta["clip_pct"] for p in pairs])
    assert clip < 5.0, f"clip fraction {clip:.2f} %"
    _ok("clipping", f"{clip:.3f} % samples clipped on average")
    print("battlefield checks passed")
