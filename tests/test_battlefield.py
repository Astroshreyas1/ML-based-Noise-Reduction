"""Battlefield v3 chain. The source-dependent tests skip when the snippets /
pools are not on this machine; the pure-signal tests always run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ancdata.channel import capsule_channel, peak_hold_release
from ancdata.loudness import loudness_lufs
from ancdata.transients import audible_span, outdoor_tail

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "battlefield.yaml"


def _have_sources() -> bool:
    from ancdata.paths import data_root
    return (data_root() / "pools.parquet").exists() and (data_root() / "snippets" / "lombard6s" / "meta.parquet").exists()


needs_sources = pytest.mark.skipif(not _have_sources(), reason="battlefield sources not on this machine")


# ---- pure signal -----------------------------------------------------------
def test_peak_hold_matches_loop():
    rng = np.random.default_rng(0)
    a = np.abs(rng.standard_normal(20000)).astype(np.float32)
    a[500] = 30.0
    r = float(np.exp(-1 / 4000))
    fast = peak_hold_release(a, r, block=1024)
    e = 0.0
    slow = np.empty_like(fast)
    for i in range(len(a)):
        e = a[i] if a[i] > e else e * r
        slow[i] = e
    assert np.allclose(fast, slow, rtol=1e-3, atol=1e-4)


def test_channel_gain_is_shared_and_ducks():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32000).astype(np.float32) * 0.05
    x[16000:16050] += 1.5                                   # a shot
    y, gain, st = capsule_channel(x, np.random.default_rng(2))
    assert y.shape == x.shape and gain.shape == x.shape
    assert np.abs(y).max() <= 1.0
    assert gain[16060] < gain[1000] * 0.5                    # ducked right after the shot
    assert gain[31000] > gain[16060] * 1.5                   # released later
    assert st["agc_min_gain_db"] < -3


def test_loudness_scale_invariance_and_gate():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(48000).astype(np.float32) * 0.1
    l1 = loudness_lufs(x)
    l2 = loudness_lufs(x * 0.5)
    assert abs((l1 - l2) - 6.02) < 0.1
    assert loudness_lufs(np.zeros(48000, dtype=np.float32)) == -70.0


def test_tail_and_span():
    rng = np.random.default_rng(4)
    x = np.zeros(1600, dtype=np.float32)
    x[10] = 1.0
    y = outdoor_tail(x, 0.05, -12.0, rng)
    assert len(y) > len(x) and np.abs(y[100:]).max() > 0.01
    a, b = audible_span(y, 1000)
    assert a >= 1000 and b > a


# ---- with sources ------------------------------------------------------------
@needs_sources
def test_plan_json_and_determinism():
    from ancdata.battlefield import build_battlefield_chain
    ch = build_battlefield_chain(CFG, "val")
    p1 = ch.plan(5)
    p2 = build_battlefield_chain(CFG, "val").plan(5)
    assert json.loads(json.dumps(p1)) == json.loads(json.dumps(p2))
    a, b = ch.generate(5), ch.generate(5)
    assert np.array_equal(a.noisy, b.noisy) and np.array_equal(a.clean, b.clean)


@needs_sources
def test_contract_alignment_and_target_rule():
    from ancdata.battlefield import build_battlefield_chain, load_battlefield_config
    from ancdata.chain import assert_contract
    cfg = dict(load_battlefield_config(CFG), debug_stems=True)
    ch = build_battlefield_chain(cfg, "val")
    for k in range(4):
        p = ch.generate(k)
        assert_contract(p)
        st = p.meta["_stems"]
        assert np.allclose(st["speech_dry"] * st["agc_gain"], p.clean, atol=1e-6)
        # the target's speech sits at lag 0 in the boom channel (the minimum-phase mic EQ
        # shifts the cross-correlation peak by at most a sample or two, never a block)
        s = p.clean - p.clean.mean()
        b = p.noisy[0] - p.noisy[0].mean()
        lags = range(-40, 41)
        xc = [float(np.dot(s[max(0, -l): len(s) - max(0, l)], b[max(0, l): len(b) - max(0, -l)])) for l in lags]
        assert abs(lags[int(np.argmax(xc))]) <= 2


@needs_sources
def test_split_leakage():
    from ancdata.pools import load_pools
    from ancdata.snippets import load_snippets
    from ancdata.paths import data_root
    sn = load_snippets(data_root() / "snippets" / "lombard6s")
    spk = {s: set(sn[sn.split == s].speaker_id) for s in ("train", "val", "test")}
    assert not (spk["train"] & spk["test"]) and not (spk["train"] & spk["val"]) and not (spk["val"] & spk["test"])
    pools = load_pools()
    pools = pools[pools.corpus != "demand"]            # DEMAND is split by time inside one recording
    key = pools.group + "|" + pools.corpus
    g = {s: set(key[pools.split == s]) for s in ("train", "val", "test")}
    assert not (g["train"] & g["test"]) and not (g["train"] & g["val"])


@needs_sources
def test_gunfire_rate_and_scenarios():
    from ancdata.battlefield import build_battlefield_chain
    ch = build_battlefield_chain(CFG, "train")
    plans = [ch.plan(k) for k in range(300)]
    rate = np.mean([any(not b["distant"] for b in q["bursts"]) for q in plans])
    assert 0.35 < rate < 0.65
    assert len({q["scenario"] for q in plans}) >= 10
    assert all(0 <= q["index"] % ch.n_snippets < ch.n_snippets for q in plans)
