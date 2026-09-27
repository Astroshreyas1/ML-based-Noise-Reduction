"""Battlefield v4 chain: config, determinism, label budget, acoustics helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ancdata.acoustics import (concat_crossfade, draw_distance, draw_environment, expander, onset_crop, outdoor_ir,
                               radio_squelch)

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "battlefield_v4.yaml"


def test_ir_drr_and_direct_first():
    rng = np.random.default_rng(0)
    for kind in ("open_field", "forest", "urban", "cabin"):
        env = draw_environment(kind, rng)
        ir = outdoor_ir(env, 5.0, rng, ground=False)
        assert np.argmax(np.abs(ir)) == 0
        drr = 10 * np.log10(ir[0] ** 2 / np.sum(ir[1:] ** 2))
        assert abs(drr - 5.0) < 0.5, (kind, drr)


def test_distance_classes_ranges():
    rng = np.random.default_rng(1)
    for cls, (lo, hi) in (("near", (5, 30)), ("mid", (30, 150)), ("far", (150, 1500))):
        d = draw_distance(cls, rng)
        assert lo <= d["m"] <= hi


def test_event_hygiene_helpers():
    rng = np.random.default_rng(2)
    x = np.concatenate([np.zeros(8000), rng.standard_normal(4000) * np.exp(-np.arange(4000) / 800), np.zeros(4000)])
    y = onset_crop(x.astype(np.float32), 1.0)
    assert abs(y[0]) < 1e-3 and len(y) < 16000
    z = expander((x + 0.01 * rng.standard_normal(len(x))).astype(np.float32))
    assert np.std(z[:7000]) < 0.01 * 0.5                       # own floor pushed down
    c = concat_crossfade([rng.standard_normal(9000).astype(np.float32) for _ in range(3)], 20000)
    assert c.shape == (20000,) and np.isfinite(c).all()
    assert np.abs(radio_squelch(rng)).max() <= 1.0


@pytest.fixture(scope="module")
def chain():
    try:
        from ancdata.battlefield_v4 import build_battlefield_v4_chain
        return build_battlefield_v4_chain(CFG, "val")
    except (FileNotFoundError, RuntimeError, KeyError) as e:
        pytest.skip(f"sources absent: {e}")


def test_plan_deterministic_and_json(chain):
    for k in (0, 5, 123):
        a, b = chain.plan(k), chain.plan(k)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        assert json.loads(json.dumps(a)) == a


def test_budget(chain):
    bud = chain.cfg["budget"]
    for k in range(300):
        p = chain.plan(k)
        assert len(chain.plan_labels(p)) <= bud["max_labels"]
        assert 1 <= len(p["beds"]) <= 2
        assert len(p["fg"]) <= 4


def test_render_contract_and_labels(chain):
    for k in (0, 1, 2):
        pair = chain.generate(k)
        assert pair.noisy.shape == (2, chain.n)
        assert np.array_equal(pair.noisy, chain.generate(k).noisy)
        assert set(pair.meta["labels"]) <= {e.category for e in pair.events}
        assert all("coarse" in e.params for e in pair.events)
