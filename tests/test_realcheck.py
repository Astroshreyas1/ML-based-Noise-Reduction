import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve

from ancdata.adc import limiter
from ancdata.audio import write_wav
from ancdata.config import SR, load_config
from ancdata.demo_check import demo_capture_check, registered_ranges
from ancdata.fixtures import fake_noise, fake_speech
from ancdata.metrics import chunked, nonintrusive_snr_db, scorer_selftest
from ancdata.realcheck import apply_mic_ir, fit_mic_ir, make_sweep, mix_realcheck
from ancdata.selftest import SMOKE_CONFIG


def test_scorer_known_answers():
    r = scorer_selftest()
    assert r["stoi_identical"] > 0.99 and r["pesq_identical"] > 4.5


def test_chunked_enforces_block_shape():
    x = np.zeros(1000, dtype=np.float32)
    with pytest.raises(ValueError):
        chunked(lambda b: b[:-1])(x)


def test_mic_fit_recovers_known_ir():
    """Play the sweep through a known 2-tap IR; the fitted IR must match it."""
    sweep, inv = make_sweep(4.0)
    true_ir = np.zeros(64, dtype=np.float32)
    true_ir[0], true_ir[20] = 1.0, -0.4
    recorded = fftconvolve(sweep, true_ir)[: len(sweep)].astype(np.float32)
    ir = fit_mic_ir(recorded, inv)
    pk = int(np.argmax(np.abs(ir)))
    assert abs(ir[pk] - 1.0) < 0.05
    assert abs(ir[pk + 20] + 0.4) < 0.08
    y = apply_mic_ir(np.ones(100, dtype=np.float32), ir)
    assert y.shape == (100,)


def test_limiter_caps_peaks_without_nans():
    x = np.zeros(SR // 2, dtype=np.float32)
    x[1000:1100] = 5.0
    x[3000] = np.nan
    y = limiter(x)
    assert np.isfinite(y).all() and np.abs(y).max() <= 0.98 + 1e-6


def test_nonintrusive_snr_orders_correctly():
    rng = np.random.default_rng(0)
    s = fake_speech(rng, 3.0, 1)
    quiet = s + 0.001 * rng.standard_normal(len(s)).astype(np.float32)
    loud = s + 0.1 * rng.standard_normal(len(s)).astype(np.float32)
    assert nonintrusive_snr_db(quiet) > nonintrusive_snr_db(loud)


def test_realcheck_mix_and_demo_check(tmp_path: Path):
    rng = np.random.default_rng(1)
    sp, nz = tmp_path / "speech", tmp_path / "noise"
    for i in range(2):
        write_wav(sp / f"take{i}.wav", fake_speech(rng, 6.0, i))
        write_wav(nz / f"babble_{i}.wav", fake_noise(rng, "wind", 5.0))
    cfg = load_config(SMOKE_CONFIG)
    out = mix_realcheck(cfg, sp, nz, 6, tmp_path / "reality_a", verbose=False)
    metas = [json.loads(l) for l in (out / "meta.jsonl").read_text().splitlines()]
    assert len(metas) == 6 and all(m["real_sources"] for m in metas)
    ranges = registered_ranges(sorted((out / "noisy").glob("*.wav")), out / "ranges.json")
    assert ranges["n"] == 6
    # a capture that matches the set passes; a near-silent capture fails
    st = demo_capture_check(out / "ranges.json", wav=sorted((out / "noisy").glob("*.wav"))[0])
    assert "peak_dbfs" in st
    write_wav(tmp_path / "silent.wav", 1e-5 * rng.standard_normal(SR * 2).astype(np.float32))
    with pytest.raises(AssertionError):
        demo_capture_check(out / "ranges.json", wav=tmp_path / "silent.wav")
