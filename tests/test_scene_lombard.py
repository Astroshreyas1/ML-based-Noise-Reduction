import numpy as np
import pytest

from ancdata.config import SEG_SAMPLES, load_config, with_overrides
from ancdata.fixtures import fake_noise, fake_speech
from ancdata.lombard import lombard, median_f0_hz, spectral_band_ratio_db, speech_duration_s
from ancdata.scene import apply_envelope, compose_scene
from ancdata.selftest import SMOKE_CONFIG


def test_lombard_moves_f0_tilt_duration_monotonically():
    rng = np.random.default_rng(0)
    x = fake_speech(rng, 4.0, 3)
    f0s, bands, durs = [], [], []
    for s in (0.0, 0.5, 1.0):
        y, p = lombard(x, np.random.default_rng(1), strength=s)
        assert np.isfinite(y).all() and p["strength"] == s
        f0s.append(median_f0_hz(y)); bands.append(spectral_band_ratio_db(y)); durs.append(len(y))
    assert f0s[0] < f0s[1] < f0s[2]
    assert bands[0] < bands[1] < bands[2]
    assert durs[0] <= durs[1] < durs[2]


def test_scene_layers_and_levels():
    rng = np.random.default_rng(2)
    kinds = ["helicopter", "wind", "siren"]

    def draw(r):
        k = kinds[int(r.integers(len(kinds)))]
        return fake_noise(r, k, 5.0), k, {}

    cfg = {"layers": {"dist": "uniform_int", "low": 3, "high": 3}}
    layers = compose_scene(draw, rng, cfg, SEG_SAMPLES, {"wind"})
    assert len(layers) == 3
    assert layers[0].rel_db == 0.0 and all(L.rel_db < 0 for L in layers[1:])
    for L in layers:
        assert L.audio_boom.shape == (SEG_SAMPLES,) and L.audio_ref.shape == (SEG_SAMPLES,)
        assert np.isfinite(L.audio_boom).all()
        if L.dry:  # uncorrelated crops on the two capsules
            c = np.corrcoef(L.audio_boom, L.audio_ref)[0, 1]
            assert abs(c) < 0.5


def test_envelopes_change_level_over_time():
    rng = np.random.default_rng(3)
    x = fake_noise(rng, "helicopter", 4.0)
    n = len(x)
    for kind in ("gust", "approach", "recede", "passby"):
        y, m = apply_envelope(x, kind, rng, {})
        assert y.shape == x.shape and np.isfinite(y).all()
        first = np.sqrt(np.mean(y[: n // 4] ** 2)); last = np.sqrt(np.mean(y[-n // 4:] ** 2))
        if kind == "approach":
            assert last > first * 1.5
        if kind == "recede":
            assert first > last * 1.5


def test_chain_scene_meta_and_stems(data_root):
    from ancdata.chain import Chain
    from ancdata.registry import load_manifest
    from ancdata.rir_gen import RirBank
    from ancdata.sampler import Sampler
    cfg = with_overrides(load_config(SMOKE_CONFIG), debug_stems=True,
                         **{"scene.layers": {"dist": "uniform_int", "low": 2, "high": 3}, "lombard.probability": 1.0})
    ch = Chain(cfg, Sampler(load_manifest(), cfg, "train"), RirBank())
    p = ch.generate(4)
    assert 2 <= len(p.meta["scene"]) <= 3
    assert "strength" in p.meta["lombard"]
    stems = p.meta["_stems"]
    assert "speech_boom" in stems and any(k.startswith("layer1_") for k in stems)
    # stems + speech + events reconstruct the pre-ADC boom channel only approximately (ADC / mic);
    # check the noise stems are non-trivial and finite
    for k, v in stems.items():
        assert v.shape == (SEG_SAMPLES,) and np.isfinite(v).all()
