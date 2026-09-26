import numpy as np
import pytest

from ancdata.chain import Chain, assert_contract
from ancdata.config import SEG_SAMPLES, load_config, with_overrides
from ancdata.registry import load_manifest
from ancdata.rir_gen import RirBank
from ancdata.sampler import Sampler
from ancdata.selftest import SMOKE_CONFIG


@pytest.fixture(scope="module")
def chain(data_root):
    cfg = load_config(SMOKE_CONFIG)
    return Chain(cfg, Sampler(load_manifest(), cfg, "train"), RirBank())


def test_shapes_and_contract(chain):
    p = chain.generate(3)
    assert p.noisy.shape == (2, SEG_SAMPLES) and p.clean.shape == (SEG_SAMPLES,)
    assert_contract(p)


def test_target_level_tracks_input_gain(chain):
    p = chain.generate(5)
    g = p.meta.get("adc", {}).get("gain", p.meta.get("safety_gain", 1.0))
    rms_db = 20 * np.log10(np.sqrt((p.clean ** 2).mean()) / g)
    assert -32.1 <= rms_db <= -17.9          # speech_level_db range, before the shared gain


def test_input_speech_is_sample_aligned_with_target(chain):
    """Reverb + proximity + no noise: cross-correlation peak must be at lag 0."""
    cfg = with_overrides(chain.cfg, **{"adc.probability": 0.0, "codec.probability": 0.0,
                                       "impulsive.probability": 0.0, "hard_negative.probability": 0.0,
                                       "mix_ambience.snr_db": 60.0, "reverb.probability": 1.0,
                                       "lombard.probability": 1.0})
    for k in range(3):
        p = Chain(cfg, chain.sampler, chain.bank).generate(k)
        x, c = p.noisy[0], p.clean
        lags = range(-40, 41)
        xc = [np.dot(np.roll(x, -l), c) for l in lags]
        assert lags[int(np.argmax(xc))] == 0, f"input lags target by {lags[int(np.argmax(xc))]} samples"


def test_determinism_across_instances(data_root):
    cfg = load_config(SMOKE_CONFIG)
    a = Chain(cfg, Sampler(load_manifest(), cfg, "train"), RirBank()).generate(7)
    b = Chain(cfg, Sampler(load_manifest(), cfg, "train"), RirBank()).generate(7)
    assert np.array_equal(a.noisy, b.noisy) and np.array_equal(a.clean, b.clean)


def test_events_recorded_with_local_snr(chain):
    loud = with_overrides(chain.cfg, **{"impulsive.probability": 1.0})
    c = Chain(loud, chain.sampler, chain.bank)
    p = c.generate(11)
    assert p.events
    for e in p.events:
        assert e.end > e.start and np.isfinite(e.local_snr_db)
        if e.category == "blast_synthetic":
            assert "standoff_m" in e.params


def test_adc_clips_when_overloaded(chain):
    hot = with_overrides(chain.cfg, **{"adc.probability": 1.0, "adc.headroom_db": -3.0,
                                       "codec.probability": 0.0})
    p = Chain(hot, chain.sampler, chain.bank).generate(2)
    assert np.abs(p.noisy).max() == pytest.approx(1.0, abs=1e-4)
    assert (np.abs(p.noisy) > 0.999).sum() > 10   # genuinely clipped samples


def test_no_post_mix_normalisation(chain):
    """Without ADC the mixture keeps its natural level (only a safety cap at 0.99)."""
    cfg = with_overrides(chain.cfg, **{"adc.probability": 0.0, "codec.probability": 0.0,
                                       "impulsive.probability": 0.0, "hard_negative.probability": 0.0,
                                       "mix_ambience.snr_db": 30.0, "reverb.probability": 0.0,
                                       "lombard.probability": 0.0, "mic.proximity_boost_db": 0.0})
    p = Chain(cfg, chain.sampler, chain.bank).generate(4)
    assert np.abs(p.noisy[0] - p.clean).max() < 0.05 * np.abs(p.clean).max()


def test_snr_schedule_by_index():
    from ancdata.chain import snr_spec_for_index
    mix = {"snr_db": 0.0, "schedule": [{"until": 10, "snr_db": 15.0}, {"until": None, "snr_db": -5.0}]}
    assert snr_spec_for_index(mix, 0) == 15.0
    assert snr_spec_for_index(mix, 9) == 15.0
    assert snr_spec_for_index(mix, 10) == -5.0
    assert snr_spec_for_index({"snr_db": 3.0}, 999) == 3.0
