import numpy as np

from ancdata.audio import attack_time_ms, crest_factor_db
from ancdata.config import SR
from ancdata.physics import friedlander, synth_blast, synth_rotor


def test_friedlander_shape():
    w = friedlander(1.0, 2.0)
    assert w[0] == 1.0                       # instantaneous rise
    assert w.min() < 0                       # negative phase kept
    zc = np.where(np.diff(np.sign(w)) != 0)[0][0]
    assert abs(zc / SR * 1000 - 2.0) < 0.15  # zero crossing at T


def test_blast_attack_is_fast_and_peaky():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, meta = synth_blast(rng)
        assert np.abs(x).max() <= 1.0 + 1e-6
        assert attack_time_ms(x) < 1.5   # far shots are low-passed (boom); ground bounce can add ~1 ms
        assert crest_factor_db(x) > 8    # a 3-round burst lowers crest vs a single shot
        assert 10 <= meta["standoff_m"] <= 500


def test_blast_burst_length_grows():
    rng = np.random.default_rng(1)
    single, _ = synth_blast(rng, {"burst": 1, "supersonic_p": 0.0})
    burst, m = synth_blast(rng, {"burst": 3, "supersonic_p": 0.0})
    assert m["n_rounds"] == 3 and len(burst) > 2 * len(single)


def test_rotor_has_bpf_harmonics():
    rng = np.random.default_rng(0)
    x, meta = synth_rotor(rng, SR * 2)
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    bpf = meta["f_bpf_hz"]
    peak_bin = spec[(f > bpf * 0.9) & (f < bpf * 1.1)].max()
    assert peak_bin > np.median(spec) * 20
