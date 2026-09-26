import numpy as np

from ancdata.adc import radio_codec, simulate_adc
from ancdata.config import SR


def test_adc_quantises_and_clips():
    x = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32) * 0.1
    y = simulate_adc(x, headroom_db=-6.0)
    assert np.abs(y).max() == 1.0
    assert np.unique(y).size < 70000          # 16-bit grid


def test_codec_bandlimits_and_keeps_length():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR * 2).astype(np.float32) * 0.2
    y = radio_codec(x, rng, dropout_p=0.0)
    assert y.shape == x.shape
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    assert spec[f > 5000].mean() < 0.05 * spec[(f > 500) & (f < 3000)].mean()
