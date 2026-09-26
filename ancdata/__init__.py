"""ancdata — scalable noisy-clean speech pair generation for defence AI-ANC.

Public surface:
    from ancdata import stream, materialize, load_config
    for pair in stream("configs/train.yaml", "train"): ...
"""
from .config import SR, SEG_SAMPLES, STFT_HOP, STFT_WIN, load_config
from .chain import Pair, Event
from .stream import stream, build_chain
from .materialize import materialize

__all__ = ["SR", "SEG_SAMPLES", "STFT_HOP", "STFT_WIN", "load_config", "Pair", "Event",
           "stream", "build_chain", "materialize"]
__version__ = "0.1.0"
