"""Speech-to-text backup (feature 2): two tiers, both on HubNet's cleaned output.

Tier 1  live partials   sherpa-onnx streaming Zipformer transducer (int8, CPU, ~1 core); text appears while
                        the operator is still talking (~0.2-0.4 s behind speech).
Tier 2  final           faster-whisper on the whole transmission when it ends (squelch close / VAD + 400 ms /
                        "over" / "out"); temperature 0, beam 5, radio vocabulary as hotwords, capped tokens.
                        Laptop: small.en (~0.2 s per 6 s on the RTX 4050). Workstation (48 GB): large-v3-turbo.
Segmentation            squelch/PTT line when the receiver provides it (zero cost), else energy VAD with a
                        400 ms hang, closed early when a partial ends in "over" / "out".
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
HOTWORDS = ("alfa bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november oscar papa quebec "
            "romeo sierra tango uniform victor whiskey x-ray yankee zulu niner fife tree fower roger wilco over out "
            "say again break copy affirmative negative contact grid sitrep medevac casevac nine line fire mission "
            "adjust fire fire for effect danger close check fire rally point phase line radio check")


PROMPT = ("Zulu zero, this is Bravo two, contact front, squad size enemy, grid four niner seven two six fife, over. "
          "Roger, wilco, out. Nine line medevac, line one, grid tree two one fower, two urgent, over.")


def cuda12_dlls() -> None:
    import site
    for sp in site.getsitepackages():
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "Lib/site-packages/nvidia/cublas/bin",
                    "Lib/site-packages/nvidia/cudnn/bin"):
            d = Path(sp) / sub
            if d.exists():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def collapse_loops(text: str, max_run: int = 5) -> str:
    """Whisper can loop on radio audio ("seven seven seven ..."): cut a run of > max_run identical words and
    mark the transcript, so rules never read a looped digit string as a grid."""
    words = text.split()
    out, run = [], 0
    for i, w in enumerate(words):
        run = run + 1 if i and w.strip(",.").lower() == words[i - 1].strip(",.").lower() else 1
        if run > max_run:
            return " ".join(out) + " [unreliable: repetition]"
        out.append(w)
    return text


class LiveASR:
    """Tier 1: streaming Zipformer."""

    def __init__(self, model_dir: Path = ROOT / "artefacts" / "asr" / "sherpa-onnx-streaming-zipformer-en-2023-06-21"):
        import sherpa_onnx
        d = Path(model_dir)
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=str(d / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(d / "decoder-epoch-99-avg-1.onnx"), joiner=str(d / "joiner-epoch-99-avg-1.int8.onnx"),
            num_threads=1, sample_rate=SR, feature_dim=80, decoding_method="greedy_search",
            enable_endpoint_detection=False)
        self.stream = self.rec.create_stream()

    def reset(self) -> None:
        self.stream = self.rec.create_stream()

    def feed(self, x: np.ndarray) -> str:
        self.stream.accept_waveform(SR, np.asarray(x, np.float32))
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        return self.rec.get_result(self.stream).strip().lower()


class FinalASR:
    """Tier 2: faster-whisper on the finished transmission."""

    def __init__(self, model: str = "small.en", device: str = "cuda"):
        if device == "cuda":
            cuda12_dlls()
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8")

    def __call__(self, x: np.ndarray) -> tuple[str, float]:
        text, dt, _ = self.transcribe(x)
        return text, dt

    def transcribe(self, x: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        """(text, seconds, [(word, probability)])."""
        t = time.perf_counter()
        segs, _ = self.m.transcribe(np.asarray(x, np.float32), language="en", beam_size=5, temperature=0.0,
                                    condition_on_previous_text=False, hotwords=HOTWORDS, word_timestamps=True,
                                    max_new_tokens=96, vad_filter=False, initial_prompt=PROMPT,
                                    compression_ratio_threshold=2.0, log_prob_threshold=-1.0)
        segs = list(segs)
        text = collapse_loops(" ".join(s.text.strip() for s in segs).strip())
        words = [(w.word.strip(), float(w.probability)) for s in segs for w in (s.words or [])]
        return text, time.perf_counter() - t, words


@dataclass
class Segmenter:
    """Transmission boundaries from the squelch line (if given) or energy VAD with a hang time."""
    thr_db: float = -42.0
    hang_s: float = 1.0          # radio procedure has 0.4-0.8 s pauses between phrases
    min_s: float = 0.5
    in_tx: bool = False
    silent_s: float = 0.0
    buf: list = field(default_factory=list)

    def push(self, hop: np.ndarray, squelch_open: bool | None = None, partial: str = "") -> np.ndarray | None:
        """Returns the finished transmission (audio) when one closes, else None."""
        dt = len(hop) / SR
        active = squelch_open if squelch_open is not None else \
            10 * np.log10(np.mean(np.square(hop, dtype=np.float64)) + 1e-12) > self.thr_db
        if active:
            self.in_tx = True
            self.silent_s = 0.0
        if self.in_tx:
            self.buf.append(np.asarray(hop, np.float32))
            if not active:
                self.silent_s += dt
            early = partial.endswith(" over") or partial.endswith(" out")
            hang = 0.1 if squelch_open is not None else self.hang_s      # squelch close is exact: no hang needed
            if self.silent_s >= hang or (early and self.silent_s >= 0.1):
                x = np.concatenate(self.buf)
                self.buf, self.in_tx, self.silent_s = [], False, 0.0
                return x if len(x) / SR >= self.min_s else None
        return None
