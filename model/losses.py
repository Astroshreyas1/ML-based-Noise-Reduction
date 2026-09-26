"""Training losses for ANC-Net v3 on dataset v3. Read this before touching weights.

What the target is (docs/DATASET_V3.md §1): the DRY Lombard speech at its drawn
level, times the boom AGC gain trajectory. It is *not* scale-free: the headset
must reproduce the talker at the level the input carried, and it ducks after a
shot. That shapes the loss:

  1. -SI-SNR (waveform, scale-invariant)          weight  w_sisnr   default 1.0 on dB/10
     Stable at -10 dB input SNR, rewards waveform structure (phase). Being scale-
     invariant it says nothing about level -- terms 2 and 3 do.
  2. multi-resolution STFT loss on COMPRESSED magnitude (|X|^0.3), windows 64 / 256 / 1024,
     L1 + spectral convergence                     weight  w_mrstft  default 1.0
     Scale-dependent, time-resolved: anchors the absolute level and the gain trajectory.
     The 64-sample window is the only term that "sees" a 0.3 ms crack; a single 256
     window averages a gunshot away. Power-law compression instead of log so that the
     target's exact-zero pauses (dry snippet silence) are not an eps-dominated log(0).
  3. compressed complex (RI) L1 at the 256 window   weight  w_ri      default 0.3
     Phase-aware level anchor (the "RI + magnitude" recipe of DCCRN / GTCRN).
  4. log-Mel (80 bands) L1                          weight  w_mel     default 0.2
     Formant / envelope preservation, the perceptual term that suppressed the
     "robotic" timbre in Mini-DCCRN V2's ablations.
  5. event head: cross-entropy per frame, 3 classes (none / gunfire / blast), hard
     negatives are explicit class-0 frames                        weight w_event  default 0.3
  6. event-window SI-SNR: SI-SNR restricted to [-50 ms, +250 ms] around every labelled
     gunfire / blast span (the ducked speech after a shot is where masks leave holes)
                                                    weight  w_evwin   default 0.3 on dB/10
     zero when the clip has no transient.
  7. gain-head regulariser: g(t) must be 1 where the target's short-time energy equals
     the dry-level expectation and can be low only where the boom envelope jumped --
     implemented as a hinge that penalises g < 0.9 on frames with NO event label
                                                    weight  w_gain    default 0.05
     Keeps the suppressor from becoming a second mask; without it g and the mask
     trade gain arbitrarily.

Sum of weights is not 1 on purpose: each term is scaled to O(1) on the untrained
model (SI-SNR terms are divided by 10). `LossReport` returns every term so the
training log shows what moves. Weights live in train.py's HP block.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
HOP = 64
EPS = 1e-8


def si_snr_db(est: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-item SI-SNR in dB, (B,). Optional sample mask (B, N) restricts the sum (event windows)."""
    if mask is not None:
        est = est * mask
        ref = ref * mask
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    s = (est * ref).sum(-1, keepdim=True) * ref / (ref.pow(2).sum(-1, keepdim=True) + EPS)
    e = est - s
    return 10 * torch.log10(s.pow(2).sum(-1) / (e.pow(2).sum(-1) + EPS) + EPS)


class STFTLoss(nn.Module):
    def __init__(self, n_fft: int, hop: int, compress: float = 0.3):
        super().__init__()
        self.n_fft, self.hop, self.c = n_fft, hop, compress
        self.register_buffer("win", torch.hann_window(n_fft))

    def spec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(x, self.n_fft, self.hop, window=self.win, return_complex=True, center=True)

    def forward(self, est: torch.Tensor, ref: torch.Tensor):
        E, R = self.spec(est.float()), self.spec(ref.float())
        me, mr = E.abs().clamp_min(1e-7) ** self.c, R.abs().clamp_min(1e-7) ** self.c
        mag = (me - mr).abs().mean()
        sc = torch.norm(mr - me, p="fro") / (torch.norm(mr, p="fro") + EPS)
        # compressed complex: magnitude^c * phase
        ec = E / E.abs().clamp_min(1e-7) * me
        rc = R / R.abs().clamp_min(1e-7) * mr
        ri = (torch.view_as_real(ec) - torch.view_as_real(rc)).abs().mean()
        return mag, sc, ri


class MelLoss(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 128, n_mels: int = 80, sr: int = SR):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", torch.hann_window(n_fft))
        self.register_buffer("fb", mel_filterbank(sr, n_fft, n_mels))

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        def logmel(x):
            p = torch.stft(x.float(), self.n_fft, self.hop, window=self.win, return_complex=True, center=True).abs() ** 2
            return torch.log(torch.matmul(self.fb, p) + 1e-5)
        return (logmel(est) - logmel(ref)).abs().mean()


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 40.0, fmax: float | None = None) -> torch.Tensor:
    fmax = fmax or sr / 2
    def hz2mel(f): return 2595 * math.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    mels = torch.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = torch.tensor([mel2hz(m) for m in mels])
    bins = torch.floor((n_fft + 1) * hz / sr).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        if c == lo:
            c = lo + 1
        if hi == c:
            hi = c + 1
        fb[m - 1, lo:c] = torch.linspace(0, 1, int(c - lo))
        fb[m - 1, c:hi] = torch.linspace(1, 0, int(hi - c))
    return fb


@dataclass
class LossWeights:
    sisnr: float = 1.0
    mrstft: float = 1.0
    ri: float = 0.3
    mel: float = 0.2
    event: float = 0.3
    evwin: float = 0.3
    gain: float = 0.05
    event_class_weights: tuple[float, float, float] = (1.0, 2.0, 2.0)


class ANCLoss(nn.Module):
    def __init__(self, w: LossWeights | None = None):
        super().__init__()
        self.w = w or LossWeights()
        self.stfts = nn.ModuleList([STFTLoss(64, 16), STFTLoss(256, 64), STFTLoss(1024, 256)])
        self.mel = MelLoss()
        self.register_buffer("cls_w", torch.tensor(self.w.event_class_weights))

    def forward(self, est: torch.Tensor, ref: torch.Tensor, event_logits: torch.Tensor, gain: torch.Tensor,
                frame_class: torch.Tensor, ev_window: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """est, ref (B, N); event_logits (B, T, 3); gain (B, T); frame_class (B, T) long;
        ev_window (B, N) float in {0, 1}. Returns (total, per-term dict)."""
        w = self.w
        terms: dict[str, torch.Tensor] = {}
        sisnr = si_snr_db(est, ref)
        terms["sisnr_db"] = sisnr.mean()
        total = -w.sisnr * sisnr.mean() / 10.0
        mag = sc = ri = 0.0
        for i, s in enumerate(self.stfts):
            m, c, r = s(est, ref)
            mag = mag + m
            sc = sc + c
            if i == 1:
                ri = r
        mag, sc = mag / len(self.stfts), sc / len(self.stfts)
        terms["mrstft_mag"], terms["mrstft_sc"], terms["ri"] = mag, sc, ri
        total = total + w.mrstft * (mag + sc) + w.ri * ri
        mel = self.mel(est, ref)
        terms["mel"] = mel
        total = total + w.mel * mel
        t = min(event_logits.shape[1], frame_class.shape[1])
        ce = F.cross_entropy(event_logits[:, :t].reshape(-1, event_logits.shape[-1]), frame_class[:, :t].reshape(-1),
                             weight=self.cls_w)
        terms["event_ce"] = ce
        total = total + w.event * ce
        has_ev = ev_window.sum(-1) > 0
        if has_ev.any():
            evw = si_snr_db(est[has_ev], ref[has_ev], ev_window[has_ev])
            terms["evwin_sisnr_db"] = evw.mean()
            total = total - w.evwin * evw.mean() / 10.0
        else:
            terms["evwin_sisnr_db"] = torch.zeros((), device=est.device)
        no_ev = (frame_class[:, :t] == 0).float()
        hinge = (F.relu(0.9 - gain[:, :t]) * no_ev).sum() / (no_ev.sum() + 1.0)
        terms["gain_hinge"] = hinge
        total = total + w.gain * hinge
        with torch.no_grad():
            pred = event_logits[:, :t].argmax(-1)
            terms["event_acc"] = (pred == frame_class[:, :t]).float().mean()
            pos = frame_class[:, :t] > 0
            terms["event_recall"] = ((pred > 0) & pos).float().sum() / (pos.float().sum() + 1.0)
        return total, {k: float(v.detach()) for k, v in terms.items()}
