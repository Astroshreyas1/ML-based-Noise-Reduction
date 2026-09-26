"""ANC-Net v3 — Track A of docs/MODEL_V3_IDEATION.md, trained on dataset v3.

    boom+ref (B, 2, N) --asym STFT 256/128/hop 64--> complex spectra (B, 2, 2, 129, T)
        input maps: boom re/im, ref re/im, [log|ref|/|boom|, log|boom|]   (no IPD: v3's ref has no phase)
        -> complex conv encoder (4 stages, causal, stride 2 in freq)
        -> 1x1 bottleneck (+ 12-scalar onset stream at 1 ms sub-hops) -> 2 x GRU(128), explicit state
        -> heads on the GRU state:  event logits (none / gunfire / blast) per frame
                                    gain g(t) in [0.1, 1] per frame (the parameterised suppressor,
                                    instant attack by construction, release learned)
        -> complex conv decoder with skips -> tanh-bounded complex ratio mask on the boom, 3-tap smoothed
        -> iSTFT(mask * boom) * upsample(g)  -> enhanced (B, N)

Algorithmic latency 8 ms (128-sample synthesis) + 4 ms hop. Everything is causal:
conv kernels look one frame back, the GRU carries state, the onset stream uses only
the current hop. Differences from model/anc_net.py: no IPD map, log|boom| map,
onset stream, 3-class event head, gain head, ref-channel augmentation hooks.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anc_net import HOP, N_BINS, N_FFT, SR, AsymSTFT, CBlock  # noqa: E402

SUB = 4                         # onset sub-hops per hop (16 samples = 1 ms)
N_ONSET = 3 * SUB               # log-energy, flux, crest per sub-hop


def onset_features(boom: torch.Tensor, n_frames: int) -> torch.Tensor:
    """boom (B, N) -> (B, T, 12): per 1 ms sub-hop log-energy, energy flux vs the
    previous sub-hop, log crest factor. The causal STFT (left pad N_FFT-HOP) makes
    frame t end at sample (t+1)*HOP, so frame t's newest hop is samples
    [t*HOP, (t+1)*HOP) -- that hop is what the onset stream describes."""
    b, n = boom.shape
    need = n_frames * HOP
    x = boom[:, :need] if n >= need else F.pad(boom, (0, need - n))
    x = x.reshape(b, n_frames, SUB, HOP // SUB)
    e = (x ** 2).mean(-1)                                       # (B, T, SUB)
    loge = torch.log(e + 1e-7)
    flat = loge.reshape(b, -1)
    prev = torch.cat([torch.full_like(flat[:, :1], math.log(1e-7)), flat[:, :-1]], 1)   # stream start = silence before t=0
    flux = (flat - prev).reshape(b, n_frames, SUB)
    crest = x.abs().amax(-1) / (e.sqrt() + 1e-4)
    return torch.cat([loge, flux, torch.log1p(crest)], -1)      # (B, T, 12)


class InputMaps(nn.Module):
    """boom re/im, ref re/im, log-magnitude ratio, log|boom| -> 3 complex channels."""

    def forward(self, spec: torch.Tensor):
        boom, ref = spec[:, 0], spec[:, 1]                       # (B, 2, F, T)
        mag_b = torch.sqrt(boom[:, 0] ** 2 + boom[:, 1] ** 2 + 1e-8)
        mag_r = torch.sqrt(ref[:, 0] ** 2 + ref[:, 1] ** 2 + 1e-8)
        ratio = torch.log(mag_r / mag_b).clamp(-6, 6)
        logb = torch.log(mag_b + 1e-5)
        xr = torch.stack([boom[:, 0], ref[:, 0], ratio], 1)
        xi = torch.stack([boom[:, 1], ref[:, 1], logb], 1)
        return xr, xi


class ANCNetV3(nn.Module):
    def __init__(self, widths=(16, 32, 48, 64), gru_hidden=128, gru_layers=2, n_event_classes=3, gain_floor=0.1,
                 frontend_pr: bool = True):
        super().__init__()
        self.frontend = AsymSTFT(pr=frontend_pr)
        self.maps = InputMaps()
        self.widths = widths
        self.gain_floor = gain_floor
        chans = [3, *widths]
        self.enc = nn.ModuleList(CBlock(chans[i], chans[i + 1]) for i in range(len(widths)))
        self.dec = nn.ModuleList(CBlock(chans[i + 1] * 2, chans[i], transpose=True) for i in reversed(range(len(widths))))
        f = N_BINS
        for _ in widths:
            f = (f - 1) // 2 + 1
        self.f_bottleneck = f
        d_in = widths[-1] * 2 * f
        self.bottleneck_in = nn.Linear(d_in, gru_hidden)
        self.onset_in = nn.Linear(N_ONSET, 16)
        self.gru = nn.GRU(gru_hidden + 16, gru_hidden, num_layers=gru_layers, batch_first=True)
        self.bottleneck_out = nn.Linear(gru_hidden, d_in)
        self.event_head = nn.Sequential(nn.Linear(gru_hidden, 64), nn.PReLU(), nn.Linear(64, n_event_classes))
        self.gain_head = nn.Sequential(nn.Linear(gru_hidden, 32), nn.PReLU(), nn.Linear(32, 1))
        self.mask_out = nn.Conv2d(2 * chans[0], 2, 1)
        sm = torch.tensor([0.15, 0.70, 0.15]).view(1, 1, 3, 1).repeat(2, 1, 1, 1)
        self.register_buffer("smooth", sm)
        self.gru_hidden, self.gru_layers = gru_hidden, gru_layers

    def init_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch, self.gru_hidden, device=device)

    def forward(self, wav: torch.Tensor, h0: torch.Tensor | None = None):
        """wav (B, 2, N) [boom, ref] -> enhanced (B, N), event_logits (B, T, 3), gain (B, T), h_n."""
        b, _, n = wav.shape
        if h0 is None:
            h0 = self.init_state(b, wav.device)
        spec = self.frontend.stft(wav)                            # (B, 2, 2, F, T)
        t_frames = spec.shape[-1]
        xr, xi = self.maps(spec)
        skips = []
        for blk in self.enc:
            xr, xi = blk(xr, xi)
            skips.append((xr, xi))
        _, c, f, t = xr.shape
        z = torch.cat([xr, xi], 1).permute(0, 3, 1, 2).reshape(b, t, 2 * c * f)
        z = self.bottleneck_in(z)
        on = self.onset_in(onset_features(wav[:, 0], t))
        z, h_n = self.gru(torch.cat([z, on], -1), h0)
        event_logits = self.event_head(z)                         # (B, T, 3)
        gain = self.gain_floor + (1.0 - self.gain_floor) * torch.sigmoid(self.gain_head(z)).squeeze(-1)   # (B, T)
        z = self.bottleneck_out(z).reshape(b, t, 2 * c, f).permute(0, 2, 3, 1)
        xr, xi = z[:, :c], z[:, c:]
        n_dec = len(self.dec)
        for i, blk in enumerate(self.dec):
            sr, si = skips[n_dec - 1 - i]
            xr, xi = blk(torch.cat([xr, sr], 1), torch.cat([xi, si], 1))
            if i < n_dec - 1:
                nxt = skips[n_dec - 2 - i][0].shape[2]
                xr, xi = xr[:, :, :nxt], xi[:, :, :nxt]
        xr, xi = xr[:, :, :N_BINS], xi[:, :, :N_BINS]
        m = torch.tanh(self.mask_out(torch.cat([xr, xi], 1)))
        m = F.conv2d(F.pad(m, (0, 0, 1, 1)), self.smooth, groups=2)
        boom = spec[:, 0]
        yr = m[:, 0] * boom[:, 0] - m[:, 1] * boom[:, 1]
        yi = m[:, 0] * boom[:, 1] + m[:, 1] * boom[:, 0]
        y = self.frontend.istft(torch.stack([yr, yi], 1), n)      # (B, N)
        g = causal_ramp(gain, n)
        return y * g, event_logits, gain, h_n


def causal_ramp(gain: torch.Tensor, n: int) -> torch.Tensor:
    """Per-frame gain (B, T) -> per-sample gain (B, n): inside hop t the gain ramps
    linearly from g[t-1] to g[t]. Causal (hop t's samples are emitted after frame t
    is known) and free of the zipper a per-hop step would leave."""
    b, t = gain.shape
    prev = torch.cat([torch.ones_like(gain[:, :1]), gain[:, :-1]], 1)     # stream start = unity gain before t=0
    ramp = torch.linspace(0.0, 1.0, HOP + 1, device=gain.device, dtype=gain.dtype)[1:]   # (HOP,)
    g = prev[:, :, None] + (gain - prev)[:, :, None] * ramp                                  # (B, T, HOP)
    g = g.reshape(b, t * HOP)
    if g.shape[1] < n:
        g = F.pad(g, (0, n - g.shape[1]), value=float(gain[:, -1].mean()))
    return g[:, :n]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    net = ANCNetV3()
    x = torch.randn(2, 2, 16000) * 0.1
    y, ev, g, h = net(x)
    print(count_params(net), y.shape, ev.shape, g.shape, h.shape)
