"""ANC-Net: the council-settled enhancer (docs/MODEL_COUNCIL.md) as a PyTorch module.

Waveform in, waveform out, fully streamable:

    boom+ref (B, 2, N)  --asym STFT 256/128/hop 64-->  complex spectra (B, 4, 129, T)
        + engineered maps (inter-channel log-magnitude ratio, IPD < 2 kHz)     -> (B, 6, 129, T)
        -> complex conv encoder (4 stages, causal in time, stride 2 in freq)
        -> 1x1 bottleneck -> 2 x GRU(128) with explicit state
        -> complex conv decoder with skips -> complex ratio mask on the BOOM channel
                                            -> 3-tap psychoacoustic smoothing
        -> mask * boom spectrum --iSTFT (128-sample synthesis window)--> enhanced (B, N)
    event head on the bottleneck: per-frame transient probability (B, T)

STFT / iSTFT are plain Conv1d / ConvTranspose1d with fixed DFT kernels so the whole
graph exports to ONNX without the STFT op and Netron shows the front end.
Algorithmic latency: 128-sample synthesis window (8 ms) + 64-sample hop (4 ms).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
N_FFT = 256          # analysis window
N_SYN = 128          # synthesis window
HOP = 64
N_BINS = N_FFT // 2 + 1   # 129
IPD_MAX_HZ = 2000.0


# ----------------------------------------------------------------------------- front end
def _dft_kernels(n_fft: int, win: torch.Tensor) -> torch.Tensor:
    """Real DFT as a conv kernel: (2*N_BINS, 1, n_fft) -> [re..., im...]."""
    n = torch.arange(n_fft, dtype=torch.float32)
    k = torch.arange(N_BINS, dtype=torch.float32)[:, None]
    ang = 2 * math.pi * k * n[None, :] / n_fft
    re = torch.cos(ang) * win
    im = -torch.sin(ang) * win
    return torch.cat([re, im], 0)[:, None, :]


class AsymSTFT(nn.Module):
    """Analysis: 256-sample sqrt-Hann window, hop 64. Synthesis: 128-sample window
    placed at the END of the analysis frame, so the output frame is complete 8 ms
    after its last input sample.

    pr=True (default since 2026-09-18): the synthesis window is built for exact
    perfect reconstruction -- for every output sample the two overlapping
    ana*syn products sum to 1 -- and the inverse-DFT kernel carries the correct
    sign on the imaginary part, so istft(stft(x)) == x to 1e-6. The original
    version (pr=False, kept so the v3a checkpoint still loads) normalised the
    OLA sum by its *mean* and had the imaginary sign flipped: istft(stft(x))
    reconstructed x at rms ratio 0.81 with errors of the order of the signal,
    which the trained mask had to compensate (docs/MODEL_V3_TRAINING.md, probes)."""

    def __init__(self, pr: bool = True):
        super().__init__()
        ana = torch.hann_window(N_FFT, periodic=True).sqrt()
        n0 = N_FFT - N_SYN
        syn = torch.zeros(N_FFT)
        if pr:
            h = torch.hann_window(N_SYN, periodic=True)          # prototype, then per-sample OLA normalisation
            for m in range(HOP):
                den = ana[n0 + m] * h[m] + ana[n0 + m + HOP] * h[m + HOP]
                syn[n0 + m] = h[m] / den
                syn[n0 + m + HOP] = h[m + HOP] / den
            im_sign = 1.0
        else:
            syn_short = torch.hann_window(N_SYN, periodic=True).sqrt()
            syn[n0:] = syn_short
            ola = torch.zeros(N_FFT + HOP * 8)
            prod = ana * syn
            for i in range(0, HOP * 8, HOP):
                ola[i:i + N_FFT] += prod
            norm = ola[N_FFT - N_SYN + HOP * 2: N_FFT + HOP * 2].mean()
            syn = syn / norm
            im_sign = -1.0
        self.pr = pr
        self.register_buffer("k_ana", _dft_kernels(N_FFT, ana))
        k_syn = _dft_kernels(N_FFT, syn)             # inverse uses the same basis (real signal)
        scale = torch.full((N_BINS,), 2.0 / N_FFT); scale[0] = 1.0 / N_FFT; scale[-1] = 1.0 / N_FFT
        k_syn = k_syn * torch.cat([scale, im_sign * scale])[:, None, None]
        self.register_buffer("k_syn", k_syn)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, C, N) -> (B, C, 2, F, T) re/im. Causal: left-pad N_FFT-HOP."""
        b, c, n = x.shape
        x = F.pad(x.reshape(b * c, 1, n), (N_FFT - HOP, 0))
        spec = F.conv1d(x, self.k_ana, stride=HOP)               # (B*C, 2F, T)
        return spec.reshape(b, c, 2, N_BINS, -1)

    def istft(self, spec: torch.Tensor, n_out: int) -> torch.Tensor:
        """spec (B, 2, F, T) -> (B, N)."""
        b = spec.shape[0]
        y = F.conv_transpose1d(spec.reshape(b, 2 * N_BINS, -1), self.k_syn, stride=HOP)
        return y[:, 0, N_FFT - HOP: N_FFT - HOP + n_out]


class InterChannelFeatures(nn.Module):
    """Log magnitude ratio (full band) and IPD (below 2 kHz only, per the council)."""

    def __init__(self):
        super().__init__()
        cutoff = int(IPD_MAX_HZ / (SR / 2) * (N_BINS - 1))
        m = torch.zeros(1, 1, N_BINS, 1); m[..., :cutoff + 1, :] = 1.0
        self.register_buffer("ipd_mask", m)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec (B, 2ch, 2, F, T) -> (B, 6, F, T): boom re/im, ref re/im, log ratio, ipd."""
        boom, ref = spec[:, 0], spec[:, 1]                       # (B, 2, F, T)
        mag_b = torch.sqrt(boom[:, 0] ** 2 + boom[:, 1] ** 2 + 1e-8)
        mag_r = torch.sqrt(ref[:, 0] ** 2 + ref[:, 1] ** 2 + 1e-8)
        ratio = torch.log(mag_r / mag_b)[:, None]                # (B, 1, F, T)
        ipd = torch.atan2(ref[:, 1], ref[:, 0]) - torch.atan2(boom[:, 1], boom[:, 0])
        ipd = torch.cos(ipd)[:, None] * self.ipd_mask            # cos(IPD) avoids wrap; zero above 2 kHz
        return torch.cat([boom, ref, ratio, ipd], 1)


# ----------------------------------------------------------------------------- complex conv blocks
class ComplexConv2d(nn.Module):
    """(a+jb)*(x+jy) via two real convs; causal in time (kernel 2, left pad 1), stride 2 in freq."""

    def __init__(self, cin: int, cout: int, k=(5, 2), stride=(2, 1)):
        super().__init__()
        self.re = nn.Conv2d(cin, cout, k, stride, padding=(k[0] // 2, 0))
        self.im = nn.Conv2d(cin, cout, k, stride, padding=(k[0] // 2, 0))
        self.tpad = k[1] - 1

    def forward(self, xr, xi):
        xr = F.pad(xr, (self.tpad, 0)); xi = F.pad(xi, (self.tpad, 0))
        return self.re(xr) - self.im(xi), self.re(xi) + self.im(xr)


class ComplexConvT2d(nn.Module):
    def __init__(self, cin: int, cout: int, k=(5, 2), stride=(2, 1)):
        super().__init__()
        self.re = nn.ConvTranspose2d(cin, cout, k, stride, padding=(k[0] // 2, 0), output_padding=(1, 0))
        self.im = nn.ConvTranspose2d(cin, cout, k, stride, padding=(k[0] // 2, 0), output_padding=(1, 0))
        self.tcut = k[1] - 1

    def forward(self, xr, xi):
        yr = self.re(xr) - self.im(xi); yi = self.re(xi) + self.im(xr)
        return yr[..., :-self.tcut] if self.tcut else yr, yi[..., :-self.tcut] if self.tcut else yi


class CBlock(nn.Module):
    def __init__(self, cin, cout, transpose=False):
        super().__init__()
        self.conv = (ComplexConvT2d if transpose else ComplexConv2d)(cin, cout)
        self.bn_r, self.bn_i = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout)

    def forward(self, xr, xi):
        yr, yi = self.conv(xr, xi)
        return self.act(self.bn_r(yr)), self.act(self.bn_i(yi))


# ----------------------------------------------------------------------------- the network
class ANCNet(nn.Module):
    def __init__(self, widths=(16, 32, 48, 64), gru_hidden=128, gru_layers=2):
        super().__init__()
        self.frontend = AsymSTFT()
        self.feats = InterChannelFeatures()
        self.widths = widths
        # encoder input: 3 "complex" channels -> (boom, ref, engineered[ratio, ipd] as re/im pair)
        chans = [3, *widths]
        self.enc = nn.ModuleList(CBlock(chans[i], chans[i + 1]) for i in range(len(widths)))
        self.dec = nn.ModuleList(CBlock(chans[i + 1] * 2, chans[i], transpose=True) for i in reversed(range(len(widths))))
        f_bottleneck = N_BINS
        for _ in widths:
            f_bottleneck = (f_bottleneck - 1) // 2 + 1                 # conv k=5, pad=2, stride=2
        self.f_bottleneck = f_bottleneck
        d_in = widths[-1] * 2 * f_bottleneck
        self.bottleneck_in = nn.Linear(d_in, gru_hidden)        # 1x1 compression before the GRU
        self.gru = nn.GRU(gru_hidden, gru_hidden, num_layers=gru_layers, batch_first=True)
        self.bottleneck_out = nn.Linear(gru_hidden, d_in)
        self.event_head = nn.Sequential(nn.Linear(gru_hidden, 64), nn.PReLU(), nn.Linear(64, 1))
        self.mask_out = nn.Conv2d(2 * chans[0], 2, 1)            # complex mask (re, im) on the boom
        sm = torch.tensor([0.15, 0.70, 0.15]).view(1, 1, 3, 1).repeat(2, 1, 1, 1)
        self.register_buffer("smooth", sm)                       # 3-tap psychoacoustic smoothing over frequency

    def forward(self, wav: torch.Tensor, h0: torch.Tensor):
        """wav (B, 2, N) [boom, ref]; h0 (layers, B, hidden). Returns enhanced (B, N), p_event (B, T), h_n."""
        n = wav.shape[-1]
        spec = self.frontend.stft(wav)                           # (B, 2, 2, F, T)
        x = self.feats(spec)                                     # (B, 6, F, T)
        xr = torch.stack([x[:, 0], x[:, 2], x[:, 4]], 1)         # boom re, ref re, ratio
        xi = torch.stack([x[:, 1], x[:, 3], x[:, 5]], 1)         # boom im, ref im, ipd
        skips = []
        for blk in self.enc:
            xr, xi = blk(xr, xi)
            skips.append((xr, xi))
        b, c, f, t = xr.shape
        z = torch.cat([xr, xi], 1).permute(0, 3, 1, 2).reshape(b, t, 2 * c * f)
        z = self.bottleneck_in(z)
        z, h_n = self.gru(z, h0)
        p_event = torch.sigmoid(self.event_head(z)).squeeze(-1)  # (B, T)
        z = self.bottleneck_out(z).reshape(b, t, 2 * c, f).permute(0, 2, 3, 1)
        xr, xi = z[:, :c], z[:, c:]
        for blk, (sr, si) in zip(self.dec, reversed(skips)):
            xr, xi = blk(torch.cat([xr, sr], 1), torch.cat([xi, si], 1))
            if len(skips) and blk is not self.dec[-1]:
                nxt = skips[len(skips) - 2 - list(self.dec).index(blk)]
                xr, xi = xr[:, :, :nxt[0].shape[2]], xi[:, :, :nxt[0].shape[2]]   # match skip freq size
        xr = xr[:, :, :N_BINS]; xi = xi[:, :, :N_BINS]
        m = self.mask_out(torch.cat([xr, xi], 1))
        m = torch.tanh(m)                                        # bounded complex mask
        m = F.conv2d(F.pad(m, (0, 0, 1, 1)), self.smooth, groups=2)
        boom = spec[:, 0]                                        # (B, 2, F, T)
        yr = m[:, 0] * boom[:, 0] - m[:, 1] * boom[:, 1]
        yi = m[:, 0] * boom[:, 1] + m[:, 1] * boom[:, 0]
        enhanced = self.frontend.istft(torch.stack([yr, yi], 1), n)
        return enhanced, p_event, h_n


def export_onnx(path: str = "model/anc_net.onnx", seconds: float = 1.0) -> str:
    torch.manual_seed(0)
    net = ANCNet().eval()
    n = int(SR * seconds)
    wav = torch.randn(1, 2, n) * 0.1
    h0 = torch.zeros(net.gru.num_layers, 1, net.gru.hidden_size)
    with torch.no_grad():
        y, p, h = net(wav, h0)
    n_params = sum(p.numel() for p in net.parameters())
    torch.onnx.export(
        net, (wav, h0), path, opset_version=17, dynamo=False,
        input_names=["boom_ref_wav", "gru_state_in"],
        output_names=["enhanced_wav", "p_event", "gru_state_out"],
        dynamic_axes={"boom_ref_wav": {2: "samples"}, "enhanced_wav": {1: "samples"}, "p_event": {1: "frames"}},
    )
    print(f"params {n_params:,}  out {tuple(y.shape)} events {tuple(p.shape)} -> {path}")
    return path


if __name__ == "__main__":
    export_onnx()
