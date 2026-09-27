"""HubNet: strict real-time (< 20 ms) single-channel enhancer for radio chatter at the hub.

Deployment (user, 2026-09-27): the model runs at a hub that receives tactical radio traffic
(CVSD / FM over VHF, battlefield noise at the talker's mic), cleans it and passes it to the
officer in charge. Constraint chosen by the user: strict real time, < 20 ms algorithmic latency.

Front end   asymmetric STFT: 512-sample (32 ms) sqrt-Hann analysis window over the PAST, 192-
            sample (12 ms) synthesis window at its end, hop 96 (6 ms), perfect reconstruction.
            Algorithmic latency = synthesis window + hop = 18 ms. Only bins 0-4 kHz (129 of 257)
            are processed: the radio passes 300-3400 Hz, the target is band-limited, the rest is 0.
Input       compressed complex spectrum |X|^0.3 e^{j arg X} (re, im) + log|X|.
Encoder     conv (1x5) -> causal conv (2x3, freq stride 2): 129 -> 65 bins, C channels.
Core        N dual-path blocks: intra-frame bi-GRU over frequency (a frame sees its own bins
            only -- causal) + inter-frame uni-GRU over time per bin, residual + LayerNorm.
Decoder     transposed conv back to 129 bins, skip from the encoder.
Output      deep filter (Schroter et al. 2022, DeepFilterNet): per bin a complex K-tap filter
            over the current and K-1 past frames, Y(t,f) = sum_k H_k(t,f) X(t-k,f). Unbounded,
            so it can re-amplify what the radio compressor / codec attenuated and use the
            codec noise's short-time structure, which a 1-frame mask cannot.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

N_FFT = 512
N_SYN = 192
HOP = 96
N_BINS = N_FFT // 2 + 1
F_USE = 129                       # 0 .. 4 kHz at 16 kHz / 512
LATENCY_MS = (N_SYN + HOP) / 16.0


def _dft_kernels(n_fft: int, window: torch.Tensor) -> torch.Tensor:
    n = torch.arange(n_fft, dtype=torch.float64)
    k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
    ang = 2 * torch.pi * k * n / n_fft
    w = window.double()
    return torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w]).float()[:, None, :]


class AsymSTFT(nn.Module):
    def __init__(self):
        super().__init__()
        ana = torch.hann_window(N_FFT, periodic=True).sqrt()
        n0 = N_FFT - N_SYN
        syn = torch.zeros(N_FFT)
        h = torch.hann_window(N_SYN, periodic=True)
        for m in range(HOP):
            den = ana[n0 + m] * h[m] + ana[n0 + m + HOP] * h[m + HOP]
            syn[n0 + m] = h[m] / den
            syn[n0 + m + HOP] = h[m + HOP] / den
        self.register_buffer("k_ana", _dft_kernels(N_FFT, ana))
        k_syn = _dft_kernels(N_FFT, syn)
        scale = torch.full((N_BINS,), 2.0 / N_FFT)
        scale[0] = scale[-1] = 1.0 / N_FFT
        k_syn = k_syn * torch.cat([scale, scale])[:, None, None]   # analysis carries -sin; synthesis the same basis
        self.register_buffer("k_syn", k_syn)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, N) -> (B, 2, F, T), T = ceil(N / HOP) (right edge zero-padded to a whole hop)."""
        extra = (-x.shape[-1]) % HOP
        spec = F.conv1d(F.pad(x[:, None], (N_FFT - HOP, extra)), self.k_ana, stride=HOP)
        return spec.reshape(x.shape[0], 2, N_BINS, -1)

    def istft(self, spec: torch.Tensor, n_out: int) -> torch.Tensor:
        y = F.conv_transpose1d(spec.reshape(spec.shape[0], 2 * N_BINS, -1), self.k_syn, stride=HOP)
        return y[:, 0, N_FFT - HOP: N_FFT - HOP + n_out]


class CausalConv2d(nn.Module):
    """(time, freq) kernel; pads the past in time only."""

    def __init__(self, cin: int, cout: int, k: tuple[int, int], stride_f: int = 1):
        super().__init__()
        self.kt = k[0]
        self.conv = nn.Conv2d(cin, cout, k, stride=(1, stride_f), padding=(0, k[1] // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x (B, C, T, F)
        return self.conv(F.pad(x, (0, 0, self.kt - 1, 0)))


class DPBlock(nn.Module):
    def __init__(self, c: int, h_intra: int, h_inter: int):
        super().__init__()
        self.intra = nn.GRU(c, h_intra, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(2 * h_intra, c)
        self.intra_ln = nn.LayerNorm(c)
        self.inter = nn.GRU(c, h_inter, batch_first=True)
        self.inter_fc = nn.Linear(h_inter, c)
        self.inter_ln = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, t, f = x.shape
        z = x.permute(0, 2, 3, 1).reshape(b * t, f, c)            # over frequency, one frame at a time
        z = self.intra_ln(z + self.intra_fc(self.intra(z)[0]))
        z = z.reshape(b, t, f, c).permute(0, 2, 1, 3).reshape(b * f, t, c)   # over time, per bin
        o, h_n = self.inter(z, h)
        z = self.inter_ln(z + self.inter_fc(o))
        return z.reshape(b, f, t, c).permute(0, 3, 2, 1), h_n


class HubNet(nn.Module):
    def __init__(self, c: int = 96, n_blocks: int = 5, h_intra: int = 64, h_inter: int = 192, df_order: int = 3):
        super().__init__()
        self.stft_ = AsymSTFT()
        self.df_order = df_order
        self.checkpoint = True
        self.enc1 = nn.Sequential(nn.Conv2d(3, c // 2, (1, 5), padding=(0, 2)), nn.PReLU(c // 2))
        self.enc2 = nn.Sequential(CausalConv2d(c // 2, c, (2, 3), stride_f=2), nn.PReLU(c))
        self.blocks = nn.ModuleList([DPBlock(c, h_intra, h_inter) for _ in range(n_blocks)])
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(2 * c, c // 2, (1, 3), stride=(1, 2), padding=(0, 1)), nn.PReLU(c // 2))
        self.dec1 = CausalConv2d(c, 2 * df_order, (2, 3))

    def features(self, X: torch.Tensor) -> torch.Tensor:
        re, im = X[:, 0], X[:, 1]
        mag = torch.sqrt(re ** 2 + im ** 2 + 1e-9)
        comp = mag ** 0.3 / mag
        f = torch.stack([re * comp, im * comp, torch.log(mag + 1e-5)], 1)   # (B, 3, F, T)
        return f.transpose(2, 3)                                          # (B, 3, T, F)

    def deep_filter(self, X: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        """X (B, 2, F, T); coef (B, 2K, T, F) -> Y (B, 2, F, T)."""
        k = self.df_order
        xr, xi = X[:, 0].transpose(1, 2), X[:, 1].transpose(1, 2)        # (B, T, F)
        yr = torch.zeros_like(xr)
        yi = torch.zeros_like(xi)
        for j in range(k):
            sr = F.pad(xr, (0, 0, j, 0))[:, : xr.shape[1]]
            si = F.pad(xi, (0, 0, j, 0))[:, : xi.shape[1]]
            hr, hi = coef[:, 2 * j], coef[:, 2 * j + 1]
            yr = yr + hr * sr - hi * si
            yi = yi + hr * si + hi * sr
        return torch.stack([yr, yi], 1).transpose(2, 3)

    def forward(self, wav: torch.Tensor, h0: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """wav (B, N) -> (enhanced (B, N), GRU states)."""
        n = wav.shape[-1]
        X_full = self.stft_.stft(F.pad(wav, (0, N_SYN)))       # the last N_SYN samples need one more frame
        X = X_full[:, :, :F_USE]
        e1 = self.enc1(self.features(X))                                 # (B, C/2, T, 129)
        e2 = self.enc2(e1)                                                # (B, C, T, 65)
        z = e2
        states = []
        for i, blk in enumerate(self.blocks):
            if self.training and self.checkpoint and h0 is None:   # activation memory of the GRUs dominates
                z, h = torch.utils.checkpoint.checkpoint(blk, z, use_reentrant=False)
            else:
                z, h = blk(z, None if h0 is None else h0[i])
            states.append(h)
        d = self.dec2(torch.cat([z, e2], 1))
        d = d[..., :F_USE]
        coef = self.dec1(torch.cat([d, e1], 1))                           # (B, 2K, T, 129)
        Y = self.deep_filter(X, coef)
        Y = F.pad(Y, (0, 0, 0, N_BINS - F_USE))                            # > 4 kHz: nothing
        return self.stft_.istft(Y, n), states


if __name__ == "__main__":
    net = HubNet()
    print(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M, latency {LATENCY_MS:.1f} ms")
    x = torch.randn(2, 64000)
    y, _ = net(x)
    print(y.shape)
    # perfect reconstruction of the front end
    st = AsymSTFT()
    z = st.istft(st.stft(x), x.shape[-1])
    print("PR max err", float((z - x)[:, N_FFT:].abs().max()))
