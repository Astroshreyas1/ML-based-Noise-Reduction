"""Frame-by-frame streaming HubNet: 96 samples (6 ms) in, 96 samples out, explicit state.

Identical to the offline model (checked by `selftest`); algorithmic latency 18 ms (192-sample synthesis
window + 96-sample hop). Per hop it keeps: the last 512 input samples (analysis window), the previous
encoder frame (causal 2x3 conv), the previous decoder frame (causal 2x3 conv), the last K-1 spectra (deep
filter), the inter-frame GRU states, and the overlap-add tail.

    s = StreamingHubNet.load("artefacts/models/hubnet_hub1_early.pt")
    for hop in chunks(x, 96): y_hop = s.step(hop)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .hub_net import F_USE, HOP, N_BINS, N_FFT, N_SYN, HubNet


class StreamingHubNet:
    def __init__(self, net: HubNet, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.dev = torch.device(device)
        self.reset()

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "StreamingHubNet":
        ck = torch.load(path, map_location="cpu")
        hp = ck["hp"]
        net = HubNet(hp["c"], hp["n_blocks"], hp["h_intra"], hp["h_inter"], hp["df_order"])
        net.load_state_dict(ck["model"])
        return cls(net, device)

    def reset(self) -> None:
        n = self.net
        c = n.enc2[0].conv.out_channels
        self.buf = torch.zeros(N_FFT, device=self.dev)
        self.prev_e1 = torch.zeros(1, c // 2, 1, F_USE, device=self.dev)
        self.prev_dec_in = torch.zeros(1, c, 1, F_USE, device=self.dev)
        self.x_hist = [torch.zeros(1, 2, F_USE, 1, device=self.dev) for _ in range(n.df_order - 1)]
        self.h = [None] * len(n.blocks)
        self.ola = torch.zeros(N_FFT, device=self.dev)

    @torch.no_grad()
    def step(self, hop: np.ndarray) -> np.ndarray:
        n = self.net
        x = torch.as_tensor(np.asarray(hop, np.float32), device=self.dev)
        self.buf = torch.cat([self.buf[HOP:], x])
        spec = F.conv1d(self.buf[None, None], n.stft_.k_ana).reshape(1, 2, N_BINS, 1)
        X = spec[:, :, :F_USE]                                             # (1, 2, F, 1)
        e1 = n.enc1(n.features(X))                                         # (1, C/2, 1, F)
        e2 = n.enc2[1](n.enc2[0].conv(torch.cat([self.prev_e1, e1], 2)))  # causal 2-tap in time
        self.prev_e1 = e1
        z = e2
        for i, blk in enumerate(n.blocks):
            z, self.h[i] = blk(z, self.h[i])
        d = n.dec2(torch.cat([z, e2], 1))[..., :F_USE]
        dec_in = torch.cat([d, e1], 1)
        coef = n.dec1.conv(torch.cat([self.prev_dec_in, dec_in], 2))      # (1, 2K, 1, F)
        self.prev_dec_in = dec_in
        hist = self.x_hist + [X]                                           # oldest ... current
        Y = torch.zeros_like(X)
        k = n.df_order
        for j in range(k):
            Xj = hist[-1 - j]
            hr, hi = coef[:, 2 * j].transpose(1, 2), coef[:, 2 * j + 1].transpose(1, 2)   # (1, F, 1)
            Y[:, 0] += hr * Xj[:, 0] - hi * Xj[:, 1]
            Y[:, 1] += hr * Xj[:, 1] + hi * Xj[:, 0]
        self.x_hist = hist[1:] if k > 1 else []
        Yf = F.pad(Y, (0, 0, 0, N_BINS - F_USE)).reshape(1, 2 * N_BINS, 1)
        frame = F.conv_transpose1d(Yf, n.stft_.k_syn)[0, 0]               # (N_FFT,)
        self.ola = self.ola + frame
        n0 = N_FFT - N_SYN                     # these HOP samples are final once this frame is added
        out = self.ola[n0: n0 + HOP].cpu().numpy().copy()
        self.ola = torch.cat([self.ola[HOP:], torch.zeros(HOP, device=self.dev)])
        return out


def selftest(path: str, seconds: float = 3.0) -> dict:
    """Streaming output == offline output (aligned by the model's delay)."""
    import time
    s = StreamingHubNet.load(path)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(int(seconds * 16000)) * 0.05).astype(np.float32)
    x = x[: len(x) // HOP * HOP]
    with torch.no_grad():
        y_off, _ = s.net(torch.from_numpy(x)[None])
    y_off = y_off[0].numpy()
    t = time.perf_counter()
    ys = np.concatenate([s.step(x[i: i + HOP]) for i in range(0, len(x), HOP)])
    per_hop_ms = (time.perf_counter() - t) / (len(x) / HOP) * 1000
    errs = {L: float(np.abs(ys[L:][N_FFT:4000] - y_off[: len(ys) - L][N_FFT:4000]).max()) for L in range(0, N_FFT)}
    lag = min(errs, key=errs.get)                                         # streaming output delay vs offline
    a = ys[lag:]
    b = y_off[: len(a)]
    return {"max_abs_diff": float(np.abs(a[N_FFT:] - b[N_FFT:]).max()), "lag_samples": lag,
            "stream_delay_ms": lag / 16, "algorithmic_latency_ms": (N_SYN + HOP) / 16, "cpu_ms_per_6ms_hop": per_hop_ms}


class GraphedStreamingHubNet(StreamingHubNet):
    """Same computation, captured once as a CUDA graph: one launch per 6 ms hop instead of ~200 kernels.
    All state lives in static tensors updated in place."""

    def __init__(self, net: HubNet, device: str = "cuda"):
        super().__init__(net, device)
        n = self.net
        f2 = n.enc2[0].conv(torch.zeros(1, self.prev_e1.shape[1], 2, F_USE, device=self.dev)).shape[-1]
        self.h = [torch.zeros(1, f2, blk.inter.hidden_size, device=self.dev) for blk in n.blocks]
        self.x_in = torch.zeros(HOP, device=self.dev)
        self.y_out = torch.zeros(HOP, device=self.dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self._body()
        self.reset_state()

    def reset_state(self) -> None:
        for t in [self.buf, self.prev_e1, self.prev_dec_in, self.ola, *self.x_hist, *self.h]:
            t.zero_()

    def _body(self) -> None:
        n = self.net
        buf = torch.cat([self.buf[HOP:], self.x_in])
        self.buf.copy_(buf)
        X = F.conv1d(buf[None, None], n.stft_.k_ana).reshape(1, 2, N_BINS, 1)[:, :, :F_USE]
        e1 = n.enc1(n.features(X))
        e2 = n.enc2[1](n.enc2[0].conv(torch.cat([self.prev_e1, e1], 2)))
        self.prev_e1.copy_(e1)
        z = e2
        for i, blk in enumerate(n.blocks):
            b, c, t, f = z.shape
            zz = z.permute(0, 2, 3, 1).reshape(b * t, f, c)
            zz = blk.intra_ln(zz + blk.intra_fc(blk.intra(zz)[0]))
            zz = zz.reshape(b, t, f, c).permute(0, 2, 1, 3).reshape(b * f, t, c)
            o, h_n = blk.inter(zz, self.h[i])
            self.h[i].copy_(h_n)
            zz = blk.inter_ln(zz + blk.inter_fc(o))
            z = zz.reshape(b, f, t, c).permute(0, 3, 2, 1)
        d = n.dec2(torch.cat([z, e2], 1))[..., :F_USE]
        dec_in = torch.cat([d, e1], 1)
        coef = n.dec1.conv(torch.cat([self.prev_dec_in, dec_in], 2))
        self.prev_dec_in.copy_(dec_in)
        hist = self.x_hist + [X]
        k = n.df_order
        yr = torch.zeros_like(X[:, 0])
        yi = torch.zeros_like(X[:, 1])
        for j in range(k):
            Xj = hist[-1 - j]
            hr, hi = coef[:, 2 * j].transpose(1, 2), coef[:, 2 * j + 1].transpose(1, 2)
            yr = yr + hr * Xj[:, 0] - hi * Xj[:, 1]
            yi = yi + hr * Xj[:, 1] + hi * Xj[:, 0]
        for j in range(k - 1):                                   # shift the spectrum history in place
            self.x_hist[j].copy_(hist[j + 1])
        Y = torch.stack([yr, yi], 1)
        Yf = F.pad(Y, (0, 0, 0, N_BINS - F_USE)).reshape(1, 2 * N_BINS, 1)
        frame = F.conv_transpose1d(Yf, n.stft_.k_syn)[0, 0]
        ola = self.ola + frame
        n0 = N_FFT - N_SYN
        self.y_out.copy_(ola[n0: n0 + HOP])
        self.ola.copy_(torch.cat([ola[HOP:], torch.zeros_like(ola[:HOP])]))

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "GraphedStreamingHubNet":
        ck = torch.load(path, map_location="cpu")
        hp = ck["hp"]
        net = HubNet(hp["c"], hp["n_blocks"], hp["h_intra"], hp["h_inter"], hp["df_order"])
        net.load_state_dict(ck["model"])
        return cls(net, device)

    def step(self, hop: np.ndarray) -> np.ndarray:
        self.x_in.copy_(torch.from_numpy(np.asarray(hop, np.float32)))
        self.graph.replay()
        return self.y_out.cpu().numpy()
