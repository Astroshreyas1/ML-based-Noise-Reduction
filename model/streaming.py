"""Per-hop streaming form of ANC-Net v3 and the ONNX exports.

    .venv/Scripts/python.exe model/streaming.py --ckpt model/runs/v3a/ckpt_best.pt
        -> model/anc_net_v3_stream.onnx   one 64-sample hop in, one hop out, every state explicit
        -> model/anc_net_v3_chunk.onnx    any-length chunk in (dynamic axis), GRU state explicit
        + verifies streaming == offline on a real test pair and times a hop in onnxruntime (CPU)

The streaming module carries every piece of context the offline graph gets for free:
    stft_buf   (2, 192)      the previous 192 samples of boom / ref (analysis window 256, hop 64)
    enc_c{i}   (1, 2, C, F, 1)  previous (re, im) input frame of encoder conv i (time kernel 2, causal)
    dec_c{i}   (1, 2, C, F, 1)  previous (re, im) input frame of decoder conv i
    h          (2, 1, 128)   GRU state
    ola        (64,)         overlap-add tail of the last synthesis frame
    loge_prev  (1,)          last 1 ms sub-hop log-energy (onset flux)
    gain_prev  (2,)          g[t-2], g[t-1] for the causal ramp of the emitted hop

Output timing: after consuming hop t it emits hop t-1 (the 128-sample synthesis window
needs frames t-1 and t), i.e. 8 ms algorithmic latency + the 4 ms hop, as documented.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from anc_net import HOP, N_BINS, N_FFT, N_SYN  # noqa: E402
from anc_net_v3 import ANCNetV3, SUB, causal_ramp  # noqa: E402


class StreamingANCNetV3(nn.Module):
    def __init__(self, net: ANCNetV3):
        super().__init__()
        self.net = net.eval()
        self.n_enc = len(net.enc)
        # frequency sizes of the encoder outputs (skip sizes) and decoder crops
        f = N_BINS
        self.enc_f = []
        for _ in net.widths:
            f = (f - 1) // 2 + 1
            self.enc_f.append(f)

    # -- states ----------------------------------------------------------------
    def init_states(self, device=None) -> dict[str, torch.Tensor]:
        net = self.net
        st = {"stft_buf": torch.zeros(2, N_FFT - HOP, device=device),
              "h": torch.zeros(net.gru_layers, 1, net.gru_hidden, device=device),
              "ola": torch.zeros(HOP, device=device), "loge_prev": torch.full((1,), np.log(1e-7), device=device),
              "gain_prev": torch.ones(2, device=device)}
        chans_in = [3, *net.widths[:-1]]
        f_in = [N_BINS, *self.enc_f[:-1]]
        for i in range(self.n_enc):
            st[f"enc_c{i}"] = torch.zeros(1, 2, chans_in[i], f_in[i], 1, device=device)      # (re, im) previous input frame
        dec_cin = [net.widths[-1 - i] * 2 for i in range(self.n_enc)]
        dec_fin = [self.enc_f[-1 - i] for i in range(self.n_enc)]
        for i in range(self.n_enc):
            st[f"dec_c{i}"] = torch.zeros(1, 2, dec_cin[i], dec_fin[i], 1, device=device)
        return st

    def state_names(self) -> list[str]:
        return list(self.init_states().keys())

    # -- one hop -----------------------------------------------------------------
    def forward(self, hop: torch.Tensor, *states: torch.Tensor):
        """hop (1, 2, 64) -> out (1, 64), event_prob (1, 3), gain (1,), *new_states."""
        net = self.net
        names = self.state_names()
        st = dict(zip(names, states))
        x = hop[0]                                                    # (2, 64)
        frame = torch.cat([st["stft_buf"], x], 1)                     # (2, 256)
        new_buf = frame[:, HOP:]
        spec = F.conv1d(frame[:, None, :], net.frontend.k_ana)        # (2, 2F, 1)
        spec = spec.reshape(1, 2, 2, N_BINS, 1)
        xr, xi = net.maps(spec)                                       # (1, 3, F, 1)
        skips = []
        new_st = {"stft_buf": new_buf}
        for i, blk in enumerate(net.enc):
            prev = st[f"enc_c{i}"]
            cr = torch.cat([prev[:, 0], xr], -1)                      # (1, C, F, 2)
            ci = torch.cat([prev[:, 1], xi], -1)
            new_st[f"enc_c{i}"] = torch.stack([xr, xi], 1)            # (1, 2, C, F, 1)
            yr = blk.conv.re(cr) - blk.conv.im(ci)
            yi = blk.conv.re(ci) + blk.conv.im(cr)
            xr, xi = blk.act(blk.bn_r(yr)), blk.act(blk.bn_i(yi))
            skips.append((xr, xi))
        _, c, f, _ = xr.shape
        z = torch.cat([xr, xi], 1).permute(0, 3, 1, 2).reshape(1, 1, 2 * c * f)
        z = net.bottleneck_in(z)
        # onset stream for this hop
        b = x[0].reshape(SUB, HOP // SUB)
        e = (b ** 2).mean(-1)
        loge = torch.log(e + 1e-7)
        flux = loge - torch.cat([st["loge_prev"], loge[:-1]])
        crest = b.abs().amax(-1) / (e.sqrt() + 1e-4)
        on = torch.cat([loge, flux, torch.log1p(crest)])[None, None]  # (1, 1, 12)
        new_st["loge_prev"] = loge[-1:]
        z, h = net.gru(torch.cat([z, net.onset_in(on)], -1), st["h"])
        new_st["h"] = h
        ev = torch.softmax(net.event_head(z), -1)[0]                  # (1, 3)
        gain = net.gain_floor + (1.0 - net.gain_floor) * torch.sigmoid(net.gain_head(z)).reshape(1)
        z = net.bottleneck_out(z).reshape(1, 1, 2 * c, f).permute(0, 2, 3, 1)
        xr, xi = z[:, :c], z[:, c:]
        n_dec = self.n_enc
        for i, blk in enumerate(net.dec):
            sr, si = skips[n_dec - 1 - i]
            inr, ini = torch.cat([xr, sr], 1), torch.cat([xi, si], 1)
            prev = st[f"dec_c{i}"]
            new_st[f"dec_c{i}"] = torch.stack([inr, ini], 1)
            cr = torch.cat([prev[:, 0], inr], -1)                     # (1, C, F, 2)
            ci = torch.cat([prev[:, 1], ini], -1)
            yr = blk.conv.re(cr) - blk.conv.im(ci)                    # (1, C', F', 3)
            yi = blk.conv.re(ci) + blk.conv.im(cr)
            yr, yi = yr[..., 1:2], yi[..., 1:2]                       # w0*cur + w1*prev
            xr, xi = blk.act(blk.bn_r(yr)), blk.act(blk.bn_i(yi))
            if i < n_dec - 1:
                nxt = skips[n_dec - 2 - i][0].shape[2]
                xr, xi = xr[:, :, :nxt], xi[:, :, :nxt]
        xr, xi = xr[:, :, :N_BINS], xi[:, :, :N_BINS]
        m = torch.tanh(net.mask_out(torch.cat([xr, xi], 1)))
        m = F.conv2d(F.pad(m, (0, 0, 1, 1)), net.smooth, groups=2)   # (1, 2, F, 1)
        boom = spec[:, 0]                                             # (1, 2, F, 1)
        yr = m[:, 0] * boom[:, 0] - m[:, 1] * boom[:, 1]
        yi = m[:, 0] * boom[:, 1] + m[:, 1] * boom[:, 0]
        y = F.conv_transpose1d(torch.stack([yr, yi], 1).reshape(1, 2 * N_BINS, 1), net.frontend.k_syn)[0, 0]   # (256,)
        # synthesis window occupies the last 128 samples: [128,192) completes the previous hop, [192,256) is the tail
        out_hop = st["ola"] + y[N_FFT - N_SYN: N_FFT - N_SYN + HOP]
        new_st["ola"] = y[N_FFT - HOP:]
        # gain for the EMITTED hop (t-1): ramp from g[t-2] to g[t-1]
        gp = st["gain_prev"]
        ramp = torch.linspace(0.0, 1.0, HOP + 1, device=gain.device, dtype=gain.dtype)[1:]
        g_samples = gp[0] + (gp[1] - gp[0]) * ramp
        new_st["gain_prev"] = torch.cat([gp[1:], gain])
        out = (out_hop * g_samples)[None]
        return (out, ev, gain, *[new_st[n] for n in names])


def offline_reference(net: ANCNetV3, wav: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return net(wav[None])[0][0]


@torch.no_grad()
def stream_clip(sm: StreamingANCNetV3, wav: torch.Tensor) -> torch.Tensor:
    st = sm.init_states()
    names = sm.state_names()
    outs = []
    n_hops = wav.shape[-1] // HOP
    for t in range(n_hops):
        hop = wav[:, t * HOP:(t + 1) * HOP][None]
        res = sm(hop, *[st[n] for n in names])
        outs.append(res[0][0])
        st = dict(zip(names, res[3:]))
    return torch.cat(outs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "model")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    hp = ck["hp"]
    net = ANCNetV3(tuple(hp["widths"]), hp["gru_hidden"], hp["gru_layers"], frontend_pr=hp.get("frontend_pr", False)).eval()
    net.load_state_dict(ck["model"])
    sm = StreamingANCNetV3(net).eval()

    # --- verification on a real test pair
    import soundfile as sf
    x, _ = sf.read(str(ROOT / "data" / "battlefield_v3" / "test" / "noisy" / "000003.wav"), dtype="float32", always_2d=True)
    wav = torch.from_numpy(x.T[:, : 2 * 16000].copy())
    ref = offline_reference(net, wav)
    strm = stream_clip(sm, wav)
    # streaming emits hop t-1 after hop t: align by one hop
    d = (strm[HOP:] - ref[: len(ref) - HOP]).abs().max().item()
    print(f"streaming vs offline: max |diff| = {d:.2e} over {len(ref) - HOP} samples (peak {ref.abs().max():.3f})")
    assert d < 1e-3, "streaming graph does not match the offline forward"

    # --- exports
    st = sm.init_states()
    names = sm.state_names()
    hop = torch.zeros(1, 2, HOP)
    stream_path = a.out_dir / "anc_net_v3_stream.onnx"
    torch.onnx.export(sm, (hop, *[st[n] for n in names]), str(stream_path), opset_version=17, dynamo=False,
                      input_names=["hop_boom_ref", *[f"{n}_in" for n in names]],
                      output_names=["hop_out", "event_prob", "gain", *[f"{n}_out" for n in names]])
    chunk_path = a.out_dir / "anc_net_v3_chunk.onnx"
    wav1 = torch.zeros(1, 2, 16000)
    h0 = net.init_state(1)

    class Chunk(nn.Module):
        def __init__(self, n):
            super().__init__(); self.n = n

        def forward(self, w, h):
            y, ev, g, hn = self.n(w, h)
            return y, torch.softmax(ev, -1), g, hn

    torch.onnx.export(Chunk(net).eval(), (wav1, h0), str(chunk_path), opset_version=17, dynamo=False,
                      input_names=["boom_ref_wav", "gru_state_in"], output_names=["enhanced_wav", "event_prob", "gain", "gru_state_out"],
                      dynamic_axes={"boom_ref_wav": {2: "samples"}, "enhanced_wav": {1: "samples"}, "event_prob": {1: "frames"}, "gain": {1: "frames"}})
    # --- onnxruntime check + per-hop timing
    import onnxruntime as ort
    sess = ort.InferenceSession(str(stream_path), providers=["CPUExecutionProvider"])
    feed = {"hop_boom_ref": hop.numpy(), **{f"{n}_in": st[n].numpy() for n in names}}
    outs = []
    cur = {n: st[n].numpy() for n in names}
    n_hops = wav.shape[-1] // HOP
    t0 = time.perf_counter()
    for t in range(n_hops):
        feed = {"hop_boom_ref": wav[:, t * HOP:(t + 1) * HOP][None].numpy(), **{f"{n}_in": cur[n] for n in names}}
        res = sess.run(None, feed)
        outs.append(res[0][0])
        cur = {n: res[3 + i] for i, n in enumerate(names)}
    dt = (time.perf_counter() - t0) / n_hops
    ort_out = np.concatenate(outs)
    d2 = np.abs(ort_out[HOP:] - ref[: len(ref) - HOP].numpy()).max()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"onnxruntime stream vs torch offline: max |diff| = {d2:.2e}; {dt * 1000:.2f} ms per 4 ms hop on CPU (onnxruntime, 1 thread default); "
          f"params {n_params:,}; files: {stream_path.name} ({stream_path.stat().st_size / 1e6:.1f} MB), {chunk_path.name} ({chunk_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
