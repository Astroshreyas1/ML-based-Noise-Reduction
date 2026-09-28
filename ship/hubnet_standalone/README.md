# HubNet standalone trainer

This trainer produces HubNet, a strict real-time speech enhancer for tactical radio chatter received at a hub. It has an algorithmic latency of 18 ms and 2.63 M parameters.

- **Input:** the received radio audio at 16 kHz.
- **Output:** clean speech limited to the radio band.

This folder is self-contained: it does not need the `ancdata` package or the laptop repo. The only other thing to copy is the built dataset.

## 1. What to copy to the workstation

| What | From the laptop | Size |
|---|---|---|
| this folder | `ANC/ship/hubnet_standalone/` | < 1 MB |
| the dataset | `ANC/data/battlefield_v31/` (train/, val/, test/, words/, DATASET.md) | ~32 GB |

## 2. Set up

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match the box's CUDA
pip install -r requirements.txt
python -m hubnet.bench          # speed / memory per batch size + projected hours for the schedule
```

## 3. Train

Run on one GPU:

```bash
python -m hubnet.train --data /data/battlefield_v31 --out runs/hub1
```

Run on several GPUs of one machine (the batch size is per GPU):

```bash
torchrun --nproc_per_node=4 -m hubnet.train --data /data/battlefield_v31 --out runs/hub1
```

To continue after an interruption, run the same command with `--resume`.

The trainer sizes itself to the GPU.

**Schedule**
- The default schedule is 1.6 M crops of 2 s, about 888 h of audio. That is the laptop's plan of 200k steps × batch 8.
- Batch per GPU is sized from memory: about 65 % of VRAM, at 1.05 GB per sample without checkpointing (0.33 GB with it), a multiple of 8, capped at 64. On a 48 GB card that is 24. Steps are then samples ÷ (batch × GPUs). The learning rate scales with √(total batch ÷ 8), capped at 2×.
- Validation runs every 4,000 steps on 192 fixed val pairs. It reports SI-SNR, STOI and PESQ-NB against their input values.
- The run writes `ckpt_best.pt`, `ckpt_last.pt` and `log.csv`.

**Speed and memory**
- Gradient checkpointing is on only when the GPU has less than 16 GB of VRAM. On bigger GPUs it stays off, which avoids recomputing the forward pass.
- `--compile` enables `torch.compile`. On Linux it is usually 10–30 % faster; if it errors, drop the flag.

Other overrides: `--batch`, `--samples`, `--steps`, `--workers`, `--val-every`, `--checkpointing on|off`.

**Why it was slow on the laptop (RTX 4050, 6 GB), from profiling on 2026-09-27**
- A step took 840 ms of pure GPU time. The data loader needs only 15 ms per batch.
- Six dual-path GRU blocks take about 92 ms each (forward plus backward). They process 2,672 frequency sequences and 520 time sequences of 334 frames per batch, and GRUs are sequential.
- Without checkpointing, training needs 8.4 GB and spills out of the 6 GB of VRAM (6.1 s per step). Checkpointing keeps it at 2.6 GB at about +30 % compute.
- A 24 GB-class GPU with checkpointing off and batch 32 should cut the schedule from about 47 h to a few hours. Use `bench` to check.

## 4. Evaluate against the deliverables

```bash
python -m hubnet.evaluate --data /data/battlefield_v31 --ckpt runs/hub1/ckpt_best.pt
```

This scores the whole test split: 1,616 pairs, with no synthetic speech. It writes `eval_test.md` next to the checkpoint, with results overall and broken down by radio link, SNR bin, scenario and speech corpus.

| Metric | Definition | Target |
|---|---|---|
| SNR | SI-SNR in dB against the radio-band clean target; the start of the transmission that push-to-talk clipped is excluded | > 15 |
| STOI | 16 kHz STOI against the same target | > 0.85 |
| PESQ | PESQ-NB (ITU-T P.862, 8 kHz), the standard for a 300–3400 Hz channel; PESQ-WB is also reported | > 2.5 |

The unprocessed input on the test split scores about SI-SNR −4 dB, STOI 0.57 and PESQ-NB 1.30.

## 5. The model

`hubnet/hub_net.py` defines the network.

- **Front end:** an asymmetric STFT. The analysis window is 512 samples (32 ms of the past), the synthesis window 192 and the hop 96, which gives an algorithmic latency of 18 ms. It reconstructs the input perfectly.
- **Band:** only bins from 0 to 4 kHz are processed (129 of 257).
- **Encoder:** a causal convolutional encoder.
- **Core:** 6 dual-path blocks. Each block has a bidirectional GRU over frequency, within a single frame, followed by a unidirectional GRU over time for each frequency bin.
- **Output:** a transposed-convolution decoder, then a complex deep filter with 3 taps (DeepFilterNet-style). The filter is unbounded, so it can undo the radio's compressor and codec loss.

Loss:
- −SI-SNR/10
- multi-resolution STFT on compressed magnitudes (L1 plus spectral convergence)
- 0.3 × compressed complex L1
- 0.5 × keyword-window SI-SNR, which up-weights radio and military vocabulary spans taken from whisper word timings in `words/`

## 6. What the dataset is

See `battlefield_v31/DATASET.md`.

- **Speech:** Lombard speech (GRID and AVID), plus ATCOSIM air-traffic radio speech, Speech Commands figure strings, and 10 % generated radio procedure (train and val only).
- **Noise:** audited battlefield noise scenes. Gunfire is present in 48 % of clips.
- **Radio link:** applied to 90 % of pairs. The links are 16 kbit/s CVSD with burst bit errors, analog NBFM with fading, clicks and squelch, and 32 kbit/s CVSD, plus push-to-talk clipping of the first syllable.
