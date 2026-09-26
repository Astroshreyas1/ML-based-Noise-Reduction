# ANC-Net live demo

`pwsh demo/run_demo.ps1` → http://localhost:8765 (Chrome / Edge; the microphone works on localhost without HTTPS).

Flow: **Record 6 s** (raw mic: browser echo cancellation / noise suppression / AGC off) → the clip goes through the *same* chain that built the dataset (`ancdata/battlefield.py`, test-split noise pools the model never saw: scene, bed, texture, gunfire / blasts, capsule channel) → ANC-Net v3 → three players: microphone (clean), battlefield input (boom mic), ANC-Net output; scene description, transient list, detector strip, SI-SNR in → out.

- **Scene** dropdown picks the scenario (or random); **SNR** forces the K-weighted SNR (auto = the dataset's −8…15 dB draw).
- **use sample clip** = a held-out Lombard snippet instead of the mic: the fallback if the venue mic misbehaves.
- The server re-reads `ckpt_best.pt` when the trainer overwrites it, so the demo tracks the best checkpoint automatically.
- Inference on CPU by default (~0.7 s for 6 s of audio, whole clip in one call; the network is causal with 8 + 4 ms algorithmic latency); `-Cuda` when the GPU is free (~0.4 s incl. transfer).
- Recording is start / stop, 3-30 s; every request draws a **new** scene (unseeded) from the `heldout` pools (val + test: 3,894 files the model never trained on) at a random K-weighted SNR of -10..15 dB.
- The **About the model** card at the bottom reads the checkpoint (`/model_info`): params, training run, validation numbers and the held-out test table (from `model/runs/v3a/eval_test*.csv`).
- Everything is local: `demo/server.py` (Flask) + `demo/static/index.html` (no build step, no external assets).
