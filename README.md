# ancdata — noisy-clean pair generation for the defence AI-ANC headset

Deliverable #1 of the problem statement: *"a scalable dataset pipeline for generating realistic noisy-clean speech pairs."* This repository is that pipeline. It produces, on the fly and reproducibly, 4-second training pairs of `(noisy[boom, ref], clean)` at 16 kHz from public speech and noise corpora, simulated rooms, a physics blast synthesiser, and (once recorded) the team's own headset measurements. It also freezes evaluation sets, scores them, and carries the demo-day guards.

Background reading: [docs/LIT_REVIEW.md](docs/LIT_REVIEW.md). Design notes, decision log, audits and training reports are kept in the team's internal docs (not published here).

## Battlefield dataset v3 (current training set)

Real Lombard speech (GRID + AVID, 6 s snippets) under multi-layer battlefield scenes with gunfire, blasts and a shared capsule channel; 2-ch `[boom, ref]`, target = dry speech x AGC gain. Recipe chosen by two blind listening trials; 16-point leakage audit passes (`scripts/leakage_check.py`); the trained ANC-Net v3 takes test SI-SNR −2.9 → +11.9 dB, STOI 0.70 → 0.87.

```bash
ancdata snippets --seconds 6            # Lombard GRID + AVID -> data/snippets/lombard6s (5,987 x 6 s)
ancdata pools                           # index + voice-screen the noise corpora -> data/pools.parquet
ancdata battlefield-selftest --n 100    # contract, determinism, leakage, SNR bookkeeping, gunfire rate
ancdata battlefield --split test        # frozen float32 test set   (or: pwsh scripts/build_battlefield.ps1 for everything)
ancdata battlefield --split train       # 55k pairs FLAC, memory-aware workers, resumable
ancdata battlefield-report              # data/battlefield_v3/BUILD_REPORT.md + test baseline
python scripts/leakage_check.py         # 16-point leakage audit -> outputs/leakage_report.md
python model/train.py --name v3a --steps 30000 --batch 16   # ANC-Net v3 (825 k params) on the GPU, ~3 h
python model/evaluate_v3.py --ckpt model/runs/v3a/ckpt_best.pt   # test: SI-SNRi, STOI, PESQ, event F1, blackout
```

```python
from ancdata.torch_dataset import MaterializedDataset, BattlefieldDataset
ds = MaterializedDataset("data/battlefield_v3/train", with_events=True)   # map-style, shuffle-friendly
ds = BattlefieldDataset("configs/battlefield.yaml", "train")               # infinite stream, worker-disjoint
```

## Live demo

`pwsh demo/run_demo.ps1` → http://localhost:8765 : record 6 s, the clip is layered with a held-out battlefield scene by the dataset chain, ANC-Net v3 cleans it; three players + detector strip. Details: [demo/README.md](demo/README.md). Model training: `model/train.py`.

## 60-second proof it works (no downloads)

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
ancdata selftest --smoke                            # synthetic fixtures, every stage, 8 checks
pytest -q                                           # 32 tests incl. the smoke run
```

Windows without make: `pwsh make.ps1 smoke`. Linux/DGX: `make smoke`. Docker: `docker build -t ancdata . && docker run --rm ancdata`.

## The real run (hackathon order — mirrors the council workflow)

| Step | Command | Notes |
|---|---|---|
| 0. downloads (background) | `make download` or `pwsh scripts/download_sources.ps1` (`--all` / `-All` adds Zenodo gunshots, RIRS_NOISES, MAD) | LibriSpeech 6 GB, ESC-50 0.6 GB. Lombard GRID and UrbanSound8K need a form: see script header |
| 1. rooms | `ancdata rir` | 300 simulated rooms x 4 RIRs -> `data/sources/rir/bank.npz`; `--n-per-preset 400` for 2000 |
| 1b. snippets | `ancdata snippets --seconds 6` | 6 s labelled Lombard snippets (GRID + AVID) -> `data/snippets/lombard6s` |
| 1c. layering trial | `python scripts/layer_trial.py --n 10` | 4 layering recipes x 10 snippets -> `outputs/layering_trial/listen.html` (blind; key.json) |
| 2. index | `ancdata registry` | manifest.parquet, hash splits, voice screen; flagged noise -> `data/voice_flagged.txt` (hand-check) |
| 3. checks | `ancdata selftest --config configs/train.yaml` | scorer, contract, determinism, leakage, SNR, spans, non-silent, sources, real-support |
| 4. session | (internal session script) | `ancdata mic-sweep` -> record -> `ancdata mic-fit`; `ancdata lombard-fit`; reality-check takes |
| 5. freeze evals | `make evals` | `eval/standard`, `eval/generalization`, `eval/lombard` (+ `ancdata realcheck-mix` for row 3a) |
| 6. baseline | `make baseline` | unprocessed-input SI-SNR / STOI / PESQ per category and per transient SNR |
| 7. plots | `ancdata plots`; `ancdata reality-gap --synthetic data/eval/standard --real data/sources/realcheck` | crest/attack and the reality-gap figure |
| 8. train | `from ancdata.torch_dataset import PairDataset` | infinite, seeded, worker-safe |
| 9. venue day | `ancdata demo-check --ranges data/eval/standard/ranges.json --wav capture.wav` | refuses if the live chain is outside training ranges |

## Using it from training code

```python
from ancdata import stream, load_config
for pair in stream("configs/train.yaml", "train"):
    pair.noisy   # (2, 64000) float32: [boom mic, reference mic]
    pair.clean   # (64000,)   float32: dry calm speech, sample-aligned with noisy[0]
    pair.events  # [(start, end, category, peak_ratio_db)] transient labels
    pair.meta    # snr_db, room, speaker_id, lombard params, adc headroom, ...
```

Torch: `PairDataset(cfg, "train", mono=True, with_events=True)` is an `IterableDataset`; each worker generates a disjoint index progression, so the union is exactly the single-process sequence. Front end is the model's choice (pairs are raw waveforms); the model uses an asymmetric STFT 256 analysis / 128 synthesis, hop 64 (8 ms + 4 ms). Score through `ancdata.metrics.chunked(hop=64)` so the slide number matches the live latency.

## Layout

```
ancdata/      config, paths, audio, registry, snippets, pools, sampler, rir_gen, physics, lombard, adc, realcheck,
              loudness, transients, channel, battlefield, build, listen (v3)
              chain, stream, materialize, metrics, plots, demo_check, selftest, fixtures, cli, torch_dataset
configs/      battlefield.yaml (v3 spec), train.yaml (v2), train_codec.yaml, eval_standard/generalization/lombard.yaml, smoke.yaml
scripts/      download_sources.sh/.ps1, pull_battlefield.sh, zget.sh, layer_trial.py, build_battlefield.sh/.ps1
docs/         LIT_REVIEW.md
tests/        pytest suite (fixtures are synthetic; no downloads)
data/         ANC_DATA_ROOT (git-ignored): sources/, manifest.parquet, eval/
```

## Licences of the sources

Lombard GRID CC BY 4.0 · AVID CC BY 4.0 · LibriSpeech CC BY 4.0 · ESC-50 CC BY-NC 3.0 and UrbanSound8K CC BY-NC 4.0 (prototype only — replace for a product) · MAD CC BY 4.0 · FSD50K per-clip CC (metadata) · DEMAND CC BY-SA 4.0 · IDMT-Traffic CC BY 4.0 · Zenodo gunshot range set (see record) · DroneAudioDataset research · in-house recordings: signed consent forms. The registry / pools store the licence per file.

## Radio hub pipeline (v3.1, 2026-09-27)

The model is deployed at a hub that receives tactical radio chatter and cleans it for the officer in charge.

- **Dataset v3.1** (`configs/battlefield_v31.yaml`):
  - the v3 battlefield scenes with fixed layering logic;
  - noise pools audited with an AudioSet tagger (`ancdata/pool_audit.py`), and labels kept only for audible sounds;
  - a tactical radio link on 90 % of clips (`ancdata/radio.py`: CVSD 16/32 kbit/s with burst bit errors, narrowband FM with fading, clicks and squelch, push-to-talk clipping);
  - radio-procedure speech (`ancdata/speech_extra.py`: ATCOSIM, Speech Commands, capped Piper TTS), with whisper keyword timings.
- **Build:**
  ```
  ancdata radio-snippets
  ancdata snippet-words --set radio6s
  ancdata battlefield --config configs/battlefield_v31.yaml --split test   # then val, train
  ```
- **Model:** HubNet (`model/hub_net.py`), 2.6 M params, 18 ms algorithmic latency, single channel, 0-4 kHz. It uses dual-path GRUs and a complex deep filter.
- **Train and evaluate:** `model/train_hub.py`, `model/evaluate_hub.py`. The targets are SI-SNR > 15 dB, STOI > 0.85 and PESQ-NB > 2.5 against the radio-band target.
- **Standalone workstation trainer:** `ship/hubnet_standalone/` (see its README).
