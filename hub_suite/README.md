# hub_suite: quality-of-life features around the radio-hub enhancer

This folder is separate from the dataset and training code. It holds copies of the artefacts it needs (the HubNet checkpoint, the radio link, the TTS voice, the ASR / DNSMOS models). It does not include the dataset; the field-adaptation replay reads that from `../data/battlefield_v31`. Research notes are in `docs/`.

```
radio audio ─▶ streaming HubNet (CUDA graph, 18 ms, 0.75-1.2 ms per 6 ms hop) ─▶ officer headset (+ comfort bed, ducked)
                        │
                        ├─▶ segmenter (squelch line, or VAD 1 s + "over"/"out")
                        │     ├─▶ live ASR partials (sherpa-onnx streaming Zipformer, CPU, RTF 0.09)
                        │     └─▶ final ASR (faster-whisper, radio-procedure prompt + hotwords, 0.1-0.5 s)
                        │           └─▶ System 1: rules (grids, callsigns, types) + Laya typed decisions (~50 ms)
                        │                 └─▶ situation picture ─▶ live map (Leaflet + milsymbol, SSE)
                        │                        ▲                     └─▶ MCP server (read + operator markers)
                        │                        └── System 2: local LLM (llama.cpp / Qwen3), async, grounded
                        └─▶ field buffer ─▶ nightly gated self-training ─▶ shadow ─▶ promote / rollback
```

## Quick start (laptop)

```bash
# from ANC/hub_suite, with the ANC venv
../.venv/Scripts/python.exe -m hubsuite.run_hub --events 12 --hold 600        # sandbox -> http://127.0.0.1:8770
../.venv/Scripts/python.exe -m hubsuite.run_hub --events 12 --laya --comfort line_alive
artefacts/llm/llama.cpp/llama-server.exe -m artefacts/llm/Qwen3-4B-Q4_K_M.gguf -ngl 99 -c 8192 --port 8080 &
../.venv/Scripts/python.exe -m hubsuite.run_hub --events 12 --llm http://127.0.0.1:8080/v1
../.venv/Scripts/python.exe -m hubsuite.mcp_server                            # MCP (stdio) for any MCP client
```

## 1. Comfort bed ("soothing frequency"), `hubsuite/comfort.py`

**Evidence** (docs/RESEARCH_comfort_and_field_learning.md §A):
- No frequency calms people through some special property. The claims for "432 Hz", solfeggio tones and binaural-beat entrainment are not supported.
- Binaural beats need stereo and a 400-500 Hz carrier, and radio is mono.
- Sound that changes over time while someone is listening harms serial recall, and callsigns and grid references are exactly that kind of material.
- Slow paced breathing (~6 breaths/min) has evidence, but only when the listener breathes along.

**Design:**
- Plays in the officer's headset only, never on transmit. Opt-in, off by default, with a kill switch and automatic off after 2 h.
- Brown noise below 250 Hz, so it does not overlap the 300-3400 Hz speech band, at least 25 dB under the speech.
- Optional "pacer" mode: 4 s in / 6 s out swells, only while idle.
- **Muted whenever speech is present.** The ducking detector reads HubNet's *input*, which is 18 ms ahead of its output. The bed therefore cuts out before the speech arrives and adds no latency.

**Measured:**
- The bed sounded for 37 % of a stream with gaps between transmissions.
- Leakage into speech frames: −98 dB.
- STOI unchanged: 1.000 with the bed and 1.000 without.

## 2. Speech-to-text backup, `hubsuite/stt.py`

| Tier | Engine | Runs on | Speed |
|---|---|---|---|
| Live partials | sherpa-onnx streaming Zipformer (int8) | 1 CPU core | RTF 0.09 |
| Final text, on squelch close | faster-whisper, temperature 0, radio-procedure prompt, hotwords, word confidences | GPU | 0.1-0.5 s |

- Final-pass model: small.en on the laptop, large-v3-turbo on the 48 GB workstation.
- Radio numerals are normalised by `hubsuite/radiotext.py`: niner / fife / tree / fower, and context-aware mishearings such as "grade" → grid and "power" → 4 next to digits.

## 3. Live map, `hubsuite/situation.py`, `mapserver.py`, `map.html`, `mcp_server.py`, `sys2.py`

- **System 1 (hot path, deterministic).** Rules produce the grids, callsigns, type, size, direction and casualties. **Laya** answers typed questions (message type, urgency, enemy contact, casualties) in one forward pass of about 50 ms. It is non-generative, so it has nothing to hallucinate.
- **Grid confidence.** A grid is shown as *confirmed* only if the mean ASR probability of its digit words is at least 0.75. On 60 sandbox transmissions this setting confirmed none of the 5 wrong grids; a wrong grid is drawn dashed as UNCONFIRMED. Every marker keeps its source transmission, so clicking it plays the audio.
- **System 2 (background).** A local LLM runs behind an OpenAI-compatible server (llama.cpp): Qwen3-32B Q5_K_M on the 48 GB box, Qwen3-4B on the laptop. It does schema-constrained JSON extraction, and a field is kept **only if it is grounded in the transcript**. Its Q&A answers must cite transmission ids, and the model never creates markers.
- **MCP server.** Tools: `get_situation`, `list_contacts`, `recent_transmissions`, `add_marker` (writes an operator marker) and `grid_to_latlon`. Laya's own MCP server (`laya-mcp-server`) can sit alongside it.
- **Sandbox** (`hubsuite/sim.py`). A scripted scenario in the 33U VP square: units move and transmit contact reports, SITREPs, MEDEVACs, fire missions, check fire, movement and radio checks, all with true grids. The audio chain is Piper TTS (904 voices) → synthetic battlefield noise → tactical radio link.

**Measured, 60 sandbox transmissions:**

| | Laptop, early HubNet checkpoint (step 8k) |
|---|---|
| End of speech → marker | p50 615 ms, p90 737 ms |
| Grid reports placed on the exact square | 28 / 57 |
| Wrong grids | 5, all shown UNCONFIRMED at the 0.75 threshold |
| HubNet streaming | 1.2 ms per 6 ms hop (p95) |

The limiting factor is ASR on radio audio behind the early HubNet checkpoint. It should improve with the workstation-trained HubNet and large-v3-turbo.

**Laya on radio traffic, base model, 51 ms per transmission:**

| Question | val | real-ASR test |
|---|---|---|
| Message type | 61 % | 63 % |
| Urgency | 48 % | 40 % |
| Enemy contact | 77 % | 73 % |
| Casualties | 86 % | 92 % |

**Fine-tuning Laya** (`finetune/`):
- **Data:** 12k training cases (×4 questions) of labelled radio traffic from the sandbox, as clean text plus ASR-style corruption. The test set is real full-chain transcripts.
- **Method:** RLCD, taken from Laya's own notebook.
- **Speed:** on the laptop it runs at 1.7 items/s, about 22 h in total, so run it on the workstation with `./finetune/run_finetune.sh` (about 1-2 h there). Then pass `--laya --laya-checkpoint artefacts/laya_radio`.

## 4. Learning from incoming data ("RL"), `hubsuite/adapt.py`

**Why not policy-gradient RL** (docs/RESEARCH_comfort_and_field_learning.md §B):
- There is no clean reference in the field.
- Non-intrusive quality scores can be gamed: systems that maximised DNSMOS ranked *worst* with human listeners in CHiME-7 UDASE.
- HubNet is deterministic, and a reward-optimised model can learn to "invent" clean-sounding speech.

**What it does instead: gated self-training.**
- **Teacher.** The promoted model produces estimates for the field segments.
- **Gate.** A segment is kept only if:
  - DNSMOS-BAK improves by at least 0.5;
  - the teacher and its EMA shadow agree (SI-SNR ≥ 15 dB);
  - optionally, whisper finds the same keywords in both.
- **Training pairs:**
  - additive remix;
  - re-degradation through the radio link, with harvested field noise;
  - at least 50 % synthetic v3.1 replay with true targets.
- **Student.** Learning rate 1e-5, plus an anchor to the teacher.
- **Promotion gate:**
  - no regression on the synthetic test probe (SI-SNR / STOI / PESQ-NB);
  - no invented output on silence;
  - a field probe, when available;
  - then a shadow period.
- **Records.** Every round is written to the registry, with a one-command rollback.

The smoke test ran end to end. Its candidate was correctly *rejected* because its output on silence rose by 4 dB. That rule now uses a −60 dB absolute floor.

Decisions still open (§B5):
- Is recording field radio audio allowed? That is a policy question.
- Can scripted probe transmissions be sent over the real radios? They are the only in-field reference.

## Workstation (48 GB) settings

| Piece | Model | Approx. memory |
|---|---|---|
| HubNet | trained by `ship/hubnet_standalone` | 0.2 GB |
| Final ASR | large-v3-turbo | ~3 GB |
| System 1 | Laya (fine-tuned) | ~2 GB |
| System 2 | Qwen3-32B Q5_K_M (llama.cpp, `-ngl 99`) | ~24 GB |
| Headroom | | ~18 GB |
