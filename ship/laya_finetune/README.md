# Laya fine-tune: radio-hub typed decisions

This fine-tunes [Laya](https://github.com/NandhaKishorM/laya), a non-generative "System 1" decision model with 421 M parameters. It learns to answer four questions about each radio transmission:

| Question | Type | Answers |
|---|---|---|
| Message type | choice | contact report / medevac / fire mission / check fire / sitrep / radio check / movement / other |
| Urgency | score | routine / priority / immediate |
| Enemy contact | yes/no | |
| Casualties | yes/no | |

This folder is self-contained, and the data is already built in `data/`.

## Data

| File | Cases | Contents |
|---|---|---|
| `train.jsonl` | 12,000 | Sandbox radio traffic with ground-truth labels: 30 % clean text, 70 % text with ASR-style errors (misheard radio numerals, "grid" → "grade", dropped words) |
| `val.jsonl` | 1,200 | Same kind of text, from different scenario seeds |
| `test.jsonl` | 48 | **Real** full-chain transcripts (TTS → noise → radio link → HubNet → whisper). Small, but it is the realistic check |

The format is LocalLLaMA/typed-decisions, the one Laya's own notebook uses. `make_dataset_reference.py` shows how the data was generated; it needs the `hub_suite` package and is not needed to train.

## Run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124    # match the box's CUDA
pip install -r requirements.txt
```

On Windows:

```powershell
.\run_finetune.ps1
```

On Linux:

```bash
./run_finetune.sh
```

For DDP on Linux:

```bash
NGPU=2 ./run_finetune.sh
```

- **Base weights:** the first run downloads `convaiinnovations/laya` from Hugging Face (about 1.1 GB for the English base only).
- **Method:** RLCD, a GRPO-style policy gradient with a proper-scoring reward plus soft cross-entropy, as in Laya's notebook. Learning rates are 2.5e-5 (encoder) and 1e-4 (head), for 3 epochs. At the end, temperatures are calibrated on a held-out slice.
- **Memory:** gradient checkpointing switches off automatically on GPUs with 30 GB or more. If you run out of memory, drop to `--micro 8 --accum 4`.
- **Time:** about 22 h on an RTX 4050 laptop; roughly 1-2 h expected on a 48 GB card.
- **Output:** the model is saved to `laya_radio/`. `eval_laya.py` then prints base vs fine-tuned accuracy per question and per text source.

**Base-model accuracy to beat:**

| Question | val | real-ASR test |
|---|---|---|
| Message type | 61 % | 63 % |
| Urgency | 48 % | 40 % |
| Enemy contact | 77 % | 73 % |
| Casualties | 86 % | 92 % |

## Use it at the hub

Copy `laya_radio/` back to the laptop at `ANC/hub_suite/artefacts/laya_radio`, then:

```bash
python -m hubsuite.run_hub --laya --laya-checkpoint artefacts/laya_radio
```
