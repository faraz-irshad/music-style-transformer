# Style-Conditioned Piano Accompaniment Generation

BSc Thesis — ELTE Budapest

## What this project does

Given a fixed melody, the system generates two piano accompaniments: one in Baroque style and one in Romantic style. The melody is held fixed; only the accompaniment changes. This is **not** full style transfer (restyling an existing arrangement) and **not** jazz-to-rock conversion.

Accurate claim: **Style-Conditioned Piano Accompaniment Generation.**

## System variants

Every output file is labeled with its provenance. The labels are:

| Label | Description |
|-------|-------------|
| `base_amt` | Pretrained Anticipatory Music Transformer (stanford-crfm/music-medium-800k, ~360M params), melody as controls only. Neutral baseline. |
| `layer_a_scripted` | Rule-based generator. **NOT a learned model.** Guarantees audible Baroque vs. Romantic contrast. Never compare it to ML output as if it were ML. |
| `layer_b_amt_history` | (Optional) Base AMT with a short key-matched MAESTRO excerpt as history. |
| `layer_c_lora` | Base AMT + LoRA adapter (adapter_baroque or adapter_romantic), trained on cleaned MAESTRO. |

## Repository layout

```
colab/          Colab notebooks and setup scripts
data/           Data cleaning, splitting, skyline extraction
layer_a/        Rule-based accompaniment generator (dependency-free)
tests/          Unit tests (run on CPU, no GPU required)
docs/           Verified notes, report draft
results/        CSV/JSON metrics, PNG figures, MIDI/WAV outputs (gitignored)
config.yaml     Paths and hyperparameters
```

## Licensing

- MAESTRO v3.0.0: CC BY-NC-SA 4.0. Any redistributed MIDI segments or adapters trained on it carry the same terms.
- anticipation library and music-medium-800k weights: Apache 2.0.
- Lakh MIDI (pretraining data): contains scraped commercial songs; memorization cannot be ruled out.

## Running tests (CPU, no GPU needed)

```bash
pip install pretty_midi mido scipy scikit-learn
python -m pytest tests/ -v
```

## Colab workflow

See PROGRESS.md → "Needs User" section for step-by-step instructions.
