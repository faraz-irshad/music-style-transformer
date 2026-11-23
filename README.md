# Style-Conditioned Piano Accompaniment Generation

BSc Thesis — ELTE Budapest

## What this project does

Given a fixed melody, the system generates two piano accompaniments: one in Baroque style and one in Romantic style. The melody is held fixed; only the accompaniment changes. This is **not** full style transfer (restyling an existing arrangement) and **not** jazz-to-rock conversion.

Accurate claim: **Style-Conditioned Piano Accompaniment Generation.**

## System variants

Every output file is labeled with its provenance. The labels are:

| Label | Description |
|-------|-------------|
| `base_amt` | Pretrained Anticipatory Music Transformer (stanford-crfm/music-medium-800k, ~360M params), melody as controls only. Neutral baseline. Pretrained on **Lakh MIDI** (not MAESTRO). |
| `layer_a_scripted` | Rule-based generator. **NOT a learned model.** Guarantees audible Baroque vs. Romantic contrast. Never compare it to ML output as if it were ML. |
| `layer_b_amt_history` | **Not implemented (optional).** Would use a short key-matched MAESTRO excerpt as history. |
| `layer_c_lora` | Base AMT + LoRA adapter (adapter_baroque or adapter_romantic), fine-tuned on cleaned MAESTRO. |

## Repository layout

```
amt_utils.py        Shared tokenisation helpers (used by train_lora.py and generate.py)
colab/              Colab notebooks and setup scripts
data/               Data cleaning, splitting, skyline extraction, feature extraction
layer_a/            Rule-based accompaniment generator (dependency-free)
tests/              Unit tests (run on CPU, no GPU required)
docs/               Verified notes, report draft
results/            CSV/JSON metrics, PNG figures, MIDI/WAV outputs (gitignored)
scripts/            Utility scripts (demo index builder)
view/               Results browser UI (React/Vite/TypeScript/Mantine)
config.yaml         Paths and hyperparameters
train_lora.py       LoRA fine-tuning (Colab GPU)
generate.py         Generation pipeline (Colab GPU)
train_classifier.py Style classifier training
evaluate.py         Evaluation metrics and statistical tests
nll_eval.py         Cross-style NLL matrix (Colab GPU)
```

## Licensing

- MAESTRO v3.0.0: CC BY-NC-SA 4.0. Any redistributed MIDI segments or adapters trained on it carry the same terms.
- anticipation library and music-medium-800k weights: Apache 2.0.
- Lakh MIDI (pretraining data for the base AMT): contains scraped commercial songs; memorization cannot be ruled out.

## Commands

### CPU tests (no GPU, no MAESTRO needed)
```bash
pip install mido pretty_midi scipy scikit-learn pyyaml pytest
python -m pytest tests/ -v
# or run directly:
python tests/run_cpu_tests.py
python tests/test_train_lora.py
```

### Data pipeline (needs MAESTRO download first — see PROGRESS.md Step A)
```bash
python data/filter_maestro.py --config config.yaml
python data/build_test_melodies.py --config config.yaml
```

### Classifier training (needs manifest.csv)
```bash
python train_classifier.py --config config.yaml
```

### LoRA training (Colab GPU)
```bash
# Dry run first (~2 min):
python train_lora.py --config config.yaml --style baroque --dry_run
# Full training:
python train_lora.py --config config.yaml --style baroque
python train_lora.py --config config.yaml --style romantic
```

### Generation (Colab GPU)
```bash
python generate.py --config config.yaml
```

### Evaluation
```bash
python evaluate.py --config config.yaml
python nll_eval.py --config config.yaml   # Colab GPU
```

### UI
```bash
python scripts/build_demo_index.py --config config.yaml
cd view && npm install && npm run dev
```

## Colab workflow

See PROGRESS.md → "Needs User" section for step-by-step instructions.
