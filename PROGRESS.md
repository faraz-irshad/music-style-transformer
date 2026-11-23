# PROGRESS

Style-Conditioned Piano Accompaniment Generation — BSc Thesis, ELTE

## Branch: fixes-and-ui

---

## Review round 2 (2026-11-23)

| Item | Status | Notes |
|------|--------|-------|
| 1. Anticipation interleave/combine/clip | in progress | GitHub `anticipation/ops.py` confirms `anticipate(events, controls, delta)` returns `(tokens, unconsumed_controls)`, `combine(events, controls)` returns tokens, and `clip(tokens, start, end, clip_duration=True, seconds=True)` accepts `seconds`. Removed silent fallbacks and added a Colab interleave/generate check. Runtime check remains UNVERIFIED until run on Colab. |
| 2. Training shuffle and transposition | done | Seed-shuffle windows and each sample epoch; choose one random transposition from -3..+3 per training window instead of concatenating shifts -6..+6. |
| 3. Cache parsing and training window cap | done | Parse each performance once per cache build, slice windows in memory, use per-cache signatures to avoid dry-run/full-run cache collisions, add a 600-window/style cap, and report a first-20-window time estimate (or actual time for smaller runs). |
| 4. Optimizer-step timing | done | Time the complete gradient-accumulation interval through the optimizer update, synchronize CUDA around timing, and report dry-run peak GPU memory (or explicitly report unavailable on CPU). |
| 5. Melody matching in evaluation | done | One-to-one pitch/onset matching at 25 ms, per-file fraction and below-80% flag, aggregate summary, and NaN skyline/clash metrics when matching is insufficient or unavailable. Added focused synthetic tests. |
| 6. Windowed accompaniment metrics and classifier | done | Parse performance MIDI once per path, slice manifest windows, use skyline removal for real-window classifier/distance/memorization features and for distribution/memorization comparisons, and document why real MAESTRO requires the skyline heuristic. |
| 7. Layer C adapter lifetime | done | Load one fresh base plus one adapter for each style, reuse for that style's melodies, and collect/clear GPU cache after the style. Base generation remains a separate first pass before PEFT wrapping. |

## Part 1 — train_lora.py

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P1.1 | Fix gradient-accumulation bug (micro_step counter) + unit test | done | |
| P1.2 | Fix anticipatory interleaving to use library routine; record CONTROL_OFFSET | done | UNVERIFIED on Colab |
| P1.3 | Fix padding: attention_mask, -100 on pad labels | done | |
| P1.4 | Fix split names; ensure val split exists; add test | done | |
| P1.5 | Fix step-50 dominance check: events_to_midi return type; wrap in try/except | done | |
| P1.6 | Cache tokenized windows to disk | done | |
| P1.7 | Assert LoRA weights fp32, base fp16; print dtypes at startup | done | |
| P1.8 | Replace YOUR_USERNAME/YOUR_REPO in colab/05_train_lora.ipynb | done | |
| P1.9 | Fix "trained on MAESTRO" → "pretrained on Lakh MIDI" in docs/comments | done | |
| P1.10 | Add --dry_run mode (5 steps, 8 windows, print loss/s/step/peak mem) | done | |

## Part 2 — generate.py

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P2.11 | Fix adapter contamination: generate base_amt first, load adapters once | done | |
| P2.12 | base_amt: generate once per melody (style=none); update evaluate.py | done | |
| P2.13 | Merge controls into output with library combine; save melody separately | done | UNVERIFIED on Colab |
| P2.14 | Fix events_to_midi return type; log failures to generation_log.csv; exit non-zero >20% | done | |
| P2.15 | WAV: add fluidsynth install to setup_cell.py; pure-numpy fallback | done | |
| P2.16 | Verify layer_a.generate import signatures; add test | done | |
| P2.17 | Move shared tokenization helpers to amt_utils.py | done | |
| P2.18 | Use or delete unused config keys (max_new_tokens, n_test_melodies, bootstrap_n) | done | |
| P2.19 | Write results/manifest_generated.csv during generation | done | |

## Part 3 — evaluate.py and classifier

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P3.20 | Read metadata from manifest CSV, not filename splitting | done | |
| P3.21 | Use known input melody for melody/accompaniment split | done | |
| P3.22 | Actually compute JSD/Wasserstein, block ablation, bootstrap CIs | done | |
| P3.23 | Add nll_eval.py (Colab) | done | needs Colab GPU |
| P3.24 | Memorization: hash-index corpus n-grams; compare generated vs corpus | done | |
| P3.25 | Store class names in classifier pickle; remove unused imports | done | |

## Part 4 — Housekeeping

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P4.26 | Add pyyaml, pytest to requirements.txt; add Commands section to README | done | |
| P4.27 | Set git author to Faraz Irshad | done | |
| P4.28 | Mark Layer B as "not implemented (optional)" in README | done | |

## Part 5 — Cleanup

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P5.cleanup | Find/prove unused items; delete only proven unused; update .gitignore | done | see CLEANUP.md |

## Part 6 — UI

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P6.ui | Replace mock UI with honest results browser | done | npm run build passes |

---

## Needs User (in order)

### Step A — Download MAESTRO v3.0.0
```
# Go to https://magenta.withgoogle.com/datasets/maestro
# Download maestro-v3.0.0-midi.zip (~57 MB), unzip to /content/maestro-v3.0.0/
# Then:
python data/filter_maestro.py --config config.yaml
python data/build_test_melodies.py --config config.yaml
```

### Step B — Smoke test (Colab)
```
# Open colab/00_smoke_test.ipynb in Google Colab, run all cells.
# Report: s/step and peak GPU memory from Cell 6.
```

### Step C — Dry run (Colab, ~2 min)
```
python train_lora.py --config config.yaml --style baroque --dry_run
# Confirm: loss decreases, no OOM, s/step reasonable.
```

### Step D — Full LoRA training (Colab, ~30 min each)
```
python train_lora.py --config config.yaml --style baroque
python train_lora.py --config config.yaml --style romantic
```

### Step E — Generation (Colab)
```
python generate.py --config config.yaml
```

### Step F — Evaluation (CPU or Colab)
```
python train_classifier.py --config config.yaml
python evaluate.py --config config.yaml
python nll_eval.py --config config.yaml   # Colab only
```

### Step G — Build UI index
```
python scripts/build_demo_index.py --config config.yaml
cd view && npm install && npm run build
```
