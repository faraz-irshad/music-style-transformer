# CLEANUP.md

Audit of unused files, imports, and config keys. Part 5 of fixes-and-ui.

## Evidence methodology
All searches run with `grep -rn` across `*.py`, `*.yaml`, `*.ts`, `*.tsx`
(excluding `node_modules` and `.pyc`).

---

## Items deleted

### `controller/` directory (empty)
- Search: `grep -rn "controller" --include="*.py"` → 0 results
- Only reference: a comment in `view/src/services/api.ts` ("will be replaced with controller endpoints")
- The comment itself is being removed as part of the UI rewrite (Part 6)
- **Deleted**: empty directory

### `model/` directory (empty)
- Search: `grep -rn "from model" --include="*.py"` → 0 results
- No references anywhere
- **Deleted**: empty directory

### `__pycache__` directories and `.pyc` files
- Found: `data/__pycache__/`, `layer_a/__pycache__/`
- These are generated artefacts, not source files
- **Removed from git tracking** and added to `.gitignore`

### Unused import: `GroupKFold` in `train_classifier.py`
- Imported: `from sklearn.model_selection import GroupKFold`
- Used: only in a docstring comment, never instantiated
- Search: `grep -rn "GroupKFold(" --include="*.py"` → 0 results
- **Deleted** the import line

---

## Items kept (with justification)

### `config.yaml` keys: `max_new_tokens`, `n_test_melodies`, `bootstrap_n`
- `max_new_tokens`: not currently read by generate.py (generate() uses end_time instead).
  Kept because it documents the intended generation budget and may be used if the
  generate() API changes. Noted as unused in a comment.
- `n_test_melodies`: read by `data/build_test_melodies.py` line 87. **Used.**
- `bootstrap_n`: read by `evaluate.py` line 222. **Used.**

### `data/features.py` — `_tonic_pc` function
- Called internally by `harmony_features` and by `evaluate.py` via import.
- **Used.**

### `view/src/components/AudioPlayer.tsx`
- Reused in the new UI (Part 6). **Kept.**

### `tests/test_layer_a.py` and `tests/test_skyline.py`
- These are pytest-compatible versions of the tests in `run_cpu_tests.py`.
  They require `mido` and `pretty_midi` which are not available in the local
  Python environment, so they cannot be run here. They are valid for Colab.
  **Kept** — do not delete test files.

---

## `.gitignore` additions
- `__pycache__/`
- `*.pyc`
- `*.pyo`
- `results/midi/`
- `results/wav/`
- `results/figures/`
- `results/*.pkl`
- `demo/` contents (large audio) — keep `demo/` directory itself
