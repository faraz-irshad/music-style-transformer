# PROGRESS

Style-Conditioned Piano Accompaniment Generation — BSc Thesis, ELTE

## Task Status

| ID | Task | Status | Notes |
|----|------|--------|-------|
| 0.1 | Read repo, create PROGRESS.md | done | |
| 0.2 | Create docs/VERIFIED_NOTES.md | done | |
| 0.3 | Write README.md with provenance rules | done | |
| 0.4 | Write requirements.txt | done | |
| 0.5 | Write colab/setup_cell.py (pinned installs) | done | |
| 0.6 | Write colab/00_smoke_test.ipynb | done | needs user: run on Colab GPU |
| 1.1 | Write data/filter_maestro.py (cleaning + splits) | done | needs user: download MAESTRO v3.0.0 |
| 1.2 | Test data/filter_maestro.py on CPU (dry-run with mock CSV) | done | |
| 2.1 | Implement skyline extractor in data/skyline.py | done | |
| 2.2 | Unit tests for skyline extractor | done | |
| 2.3 | Build 30 held-out test melodies script | todo | blocked on Task 1 (needs MAESTRO) |
| 3.1 | Improve Layer A: arbitrary melody, chord detection, key transposition | done | |
| 3.2 | Layer A: WAV preview rendering (fluidsynth/pretty_midi) | done | |
| 3.3 | Layer A tests: valid MIDI, melody preserved, styles differ, deterministic | done | |
| 4.1 | Evaluation classifier (logistic regression + random forest) | done | needs user: run after Task 1 |
| 4.2 | Feature extraction (harmony, texture, rhythm blocks) | done | |
| 4.3 | JSD/Wasserstein and other metrics | done | |
| 5.1 | Write train_lora.py | done | needs user: run on Colab GPU |
| 5.2 | Write colab/05_train_lora.ipynb | done | needs user: run on Colab GPU |
| 6.1 | Write generate.py | done | needs user: run on Colab GPU |
| 6.2 | Write colab/06_generate.ipynb | done | needs user: run on Colab GPU |
| 7.1 | Write evaluate.py | done | needs user: run after generation |
| 8.1 | Demo index (README table) | todo | blocked on generation outputs |
| 8.2 | Write docs/REPORT_DRAFT.md | in progress | |

## Needs User

### Step A — Download MAESTRO v3.0.0
1. Go to https://magenta.withgoogle.com/datasets/maestro
2. Download `maestro-v3.0.0-midi.zip` (MIDI only, ~57 MB)
3. Unzip to a local folder, e.g. `/content/maestro-v3.0.0/`
4. Note the path to `maestro-v3.0.0.csv` inside the zip
5. Update `config.yaml` with `maestro_csv` and `maestro_midi_dir` paths
6. Run: `python data/filter_maestro.py --config config.yaml`

### Step B — Run smoke test (Task 0.6)
1. Open `colab/00_smoke_test.ipynb` in Google Colab
2. Run all cells; confirm no import errors and generation produces a MIDI file
3. Report back: s/step for LoRA benchmark and peak GPU memory

### Step C — Run LoRA training (Task 5)
1. Ensure Step A and Step B are done
2. Open `colab/05_train_lora.ipynb` in Google Colab
3. Mount Google Drive, set `DRIVE_DIR` in the first cell
4. Run all cells; training takes ~30 min per adapter on T4
5. Adapters saved to `$DRIVE_DIR/adapter_baroque/` and `$DRIVE_DIR/adapter_romantic/`

### Step D — Run generation (Task 6)
1. Ensure Step C is done
2. Open `colab/06_generate.ipynb` in Google Colab
3. Set adapter paths and melody directory
4. Run all cells; outputs saved to `results/midi/` and `results/wav/`

### Step E — Run evaluation (Task 7)
1. Ensure Step D is done
2. Run: `python evaluate.py --config config.yaml`
3. Results written to `results/*.csv` and `results/figures/*.png`
