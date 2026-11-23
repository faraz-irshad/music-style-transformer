"""
train_classifier.py

Trains an L2-regularized logistic regression (cross-checked with random forest)
on real MAESTRO windows (Baroque vs. Romantic), group-split by work.

Usage:
    python train_classifier.py --config config.yaml

Outputs:
    results/classifier.pkl       — trained LogisticRegression
    results/classifier_report.json
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data.features import Note, extract_all


# ---------------------------------------------------------------------------
# Load windows from manifest
# ---------------------------------------------------------------------------

def load_window_notes(row: pd.Series) -> list[Note]:
    """Load notes from a MIDI window (start_sec to end_sec)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(row["midi_path"])
    notes: list[Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= row["start_sec"] and n.end <= row["end_sec"]:
                notes.append(Note(n.start - row["start_sec"],
                                  n.end - row["start_sec"],
                                  n.pitch, n.velocity))
    return notes


def augment_transpose(notes: list[Note], semitones: int) -> list[Note]:
    return [Note(n.start, n.end, n.pitch + semitones, n.velocity) for n in notes]


def build_dataset(
    manifest: pd.DataFrame,
    split: str,
    block: str = "all",
    augment: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X (features), y (0=baroque, 1=romantic), groups (work id for GroupKFold).
    """
    rows = manifest[manifest["split"] == split].reset_index(drop=True)
    X_list, y_list, g_list = [], [], []
    label_map = {"baroque": 0, "romantic": 1}

    for _, row in rows.iterrows():
        try:
            notes = load_window_notes(row)
        except Exception as e:
            print(f"  skip {row['window_id']}: {e}")
            continue
        if len(notes) < 10:
            continue

        work_id = f"{row['composer']} | {row['title']}"
        label = label_map[row["style"]]

        for shift in ([0] + list(range(-6, 7)) if augment else [0]):
            aug_notes = augment_transpose(notes, shift) if shift != 0 else notes
            feats = extract_all(aug_notes, block=block)
            X_list.append(feats)
            y_list.append(label)
            g_list.append(work_id)

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int32),
            np.array(g_list))


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    C: float = 1.0,
) -> dict:
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
    ])
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    lr_f1 = f1_score(y_test, lr.predict(X_test), average="macro")
    rf_f1 = f1_score(y_test, rf.predict(X_test), average="macro")

    return {
        "lr_accuracy": lr_acc,
        "lr_macro_f1": lr_f1,
        "rf_accuracy": rf_acc,
        "rf_macro_f1": rf_f1,
        "lr_model": lr,
    }


def block_ablation(
    manifest: pd.DataFrame, C: float = 1.0
) -> dict:
    results = {}
    for block in ("harmony", "texture", "rhythm", "all"):
        print(f"  block={block} ...")
        X_tr, y_tr, g_tr = build_dataset(manifest, "train", block=block)
        X_te, y_te, _ = build_dataset(manifest, "test", block=block, augment=False)
        if len(X_tr) == 0 or len(X_te) == 0:
            print(f"    no data for block={block}")
            continue
        res = train_and_evaluate(X_tr, y_tr, g_tr, X_te, y_te, C=C)
        results[block] = {
            "lr_accuracy": res["lr_accuracy"],
            "lr_macro_f1": res["lr_macro_f1"],
            "rf_accuracy": res["rf_accuracy"],
            "rf_macro_f1": res["rf_macro_f1"],
        }
        print(f"    LR acc={res['lr_accuracy']:.3f} F1={res['lr_macro_f1']:.3f} "
              f"| RF acc={res['rf_accuracy']:.3f} F1={res['rf_macro_f1']:.3f}")
    return results


def main(config_path: str) -> None:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    manifest_path = Path(cfg["data"]["output_dir"]) / "manifest.csv"
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    C = cfg["evaluation"]["classifier_C"]

    manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(manifest)} windows")

    print("Running block ablation...")
    ablation = block_ablation(manifest, C=C)

    # Train final classifier on all features
    print("Training final classifier (all features)...")
    X_tr, y_tr, g_tr = build_dataset(manifest, "train", block="all")
    X_te, y_te, _ = build_dataset(manifest, "test", block="all", augment=False)
    res = train_and_evaluate(X_tr, y_tr, g_tr, X_te, y_te, C=C)

    report = {
        "final_lr_accuracy": res["lr_accuracy"],
        "final_lr_macro_f1": res["lr_macro_f1"],
        "final_rf_accuracy": res["rf_accuracy"],
        "final_rf_macro_f1": res["rf_macro_f1"],
        "gate_passed": res["lr_accuracy"] >= 0.75,
        "block_ablation": ablation,
    }

    with open(results_dir / "classifier_report.json", "w") as f:
        json.dump(report, f, indent=2)
    # Store class names alongside the model so evaluate.py never assumes 0=baroque
    clf_bundle = {"model": res["lr_model"], "class_names": ["baroque", "romantic"]}
    with open(results_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf_bundle, f)

    print(f"\nFinal LR: acc={res['lr_accuracy']:.3f} F1={res['lr_macro_f1']:.3f}")
    print(f"Gate (>=0.75): {'PASSED' if report['gate_passed'] else 'FAILED — audit features'}")
    print(f"Saved to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
