"""
data/filter_maestro.py

Cleans MAESTRO v3.0.0, assigns Baroque/Romantic labels, cuts 15-second windows,
and produces piece-disjoint train/validation/test splits.

Usage:
    python data/filter_maestro.py --config config.yaml

Outputs (in config.data.output_dir):
    manifest.csv          — one row per window with split, style, file path
    split_counts.json     — counts per style and split
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------------------------
# Composer → style mapping
# ---------------------------------------------------------------------------
BAROQUE_COMPOSERS = {"Bach", "Scarlatti", "Handel"}
ROMANTIC_COMPOSERS = {"Chopin", "Liszt"}

def _style(composer: str) -> Optional[str]:
    """Return 'baroque', 'romantic', or None if not in scope."""
    for name in BAROQUE_COMPOSERS:
        if name.lower() in composer.lower():
            return "baroque"
    for name in ROMANTIC_COMPOSERS:
        if name.lower() in composer.lower():
            return "romantic"
    return None


def load_and_filter(csv_path: str) -> pd.DataFrame:
    """Load MAESTRO CSV, drop transcriptions and out-of-scope composers."""
    df = pd.read_csv(csv_path)
    # Drop rows with "/" in canonical_composer (e.g. Bach/Liszt transcriptions)
    df = df[~df["canonical_composer"].str.contains("/", na=False)].copy()
    df["style"] = df["canonical_composer"].apply(_style)
    df = df[df["style"].notna()].copy()
    return df


def cap_performances(df: pd.DataFrame, max_perfs: int, seed: int) -> pd.DataFrame:
    """Keep at most max_perfs performances per (composer, title)."""
    rng = random.Random(seed)
    rows = []
    for _, group in df.groupby(["canonical_composer", "canonical_title"]):
        idxs = list(group.index)
        if len(idxs) > max_perfs:
            idxs = rng.sample(idxs, max_perfs)
        rows.extend(idxs)
    return df.loc[rows].reset_index(drop=True)


def piece_disjoint_split(
    df: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """
    Split by (canonical_composer, canonical_title) so no piece appears in
    more than one split. Ratios: 70/15/15.
    """
    work_ids = (df["canonical_composer"] + " | " + df["canonical_title"]).values
    unique_works = np.unique(work_ids)

    # First split: 70% train, 30% temp
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=work_ids))

    # Second split: 50/50 on temp → val and test (each ~15% of total)
    temp_df = df.iloc[temp_idx]
    temp_works = work_ids[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(gss2.split(temp_df, groups=temp_works))

    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[temp_idx[val_rel]] = "val"
    split[temp_idx[test_rel]] = "test"
    df = df.copy()
    df["split"] = split
    return df


def cut_windows(
    row: pd.Series,
    midi_dir: str,
    window_sec: float,
    max_windows: int,
) -> list[dict]:
    """
    Cut a performance into non-overlapping windows.
    Returns list of dicts with window metadata (no actual MIDI slicing here;
    slicing happens at training time using start_sec / end_sec).
    """
    midi_path = os.path.join(midi_dir, row["midi_filename"])
    duration = float(row["duration"])
    n = min(max_windows, int(duration // window_sec))
    windows = []
    for i in range(n):
        windows.append({
            "midi_path": midi_path,
            "start_sec": i * window_sec,
            "end_sec": (i + 1) * window_sec,
            "style": row["style"],
            "split": row["split"],
            "composer": row["canonical_composer"],
            "title": row["canonical_title"],
            "window_id": f"{Path(midi_path).stem}_w{i:02d}",
        })
    return windows


def main(config_path: str) -> None:
    import yaml  # only needed at runtime
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"]
    csv_path = dcfg["maestro_csv"]
    midi_dir = dcfg["maestro_midi_dir"]
    out_dir = Path(dcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = dcfg["random_seed"]

    print("Loading MAESTRO CSV...")
    df = load_and_filter(csv_path)
    print(f"After filtering: {len(df)} performances")
    print("Composer counts:")
    print(df["canonical_composer"].value_counts().to_string())

    df = cap_performances(df, dcfg["max_perfs_per_work"], seed)
    print(f"After capping {dcfg['max_perfs_per_work']} perfs/work: {len(df)} performances")

    df = piece_disjoint_split(df, seed)

    # Cut windows
    all_windows = []
    for _, row in df.iterrows():
        all_windows.extend(
            cut_windows(row, midi_dir, dcfg["window_sec"], dcfg["max_windows_per_perf"])
        )

    manifest = pd.DataFrame(all_windows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest saved to {manifest_path}")

    # Print counts
    counts: dict = {}
    for style in ("baroque", "romantic"):
        counts[style] = {}
        for split in ("train", "val", "test"):
            n = len(manifest[(manifest.style == style) & (manifest.split == split)])
            counts[style][split] = n
            print(f"  {style:10s} {split:5s}: {n:4d} windows")

    with open(out_dir / "split_counts.json", "w") as f:
        json.dump(counts, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
