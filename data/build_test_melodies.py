"""
data/build_test_melodies.py

Builds the set of test melodies:
  - Up to 30 skyline-extracted melodies from the test split of MAESTRO
  - Hand-written demo melodies (Twinkle Twinkle, Ode to Joy, etc.)

Output: config.data.output_dir/test_melodies/*.mid

Usage:
    python data/build_test_melodies.py --config config.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import pretty_midi

from data.skyline import skyline, notes_to_pretty_midi, Note


HAND_WRITTEN: dict[str, list[tuple[float, float, int]]] = {
    "twinkle": [
        (0.0, 0.6, 60), (0.6, 1.2, 60), (1.2, 1.8, 67), (1.8, 2.4, 67),
        (2.4, 3.0, 69), (3.0, 3.6, 69), (3.6, 4.8, 67),
        (4.8, 5.4, 65), (5.4, 6.0, 65), (6.0, 6.6, 64), (6.6, 7.2, 64),
        (7.2, 7.8, 62), (7.8, 8.4, 62), (8.4, 9.6, 60),
    ],
    "ode_to_joy": [
        (0.0, 0.5, 64), (0.5, 1.0, 64), (1.0, 1.5, 65), (1.5, 2.0, 67),
        (2.0, 2.5, 67), (2.5, 3.0, 65), (3.0, 3.5, 64), (3.5, 4.0, 62),
        (4.0, 4.5, 60), (4.5, 5.0, 60), (5.0, 5.5, 62), (5.5, 6.0, 64),
        (6.0, 7.0, 64), (7.0, 7.5, 62), (7.5, 8.0, 62),
    ],
    "mary_had_a_little_lamb": [
        (0.0, 0.5, 64), (0.5, 1.0, 62), (1.0, 1.5, 60), (1.5, 2.0, 62),
        (2.0, 2.5, 64), (2.5, 3.0, 64), (3.0, 4.0, 64),
        (4.0, 4.5, 62), (4.5, 5.0, 62), (5.0, 6.0, 62),
        (6.0, 6.5, 64), (6.5, 7.0, 67), (7.0, 8.0, 67),
    ],
    "happy_birthday": [
        (0.0, 0.4, 60), (0.4, 0.8, 60), (0.8, 1.6, 62), (1.6, 2.4, 60),
        (2.4, 3.2, 65), (3.2, 4.8, 64),
        (4.8, 5.2, 60), (5.2, 5.6, 60), (5.6, 6.4, 62), (6.4, 7.2, 60),
        (7.2, 8.0, 67), (8.0, 9.6, 65),
    ],
    "fur_elise_opening": [
        (0.0, 0.3, 76), (0.3, 0.6, 75), (0.6, 1.2, 76), (1.2, 1.5, 75),
        (1.5, 1.8, 76), (1.8, 2.1, 71), (2.1, 2.4, 74), (2.4, 2.7, 72),
        (2.7, 3.6, 69),
    ],
}


def extract_test_melody(row: pd.Series, window_sec: float = 15.0) -> list[Note] | None:
    """Extract skyline melody from a MAESTRO test window."""
    try:
        pm = pretty_midi.PrettyMIDI(row["midi_path"])
    except Exception:
        return None
    all_notes: list[Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= row["start_sec"] and n.end <= row["end_sec"]:
                all_notes.append(Note(
                    n.start - row["start_sec"],
                    n.end - row["start_sec"],
                    n.pitch,
                ))
    if len(all_notes) < 10:
        return None
    mel = skyline(all_notes)
    return mel if len(mel) >= 5 else None


def main(config_path: str) -> None:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"]
    out_dir = Path(dcfg["output_dir"]) / "test_melodies"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_target = cfg["generation"]["n_test_melodies"]

    # Save hand-written melodies
    for name, notes_raw in HAND_WRITTEN.items():
        notes = [Note(s, e, p) for s, e, p in notes_raw]
        pm = notes_to_pretty_midi(notes)
        pm.write(str(out_dir / f"handwritten_{name}.mid"))
    print(f"Saved {len(HAND_WRITTEN)} hand-written melodies")

    # Load MAESTRO test split
    manifest_path = Path(dcfg["output_dir"]) / "manifest.csv"
    if not manifest_path.exists():
        print("manifest.csv not found — skipping MAESTRO test melodies")
        print(f"Run data/filter_maestro.py first.")
        return

    manifest = pd.read_csv(manifest_path)
    test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)

    # Stratify by style
    saved = 0
    per_style = n_target // 2
    for style in ("baroque", "romantic"):
        style_rows = test_rows[test_rows["style"] == style]
        count = 0
        for _, row in style_rows.iterrows():
            if count >= per_style:
                break
            mel = extract_test_melody(row, dcfg["window_sec"])
            if mel is None:
                continue
            pm = notes_to_pretty_midi(mel)
            fname = f"maestro_{style}_{row['window_id']}.mid"
            pm.write(str(out_dir / fname))
            count += 1
            saved += 1
        print(f"  {style}: {count} melodies saved")

    print(f"Total test melodies: {saved + len(HAND_WRITTEN)} in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
