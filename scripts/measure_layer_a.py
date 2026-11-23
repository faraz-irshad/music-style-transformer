"""Render several Layer A examples and compare their classifier/features."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from model.data.features import extract_all
from model.layer_a.generate import TWINKLE_MELODY, generate_accompaniment, notes_to_midi


MELODIES = {
    "twinkle": TWINKLE_MELODY,
    "g_major": [
        (0.0, 0.6, 67), (0.6, 1.2, 69), (1.2, 1.8, 71), (1.8, 2.4, 74),
        (2.4, 3.0, 76), (3.0, 3.6, 74), (3.6, 4.2, 71), (4.2, 4.8, 67),
        (4.8, 5.4, 69), (5.4, 6.0, 71), (6.0, 6.6, 74), (6.6, 7.2, 67),
    ],
    "a_minor": [
        (0.0, 0.6, 69), (0.6, 1.2, 72), (1.2, 1.8, 76), (1.8, 2.4, 72),
        (2.4, 3.0, 71), (3.0, 3.6, 74), (3.6, 4.2, 77), (4.2, 4.8, 74),
        (4.8, 5.4, 72), (5.4, 6.0, 76), (6.0, 6.6, 79), (6.6, 7.2, 76),
    ],
}


def _simple_features(notes) -> dict[str, float]:
    durations = [note.end - note.start for note in notes]
    onsets = sorted(note.start for note in notes)
    intervals = np.diff(onsets)
    intervals = intervals[intervals > 1e-4]
    return {
        "note_count": float(len(notes)),
        "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        "ioi_cv": (
            float(np.std(intervals) / np.mean(intervals))
            if len(intervals) and np.mean(intervals) > 0 else 0.0
        ),
        "pitch_range": float(
            max((note.pitch for note in notes), default=0)
            - min((note.pitch for note in notes), default=0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="results/classifier.pkl")
    parser.add_argument("--output-dir", default="results/layer_a_measurements")
    args = parser.parse_args()

    classifier_path = Path(args.classifier)
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier not found at {classifier_path}; train it first."
        )
    with classifier_path.open("rb") as stream:
        classifier_bundle = pickle.load(stream)
    classifier = classifier_bundle["model"]
    class_names = classifier_bundle["class_names"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    measurements = {}
    for melody_name, melody in MELODIES.items():
        measurements[melody_name] = {}
        for style in ("baroque", "romantic"):
            accompaniment = generate_accompaniment(melody, style, seed=42)
            midi_path = output_dir / f"{melody_name}_{style}.mid"
            notes_to_midi(melody, accompaniment, style, str(midi_path))
            probabilities = classifier.predict_proba(
                [extract_all(accompaniment, block="all")]
            )[0]
            measurements[melody_name][style] = {
                "probabilities": dict(zip(class_names, map(float, probabilities))),
                "features": _simple_features(accompaniment),
            }
            print(
                f"{melody_name}/{style}: "
                f"probabilities={measurements[melody_name][style]['probabilities']} "
                f"features={measurements[melody_name][style]['features']} "
                f"MIDI={midi_path}"
            )

        baroque = measurements[melody_name]["baroque"]["features"]
        romantic = measurements[melody_name]["romantic"]["features"]
        differences = {
            key: romantic[key] - baroque[key] for key in baroque
        }
        print(f"{melody_name}: romantic - baroque feature differences={differences}")


if __name__ == "__main__":
    main()
