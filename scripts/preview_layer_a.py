#!/usr/bin/env python3
"""Create reproducible MIDI and WAV previews of the rule-based Layer A styles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.amt_utils import render_wav
from model.layer_a.generate import (
    generate_accompaniment,
    notes_to_midi,
    plan_harmony,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--melody", required=True, help="Input single-melody MIDI file")
    parser.add_argument("--tempo", type=float, default=100.0, help="Tempo in BPM")
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument(
        "--downbeat-sec",
        type=float,
        default=0.0,
        help="Absolute timestamp of a known bar downbeat; pickups may precede it",
    )
    parser.add_argument("--output-dir", required=True, help="New or empty output directory")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument(
        "--renderer", choices=("fluidsynth", "numpy"), default="fluidsynth"
    )
    parser.add_argument("--soundfont", help="Explicit SoundFont for FluidSynth")
    return parser.parse_args()


def _render(path: Path, renderer: str, soundfont: str | None) -> dict[str, object]:
    try:
        result = render_wav_from_midi(path, renderer, soundfont)
    except Exception as exc:
        result = {
            "success": False,
            "backend": renderer,
            "soundfont": soundfont or "not-resolved",
            "path": str(path.with_suffix(".wav")),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not result["success"]:
        print(f"ERROR: could not render {path.name}: {result['error']}", file=sys.stderr)
    return result


def render_wav_from_midi(
    path: Path, renderer: str = "fluidsynth", soundfont: str | None = None
) -> dict[str, object]:
    import pretty_midi

    return render_wav(
        pretty_midi.PrettyMIDI(str(path)),
        str(path.with_suffix(".wav")),
        backend=renderer,
        soundfont_path=soundfont,
    ).as_dict()


def main() -> int:
    args = _arguments()
    melody_path = Path(args.melody).expanduser().resolve()
    if not melody_path.is_file():
        raise FileNotFoundError(f"input MIDI does not exist: {melody_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = (
        "source_melody.mid",
        "source_melody.wav",
        "layer_a_baroque.mid",
        "layer_a_baroque.wav",
        "layer_a_romantic.mid",
        "layer_a_romantic.wav",
        "preview_metadata.json",
    )
    existing = [name for name in output_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing preview output(s): " + ", ".join(existing)
        )

    import pretty_midi

    source = pretty_midi.PrettyMIDI(str(melody_path))
    melody = sorted(
        (note.start, note.end, note.pitch)
        for instrument in source.instruments
        if not instrument.is_drum
        for note in instrument.notes
    )
    if not melody:
        raise ValueError(f"input MIDI contains no non-drum melody notes: {melody_path}")

    plan = plan_harmony(
        melody, args.tempo, args.beats_per_bar, args.downbeat_sec
    )
    source_midi = output_dir / "source_melody.mid"
    notes_to_midi(
        melody, [], "baroque", str(source_midi), args.tempo,
        args.beats_per_bar, args.downbeat_sec,
    )
    rendering = {
        "source_melody.wav": _render(source_midi, args.renderer, args.soundfont)
    }

    for style, stem in (
        ("baroque", "layer_a_baroque"),
        ("romantic", "layer_a_romantic"),
    ):
        accompaniment = generate_accompaniment(
            melody,
            style,
            tempo_bpm=args.tempo,
            seed=args.seed,
            beats_per_bar=args.beats_per_bar,
            downbeat_sec=args.downbeat_sec,
            harmonic_plan=plan,
        )
        midi_path = output_dir / f"{stem}.mid"
        notes_to_midi(
            melody,
            accompaniment,
            style,
            str(midi_path),
            args.tempo,
            args.beats_per_bar,
            args.downbeat_sec,
        )
        rendering[f"{stem}.wav"] = _render(midi_path, args.renderer, args.soundfont)

    metadata = {
        "input_melody": str(melody_path),
        "tempo_bpm": args.tempo,
        "beats_per_bar": args.beats_per_bar,
        "downbeat_sec": args.downbeat_sec,
        "seed": args.seed,
        "estimated_key": plan.key,
        "mode": plan.mode,
        "harmonic_plan": plan.as_dict(),
        "renderer": args.renderer,
        "soundfont": args.soundfont or "library-default",
        "rendering": rendering,
    }
    metadata_path = output_dir / "preview_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    failures = [name for name, result in rendering.items() if not result["success"]]
    print(f"Preview files written to {output_dir}")
    if failures:
        print("WAV rendering failed for: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
