"""
generate.py

For each test melody, generates Baroque and Romantic accompaniments using:
  - Base AMT (layer: base_amt)
  - Layer A scripted (layer: layer_a_scripted)
  - Layer C LoRA (layer: layer_c_lora)

Run on Colab GPU via colab/06_generate.ipynb.

Output filenames: {layer}_{melody_id}_{style}.mid / .wav
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import torch
import yaml


STYLES = ("baroque", "romantic")
LAYERS = ("base_amt", "layer_a_scripted", "layer_c_lora")


# ---------------------------------------------------------------------------
# Melody loading
# ---------------------------------------------------------------------------

def load_melody_notes(midi_path: str) -> list[tuple[float, float, int]]:
    """Load melody as (start, end, pitch) from a single-track MIDI file."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))
    return sorted(notes)


def melody_to_controls(
    melody_notes: list[tuple[float, float, int]],
    control_offset: int,
) -> list[int]:
    """Convert melody notes to anticipation control tokens."""
    from data.skyline import Note as SNote
    from train_lora import _notes_to_events
    snotes = [SNote(s, e, p) for s, e, p in melody_notes]
    events = _notes_to_events(snotes)
    return [t + control_offset for t in events]


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_base_amt(
    model,
    melody_notes: list[tuple[float, float, int]],
    control_offset: int,
    duration: float,
    top_p: float,
    seed: int,
) -> list:
    """Base AMT: melody as controls, no style adapter."""
    from anticipation.sample import generate
    controls = melody_to_controls(melody_notes, control_offset)
    torch.manual_seed(seed)
    with torch.no_grad():
        output = generate(model, start_time=0, end_time=duration,
                          inputs=[], controls=controls, top_p=top_p)
    return output


def generate_layer_a(
    melody_notes: list[tuple[float, float, int]],
    style: str,
    seed: int,
) -> "pretty_midi.PrettyMIDI":
    """Layer A scripted: rule-based, no ML."""
    from layer_a.generate import generate_accompaniment, notes_to_midi
    import tempfile, pretty_midi
    acc = generate_accompaniment(melody_notes, style, seed=seed)
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp.close()
    notes_to_midi(melody_notes, acc, style, tmp.name)
    pm = pretty_midi.PrettyMIDI(tmp.name)
    os.unlink(tmp.name)
    return pm


def generate_layer_c(
    base_model,
    adapter_path: str,
    melody_notes: list[tuple[float, float, int]],
    control_offset: int,
    duration: float,
    top_p: float,
    seed: int,
) -> list:
    """Layer C: base model + LoRA adapter."""
    from peft import PeftModel
    from anticipation.sample import generate
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    controls = melody_to_controls(melody_notes, control_offset)
    torch.manual_seed(seed)
    with torch.no_grad():
        output = generate(model, start_time=0, end_time=duration,
                          inputs=[], controls=controls, top_p=top_p)
    return output


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def events_to_pm(events: list) -> "pretty_midi.PrettyMIDI":
    from anticipation.convert import events_to_midi
    return events_to_midi(events)


def save_midi(pm, path: str) -> None:
    pm.write(path)


def save_wav(pm, path: str, sr: int = 44100) -> None:
    """Render WAV using pretty_midi's fluidsynth synthesis if available."""
    try:
        audio = pm.fluidsynth(fs=sr)
        import scipy.io.wavfile as wf
        import numpy as np
        wf.write(path, sr, (audio * 32767).astype(np.int16))
    except Exception as e:
        print(f"  WAV render failed ({e}), skipping {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    import anticipation.config as acfg

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gcfg = cfg["generation"]
    top_p = gcfg["top_p"]
    seed = gcfg["random_seed"]
    duration = cfg["data"]["window_sec"]
    drive_dir = cfg["training"]["drive_dir"]

    results_dir = Path(cfg["evaluation"]["results_dir"])
    midi_dir = results_dir / "midi"
    wav_dir = results_dir / "wav"
    midi_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Load test melodies
    melody_dir = Path(cfg["data"]["output_dir"]) / "test_melodies"
    melody_files = sorted(melody_dir.glob("*.mid"))
    if not melody_files:
        print(f"No melody files found in {melody_dir}")
        return
    print(f"Found {len(melody_files)} test melodies")

    control_offset = acfg.CONTROL_OFFSET
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model once
    model_name = cfg["model"]["name"]
    try:
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
    except Exception:
        from transformers import GPT2LMHeadModel
        base_model = GPT2LMHeadModel.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
    base_model.eval()

    for mel_path in melody_files:
        mel_id = mel_path.stem
        melody_notes = load_melody_notes(str(mel_path))
        print(f"\nMelody: {mel_id} ({len(melody_notes)} notes)")

        for style in STYLES:
            # --- Base AMT ---
            try:
                events = generate_base_amt(
                    base_model, melody_notes, control_offset, duration, top_p, seed
                )
                pm = events_to_pm(events)
                out_stem = f"base_amt_{mel_id}_{style}"
                save_midi(pm, str(midi_dir / f"{out_stem}.mid"))
                save_wav(pm, str(wav_dir / f"{out_stem}.wav"))
                print(f"  base_amt {style}: ok")
            except Exception as e:
                print(f"  base_amt {style}: FAILED — {e}")

            # --- Layer A scripted ---
            try:
                pm = generate_layer_a(melody_notes, style, seed)
                out_stem = f"layer_a_scripted_{mel_id}_{style}"
                save_midi(pm, str(midi_dir / f"{out_stem}.mid"))
                save_wav(pm, str(wav_dir / f"{out_stem}.wav"))
                print(f"  layer_a_scripted {style}: ok")
            except Exception as e:
                print(f"  layer_a_scripted {style}: FAILED — {e}")

            # --- Layer C LoRA ---
            adapter_path = str(Path(drive_dir) / f"adapter_{style}" / "final")
            if not Path(adapter_path).exists():
                print(f"  layer_c_lora {style}: SKIPPED (adapter not found: {adapter_path})")
                continue
            try:
                events = generate_layer_c(
                    base_model, adapter_path, melody_notes,
                    control_offset, duration, top_p, seed
                )
                pm = events_to_pm(events)
                out_stem = f"layer_c_lora_{mel_id}_{style}"
                save_midi(pm, str(midi_dir / f"{out_stem}.mid"))
                save_wav(pm, str(wav_dir / f"{out_stem}.wav"))
                print(f"  layer_c_lora {style}: ok")
            except Exception as e:
                print(f"  layer_c_lora {style}: FAILED — {e}")

    print(f"\nGeneration complete. MIDI: {midi_dir}, WAV: {wav_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
