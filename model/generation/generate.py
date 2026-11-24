"""
generate.py

For each test melody, generates Baroque and Romantic accompaniments using:
  - Base AMT  (layer: base_amt)         -- generated ONCE per melody, style=none
  - Layer A   (layer: layer_a_scripted) -- rule-based, NOT a learned model
  - Layer C   (layer: layer_c_lora)     -- base + LoRA adapter

Run on Colab GPU via colab/06_generate.ipynb.

Output filenames:
  base_amt_{mel_id}_none.mid/.wav
  layer_a_scripted_{mel_id}_{style}.mid/.wav
  layer_c_lora_{mel_id}_{style}.mid/.wav
  melody_{mel_id}.mid   -- input melody saved separately for evaluation

UNVERIFIED (require Colab run):
  - anticipation.ops combine/clip routine name and signature
  - events_to_midi return type (normalised in amt_utils.events_to_pretty_midi)
"""
from __future__ import annotations
import argparse, csv, sys
from collections import Counter
from pathlib import Path

import torch
import yaml

STYLES = ("baroque", "romantic")


# ---------------------------------------------------------------------------
# Melody loading
# ---------------------------------------------------------------------------

def load_melody_notes(midi_path: str) -> list[tuple[float, float, int]]:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))
    return sorted(notes)


def melody_endpoint(melody_notes: list[tuple[float, float, int]]) -> float:
    """Return the last melody note's endpoint for generation-bound selection."""
    return max((end for _, end, _ in melody_notes), default=0.0)


def save_melody_midi(melody_notes: list[tuple[float, float, int]], path: str) -> None:
    """Save the input melody as a separate MIDI file for evaluation."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="melody")
    for s, e, p in melody_notes:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=s, end=e))
    pm.instruments.append(inst)
    pm.write(path)


# ---------------------------------------------------------------------------
# Base model loader with LoRA contamination guard
# ---------------------------------------------------------------------------

def load_clean_base(model_name: str, device: str):
    """Load a fresh base model with no LoRA modules."""
    import traceback

    try:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
    except Exception:
        auto_error = traceback.format_exc()
        print("AutoModelForCausalLM failed; trying GPT2LMHeadModel:\n" + auto_error)
        from transformers import GPT2LMHeadModel
        try:
            m = GPT2LMHeadModel.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(device)
        except Exception as fallback_error:
            raise RuntimeError(
                "base-model loading failed in both AutoModelForCausalLM and "
                f"GPT2LMHeadModel fallback.\nAutoModel traceback:\n{auto_error}"
            ) from fallback_error
    m.eval()
    return m


def _assert_no_lora(model, label: str) -> None:
    """Fail loudly if the model has any LoRA modules injected."""
    lora_modules = [n for n, _ in model.named_modules() if "lora_" in n.lower()]
    if lora_modules:
        raise RuntimeError(
            f"{label}: model has LoRA modules — do not reuse a base model "
            f"after wrapping with PEFT. Found: {lora_modules[:5]}"
        )


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _melody_to_controls(
    melody_notes: list[tuple[float, float, int]], control_offset: int
) -> list[int]:
    from model.amt_utils import melody_notes_to_controls
    return melody_notes_to_controls(melody_notes, control_offset)


def _combine_and_slice(
    generated_events: list[int],
    controls: list[int],
    duration: float,
) -> list[int]:
    """Clip generated events; preserve complete melody controls in the result."""
    from anticipation import ops
    from anticipation import vocab

    clipped_events = ops.clip(generated_events, 0, duration, seconds=True)
    combined = ops.combine(clipped_events, controls)
    generated_pairs = Counter(
        (time, pitch)
        for time, _note_duration, pitch in zip(
            clipped_events[0::3], clipped_events[1::3], clipped_events[2::3]
        )
    )
    melody_pairs = [
        (time - vocab.CONTROL_OFFSET, pitch - vocab.CONTROL_OFFSET)
        for time, _duration, pitch in zip(
            controls[0::3], controls[1::3], controls[2::3]
        )
    ]
    matched = 0
    for pair in melody_pairs:
        if generated_pairs[pair]:
            matched += 1
            generated_pairs[pair] -= 1
    print(f"  clipped melody notes present: {matched}/{len(melody_pairs)}")
    return combined


def _generated_group_count(
    events: list[int],
    melody_notes: list[tuple[float, float, int]],
    control_offset: int,
) -> int:
    """Count event groups after subtracting the melody controls."""
    controls = _melody_to_controls(melody_notes, control_offset)
    remaining = Counter(
        (time, pitch)
        for time, _duration, pitch in zip(
            events[0::3], events[1::3], events[2::3]
        )
    )
    for time, _duration, pitch in zip(
        controls[0::3], controls[1::3], controls[2::3]
    ):
        pair = (time - control_offset, pitch - control_offset)
        if remaining[pair]:
            remaining[pair] -= 1
    return sum(remaining.values())


def _layer_a_group_count(pm, melody_notes: list[tuple[float, float, int]]) -> int:
    remaining = Counter(
        (round(note.start, 3), note.pitch)
        for instrument in pm.instruments
        for note in instrument.notes
    )
    for start, _end, pitch in melody_notes:
        pair = (round(start, 3), pitch)
        if remaining[pair]:
            remaining[pair] -= 1
    return sum(remaining.values())


def _warn_if_sparse(stem: str, generated_groups: int) -> None:
    if generated_groups < 10:
        print(
            f"  WARNING: {stem} generated only {generated_groups} "
            "accompaniment event groups (<10)"
        )


def generate_base_amt(
    model,
    melody_notes: list[tuple[float, float, int]],
    control_offset: int,
    duration: float,
    top_p: float,
    seed: int,
) -> list[int]:
    """Base AMT: melody as controls, no style adapter. Generated ONCE per melody."""
    from model.generation.sampling import constrained_sampling
    from model.generation.anticipation_compat import generate_conditioned
    _assert_no_lora(model, "base_amt")
    controls = _melody_to_controls(melody_notes, control_offset)
    torch.manual_seed(seed)
    with constrained_sampling(), torch.no_grad():
        output = generate_conditioned(
            model, start_time=0, end_time=duration,
            inputs=[], controls=controls, top_p=top_p
        )
    return _combine_and_slice(output, controls, duration)


def generate_layer_a(
    melody_notes: list[tuple[float, float, int]],
    style: str,
    seed: int,
) -> "pretty_midi.PrettyMIDI":
    """Layer A scripted: rule-based, NOT a learned model."""
    from model.layer_a.generate import generate_accompaniment, notes_to_midi
    import tempfile, pretty_midi, os
    acc = generate_accompaniment(melody_notes, style, seed=seed)
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp.close()
    try:
        notes_to_midi(melody_notes, acc, style, tmp.name)
        return pretty_midi.PrettyMIDI(tmp.name)
    finally:
        os.unlink(tmp.name)


def generate_layer_c(
    model,
    melody_notes: list[tuple[float, float, int]],
    control_offset: int,
    duration: float,
    top_p: float,
    seed: int,
) -> list[int]:
    """Generate Layer C output with a style adapter loaded by the caller."""
    from model.generation.sampling import constrained_sampling
    from model.generation.anticipation_compat import generate_conditioned
    controls = _melody_to_controls(melody_notes, control_offset)
    torch.manual_seed(seed)
    with constrained_sampling(), torch.no_grad():
        output = generate_conditioned(
            model, start_time=0, end_time=duration,
            inputs=[], controls=controls, top_p=top_p
        )
    result = _combine_and_slice(output, controls, duration)
    return result


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def events_to_pm(events: list[int]):
    from model.amt_utils import events_to_pretty_midi
    return events_to_pretty_midi(events)


def save_midi(pm, path: str) -> None:
    from model.amt_utils import save_midi_file
    save_midi_file(pm, path)


def do_render_wav(pm, path: str) -> bool:
    from model.amt_utils import render_wav
    return render_wav(pm, path)


# ---------------------------------------------------------------------------
# Generation log / manifest
# ---------------------------------------------------------------------------

class GenerationLog:
    FIELDS = ["file", "layer", "melody_id", "style", "seed",
              "adapter", "duration", "generated_groups", "status", "error"]

    def __init__(self, path: str) -> None:
        self.path = path
        self.rows: list[dict] = []

    def record(self, **kwargs) -> None:
        self.rows.append({f: kwargs.get(f, "") for f in self.FIELDS})

    def save(self) -> None:
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.rows if r["status"] == "failed")

    @property
    def n_total(self) -> int:
        return len(self.rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    import anticipation.vocab as acfg
    from model.generation.adapter_paths import resolve_adapter_path

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    top_p = cfg["generation"]["top_p"]
    seed = cfg["generation"]["random_seed"]
    drive_dir = cfg["training"]["drive_dir"]
    model_name = cfg["model"]["name"]

    results_dir = Path(cfg["evaluation"]["results_dir"])
    generation_dir = Path(cfg["generation"].get(
        "output_dir", results_dir / "neural_alignment_v1"
    ))
    if generation_dir.exists() and any(generation_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite generation output directory {generation_dir}"
        )
    generation_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = generation_dir / "midi"
    wav_dir = generation_dir / "wav"
    midi_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    melody_dir = Path(cfg["data"]["output_dir"]) / "test_melodies"
    melody_files = sorted(melody_dir.glob("*.mid"))
    if not melody_files:
        print(f"No melody files found in {melody_dir}")
        sys.exit(1)
    print(f"Found {len(melody_files)} test melodies")

    control_offset = acfg.CONTROL_OFFSET
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = GenerationLog(str(generation_dir / "generation_log.csv"))
    n_wav = 0

    # ------------------------------------------------------------------
    # Pass 1: Base AMT — load model ONCE, generate for all melodies
    # ------------------------------------------------------------------
    print("\n=== Pass 1: base_amt ===")
    try:
        base_model = load_clean_base(model_name, device)
    except Exception as exc:
        import traceback
        raise RuntimeError(
            "pipeline stage base-model load failed:\n" + traceback.format_exc()
        ) from exc

    for mel_path in melody_files:
        mel_id = mel_path.stem
        melody_notes = load_melody_notes(str(mel_path))
        duration = melody_endpoint(melody_notes)
        # Save input melody separately for evaluation
        save_melody_midi(melody_notes, str(midi_dir / f"melody_{mel_id}.mid"))

        stem = f"base_amt_{mel_id}_none"
        try:
            events = generate_base_amt(
                base_model, melody_notes, control_offset, duration, top_p, seed
            )
            pm = events_to_pm(events)
            save_midi(pm, str(midi_dir / f"{stem}.mid"))
            if do_render_wav(pm, str(wav_dir / f"{stem}.wav")):
                n_wav += 1
            generated_groups = _generated_group_count(
                events, melody_notes, control_offset
            )
            _warn_if_sparse(stem, generated_groups)
            log.record(file=stem, layer="base_amt", melody_id=mel_id,
                       style="none", seed=seed, adapter="none",
                       duration=duration, status="ok",
                       generated_groups=generated_groups)
            print(f"  {stem}: ok")
        except Exception:
            import traceback
            error = "base_amt generation:\n" + traceback.format_exc()
            log.record(file=stem, layer="base_amt", melody_id=mel_id,
                       style="none", seed=seed, adapter="none",
                       duration=duration, status="failed", error=error)
            print(f"  {stem}: FAILED\n{error}")

    del base_model  # discard — never reuse after wrapping

    # ------------------------------------------------------------------
    # Pass 2: Layer A scripted
    # ------------------------------------------------------------------
    print("\n=== Pass 2: layer_a_scripted ===")
    for mel_path in melody_files:
        mel_id = mel_path.stem
        melody_notes = load_melody_notes(str(mel_path))
        duration = melody_endpoint(melody_notes)
        for style in STYLES:
            stem = f"layer_a_scripted_{mel_id}_{style}"
            try:
                pm = generate_layer_a(melody_notes, style, seed)
                save_midi(pm, str(midi_dir / f"{stem}.mid"))
                if do_render_wav(pm, str(wav_dir / f"{stem}.wav")):
                    n_wav += 1
                generated_groups = _layer_a_group_count(pm, melody_notes)
                _warn_if_sparse(stem, generated_groups)
                log.record(file=stem, layer="layer_a_scripted", melody_id=mel_id,
                           style=style, seed=seed, adapter="none",
                           duration=duration, status="ok",
                           generated_groups=generated_groups)
                print(f"  {stem}: ok")
            except Exception:
                import traceback
                error = "Layer A generation:\n" + traceback.format_exc()
                log.record(file=stem, layer="layer_a_scripted", melody_id=mel_id,
                           style=style, seed=seed, adapter="none",
                           duration=duration, status="failed", error=error)
                print(f"  {stem}: FAILED\n{error}")

    # ------------------------------------------------------------------
    # Pass 3: Layer C LoRA — fresh base per adapter, all melodies per adapter
    # ------------------------------------------------------------------
    print("\n=== Pass 3: layer_c_lora ===")
    for style in STYLES:
        import gc

        experiment_root = Path(
            cfg["training"].get("experiment_dir")
            or Path(drive_dir) / "experiments" / cfg["training"].get(
                "experiment_name", "neural_alignment_v1"
            )
        )
        adapter_path = str(resolve_adapter_path(drive_dir, experiment_root, style))
        if not Path(adapter_path).exists():
            print(f"  adapter_{style}: NOT FOUND at {adapter_path} — skipping")
            for mel_path in melody_files:
                log.record(file=f"layer_c_lora_{mel_path.stem}_{style}",
                           layer="layer_c_lora", melody_id=mel_path.stem,
                           style=style, seed=seed, adapter=adapter_path,
                           duration=duration, status="skipped",
                           error="adapter not found")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        fresh_base = None
        layer_c_model = None
        try:
            from peft import PeftModel
            fresh_base = load_clean_base(model_name, device)
            layer_c_model = PeftModel.from_pretrained(fresh_base, adapter_path)
            layer_c_model.eval()
        except Exception:
            import traceback
            error = "style adapter loading:\n" + traceback.format_exc()
            for mel_path in melody_files:
                log.record(
                    file=f"layer_c_lora_{mel_path.stem}_{style}",
                    layer="layer_c_lora",
                    melody_id=mel_path.stem,
                    style=style,
                    seed=seed,
                    adapter=adapter_path,
                    duration=0.0,
                    status="failed",
                    error=error,
                )
            print(f"  adapter_{style}: FAILED to load\n{error}")
        else:
            for mel_path in melody_files:
                mel_id = mel_path.stem
                melody_notes = load_melody_notes(str(mel_path))
                duration = melody_endpoint(melody_notes)
                stem = f"layer_c_lora_{mel_id}_{style}"
                try:
                    events = generate_layer_c(
                        layer_c_model, melody_notes, control_offset,
                        duration, top_p, seed
                    )
                    pm = events_to_pm(events)
                    save_midi(pm, str(midi_dir / f"{stem}.mid"))
                    if do_render_wav(pm, str(wav_dir / f"{stem}.wav")):
                        n_wav += 1
                    generated_groups = _generated_group_count(
                        events, melody_notes, control_offset
                    )
                    _warn_if_sparse(stem, generated_groups)
                    log.record(file=stem, layer="layer_c_lora", melody_id=mel_id,
                               style=style, seed=seed, adapter=adapter_path,
                               duration=duration, status="ok",
                               generated_groups=generated_groups)
                    print(f"  {stem}: ok")
                except Exception:
                    import traceback
                    error = "Layer C generation/export:\n" + traceback.format_exc()
                    log.record(file=stem, layer="layer_c_lora", melody_id=mel_id,
                               style=style, seed=seed, adapter=adapter_path,
                               duration=duration, status="failed", error=error)
                    print(f"  {stem}: FAILED\n{error}")
        finally:
            del layer_c_model, fresh_base
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.save()
    print(f"\nGeneration complete.")
    print(f"  Total: {log.n_total}  Failed: {log.n_failed}  WAVs: {n_wav}")
    print(f"  Log: {results_dir}/generation_log.csv")

    fail_rate = log.n_failed / max(log.n_total, 1)
    if fail_rate > 0.20:
        print(f"ERROR: failure rate {fail_rate:.1%} > 20% threshold.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
