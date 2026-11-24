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
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

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
    import torch

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
    """Clip accompaniment and verify merged melody at 25/50 ms onset/duration tolerances."""
    from anticipation import ops
    from anticipation import vocab
    from anticipation.config import TIME_RESOLUTION

    clipped_events = ops.clip(generated_events, 0, duration, seconds=True)
    combined = ops.combine(clipped_events, controls)
    time_tolerance = max(1, round(0.025 * TIME_RESOLUTION))
    duration_tolerance = max(1, round(0.05 * TIME_RESOLUTION))
    candidates = [
        tuple(combined[index:index + 3])
        for index in range(0, len(combined), 3)
    ]
    unmatched = set(range(len(candidates)))
    matched = 0
    for index in range(0, len(controls), 3):
        expected = tuple(
            token - vocab.CONTROL_OFFSET for token in controls[index:index + 3]
        )
        matches = [
            candidate_index for candidate_index in unmatched
            if candidates[candidate_index][2] == expected[2]
            and abs(candidates[candidate_index][0] - expected[0]) <= time_tolerance
            and abs(candidates[candidate_index][1] - expected[1])
            <= duration_tolerance
        ]
        if matches:
            best = min(
                matches,
                key=lambda candidate_index: (
                    abs(candidates[candidate_index][0] - expected[0])
                    + abs(candidates[candidate_index][1] - expected[1]),
                    candidate_index,
                ),
            )
            unmatched.remove(best)
            matched += 1
    print(f"  melody notes present after merge: {matched}/{len(controls) // 3}")
    if matched != len(controls) // 3:
        raise RuntimeError(
            f"merged stream contains {matched}/{len(controls) // 3} melody notes "
            f"within {time_tolerance} onset ticks and {duration_tolerance} "
            "duration ticks"
        )
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
    max_events: int = 100_000,
) -> list[int]:
    """Base AMT: melody as controls, no style adapter. Generated ONCE per melody."""
    from model.generation.sampling import constrained_sampling
    from model.generation.anticipation_compat import generate_conditioned
    import torch
    _assert_no_lora(model, "base_amt")
    controls = _melody_to_controls(melody_notes, control_offset)
    torch.manual_seed(seed)
    with constrained_sampling(), torch.no_grad():
        output = generate_conditioned(
            model, start_time=0, end_time=duration,
            inputs=[], controls=controls, top_p=top_p, max_events=max_events
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
    max_events: int = 100_000,
) -> list[int]:
    """Generate Layer C output with a style adapter loaded by the caller."""
    from model.generation.sampling import constrained_sampling
    from model.generation.anticipation_compat import generate_conditioned
    import torch
    controls = _melody_to_controls(melody_notes, control_offset)
    torch.manual_seed(seed)
    with constrained_sampling(), torch.no_grad():
        output = generate_conditioned(
            model, start_time=0, end_time=duration,
            inputs=[], controls=controls, top_p=top_p, max_events=max_events
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


def do_render_wav(pm, path: str, rendering: dict | None = None):
    from model.amt_utils import render_wav
    rendering = rendering or {}
    return render_wav(
        pm,
        path,
        sr=int(rendering.get("sample_rate", 44100)),
        soundfont_path=rendering.get("soundfont_path"),
        backend=rendering.get("backend", "fluidsynth"),
    )


# ---------------------------------------------------------------------------
# Generation log / manifest
# ---------------------------------------------------------------------------

class GenerationLog:
    FIELDS = ["file", "layer", "melody_id", "style", "seed",
              "adapter", "duration", "generated_groups", "status",
              "audio_status", "failure_stage", "render_backend",
              "soundfont", "balance", "error"]

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
    def n_audio_failed(self) -> int:
        return sum(1 for r in self.rows if r["audio_status"] == "failed")

    @property
    def n_total(self) -> int:
        return len(self.rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    import torch
    import anticipation.vocab as acfg
    from model.generation.adapter_paths import resolve_adapter_path

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    top_p = cfg["generation"]["top_p"]
    seed = cfg["generation"]["random_seed"]
    max_events = cfg["generation"].get("max_events", 100_000)
    rendering = cfg.get("rendering", {})
    balance_enabled = bool(rendering.get("balance_melody_accompaniment", False))
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
        stage = "sampling"
        audio_status = "not_attempted"
        render_result = None
        balance_info = {"enabled": False}
        error = ""
        try:
            events = generate_base_amt(
                base_model, melody_notes, control_offset, duration, top_p, seed,
                max_events=max_events,
            )
            stage = "token conversion"
            pm = events_to_pm(events)
            stage = "MIDI write"
            save_midi(pm, str(midi_dir / f"{stem}.mid"))
            stage = "rendering"
            render_result = do_render_wav(
                pm, str(wav_dir / f"{stem}.wav"), rendering
            )
            audio_status = "ok" if render_result.success else "failed"
            if render_result.success:
                n_wav += 1
            else:
                error = render_result.error
            if balance_enabled:
                stage = "balancing"
                from model.amt_utils import balance_melody_accompaniment
                balanced_pm, balance_info = balance_melody_accompaniment(
                    pm,
                    melody_notes,
                    melody_velocity=int(rendering.get("melody_velocity", 80)),
                    accompaniment_gain=float(
                        rendering.get("accompaniment_gain", 0.55)
                    ),
                    onset_tolerance_sec=float(
                        rendering.get("onset_tolerance_sec", 0.03)
                    ),
                    duration_tolerance_sec=float(
                        rendering.get("duration_tolerance_sec", 0.05)
                    ),
                )
                stage = "rendering"
                balanced_result = do_render_wav(
                    balanced_pm,
                    str(wav_dir / f"{stem}_balanced.wav"),
                    rendering,
                )
                balance_info["render"] = balanced_result.as_dict()
                if balanced_result.success:
                    n_wav += 1
                else:
                    audio_status = "failed"
                    error = "; ".join(filter(None, (error, balanced_result.error)))
            stage = "metric counting"
            generated_groups = _generated_group_count(
                events, melody_notes, control_offset
            )
            _warn_if_sparse(stem, generated_groups)
            log.record(file=stem, layer="base_amt", melody_id=mel_id,
                       style="none", seed=seed, adapter="none",
                       duration=duration,
                       status="ok" if audio_status == "ok" else "partial",
                       audio_status=audio_status,
                       failure_stage="rendering" if audio_status == "failed" else "",
                       render_backend=render_result.backend,
                       soundfont=render_result.soundfont,
                       balance=json.dumps(balance_info, sort_keys=True),
                       error=error,
                       generated_groups=generated_groups)
            print(
                f"  {stem}: MIDI ready, audio {audio_status}"
            )
        except Exception:
            import traceback
            error = f"{stage}:\n" + traceback.format_exc()
            log.record(file=stem, layer="base_amt", melody_id=mel_id,
                       style="none", seed=seed, adapter="none",
                       duration=duration, status="failed",
                       audio_status=audio_status, failure_stage=stage,
                       render_backend=getattr(render_result, "backend", ""),
                       soundfont=getattr(render_result, "soundfont", ""),
                       balance=json.dumps(balance_info, sort_keys=True),
                       error=error)
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
            stage = "sampling"
            audio_status = "not_attempted"
            render_result = None
            balance_info = {"enabled": False}
            error = ""
            try:
                pm = generate_layer_a(melody_notes, style, seed)
                stage = "MIDI write"
                save_midi(pm, str(midi_dir / f"{stem}.mid"))
                stage = "rendering"
                render_result = do_render_wav(
                    pm, str(wav_dir / f"{stem}.wav"), rendering
                )
                audio_status = "ok" if render_result.success else "failed"
                if render_result.success:
                    n_wav += 1
                else:
                    error = render_result.error
                if balance_enabled:
                    stage = "balancing"
                    from model.amt_utils import balance_melody_accompaniment
                    balanced_pm, balance_info = balance_melody_accompaniment(
                        pm,
                        melody_notes,
                        melody_velocity=int(rendering.get("melody_velocity", 80)),
                        accompaniment_gain=float(
                            rendering.get("accompaniment_gain", 0.55)
                        ),
                        onset_tolerance_sec=float(
                            rendering.get("onset_tolerance_sec", 0.03)
                        ),
                        duration_tolerance_sec=float(
                            rendering.get("duration_tolerance_sec", 0.05)
                        ),
                    )
                    stage = "rendering"
                    balanced_result = do_render_wav(
                        balanced_pm,
                        str(wav_dir / f"{stem}_balanced.wav"),
                        rendering,
                    )
                    balance_info["render"] = balanced_result.as_dict()
                    if balanced_result.success:
                        n_wav += 1
                    else:
                        audio_status = "failed"
                        error = "; ".join(filter(None, (error, balanced_result.error)))
                stage = "metric counting"
                generated_groups = _layer_a_group_count(pm, melody_notes)
                _warn_if_sparse(stem, generated_groups)
                log.record(file=stem, layer="layer_a_scripted", melody_id=mel_id,
                           style=style, seed=seed, adapter="none",
                           duration=duration,
                           status="ok" if audio_status == "ok" else "partial",
                           audio_status=audio_status,
                           failure_stage="rendering" if audio_status == "failed" else "",
                           render_backend=render_result.backend,
                           soundfont=render_result.soundfont,
                           balance=json.dumps(balance_info, sort_keys=True),
                           error=error,
                           generated_groups=generated_groups)
                print(f"  {stem}: MIDI ready, audio {audio_status}")
            except Exception:
                import traceback
                error = f"{stage}:\n" + traceback.format_exc()
                log.record(file=stem, layer="layer_a_scripted", melody_id=mel_id,
                           style=style, seed=seed, adapter="none",
                           duration=duration, status="failed",
                           audio_status=audio_status, failure_stage=stage,
                           render_backend=getattr(render_result, "backend", ""),
                           soundfont=getattr(render_result, "soundfont", ""),
                           balance=json.dumps(balance_info, sort_keys=True),
                           error=error)
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
                melody_duration = melody_endpoint(
                    load_melody_notes(str(mel_path))
                )
                log.record(file=f"layer_c_lora_{mel_path.stem}_{style}",
                           layer="layer_c_lora", melody_id=mel_path.stem,
                           style=style, seed=seed, adapter=adapter_path,
                           duration=melody_duration, status="skipped",
                           failure_stage="adapter loading",
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
                    failure_stage="adapter loading",
                    error=error,
                )
            print(f"  adapter_{style}: FAILED to load\n{error}")
        else:
            for mel_path in melody_files:
                mel_id = mel_path.stem
                melody_notes = load_melody_notes(str(mel_path))
                duration = melody_endpoint(melody_notes)
                stem = f"layer_c_lora_{mel_id}_{style}"
                stage = "sampling"
                audio_status = "not_attempted"
                render_result = None
                balance_info = {"enabled": False}
                error = ""
                try:
                    events = generate_layer_c(
                        layer_c_model, melody_notes, control_offset,
                        duration, top_p, seed, max_events=max_events
                    )
                    stage = "token conversion"
                    pm = events_to_pm(events)
                    stage = "MIDI write"
                    save_midi(pm, str(midi_dir / f"{stem}.mid"))
                    stage = "rendering"
                    render_result = do_render_wav(
                        pm, str(wav_dir / f"{stem}.wav"), rendering
                    )
                    audio_status = "ok" if render_result.success else "failed"
                    if render_result.success:
                        n_wav += 1
                    else:
                        error = render_result.error
                    if balance_enabled:
                        stage = "balancing"
                        from model.amt_utils import balance_melody_accompaniment
                        balanced_pm, balance_info = balance_melody_accompaniment(
                            pm,
                            melody_notes,
                            melody_velocity=int(rendering.get("melody_velocity", 80)),
                            accompaniment_gain=float(
                                rendering.get("accompaniment_gain", 0.55)
                            ),
                            onset_tolerance_sec=float(
                                rendering.get("onset_tolerance_sec", 0.03)
                            ),
                            duration_tolerance_sec=float(
                                rendering.get("duration_tolerance_sec", 0.05)
                            ),
                        )
                        stage = "rendering"
                        balanced_result = do_render_wav(
                            balanced_pm,
                            str(wav_dir / f"{stem}_balanced.wav"),
                            rendering,
                        )
                        balance_info["render"] = balanced_result.as_dict()
                        if balanced_result.success:
                            n_wav += 1
                        else:
                            audio_status = "failed"
                            error = "; ".join(
                                filter(None, (error, balanced_result.error))
                            )
                    stage = "metric counting"
                    generated_groups = _generated_group_count(
                        events, melody_notes, control_offset
                    )
                    _warn_if_sparse(stem, generated_groups)
                    log.record(file=stem, layer="layer_c_lora", melody_id=mel_id,
                               style=style, seed=seed, adapter=adapter_path,
                               duration=duration,
                               status="ok" if audio_status == "ok" else "partial",
                               audio_status=audio_status,
                               failure_stage="rendering" if audio_status == "failed" else "",
                               render_backend=render_result.backend,
                               soundfont=render_result.soundfont,
                               balance=json.dumps(balance_info, sort_keys=True),
                               error=error,
                               generated_groups=generated_groups)
                    print(f"  {stem}: MIDI ready, audio {audio_status}")
                except Exception:
                    import traceback
                    error = f"{stage}:\n" + traceback.format_exc()
                    log.record(file=stem, layer="layer_c_lora", melody_id=mel_id,
                               style=style, seed=seed, adapter=adapter_path,
                               duration=duration, status="failed",
                               audio_status=audio_status, failure_stage=stage,
                               render_backend=getattr(render_result, "backend", ""),
                               soundfont=getattr(render_result, "soundfont", ""),
                               balance=json.dumps(balance_info, sort_keys=True),
                               error=error)
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
    print(
        f"  Total: {log.n_total}  Failed MIDI/metrics: {log.n_failed}  "
        f"Failed audio renders: {log.n_audio_failed}  WAVs: {n_wav}"
    )
    print(f"  Log: {log.path}")

    fail_rate = log.n_failed / max(log.n_total, 1)
    if fail_rate > 0.20:
        print(f"ERROR: failure rate {fail_rate:.1%} > 20% threshold.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
