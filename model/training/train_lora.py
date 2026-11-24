"""
train_lora.py

Fine-tunes LoRA adapters (adapter_baroque, adapter_romantic) on top of the
frozen Anticipatory Music Transformer (stanford-crfm/music-medium-800k).

The base model was pretrained on Lakh MIDI (NOT MAESTRO).
Fine-tuning uses cleaned MAESTRO windows (Baroque: Bach/Scarlatti/Handel;
Romantic: Chopin/Liszt).

Run on Colab GPU via colab/05_train_lora.ipynb.
Do NOT run locally — requires GPU and the anticipation library.

Provenance: Layer C (learned).

Anticipation compatibility is pinned and tested against upstream commit
af37397922665a0fb8d474d7988b0f3755a38d45. Real Colab/model execution remains
UNVERIFIED until run in the target environment.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, pickle, random, tempfile, time
from pathlib import Path

import yaml

from model.training.sequence import (
    CACHE_SCHEMA,
    INVALID_WINDOW_POLICY,
    SEQUENCE_POLICY,
    ContextOverflowError,
    SequenceSettings,
    UnconsumedControlsError,
    chunk_sequence,
    example_counts,
    interleave_anticipated_controls,
    pad_event_groups,
    token_weighted_loss_weights,
)
from model.training.experiment import new_experiment_root, prepare_new_adapter_dir


# ---------------------------------------------------------------------------
# Data loading and tokenisation
# ---------------------------------------------------------------------------

def load_windows(manifest_path: str, style: str, split: str) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(manifest_path)
    rows = df[(df["style"] == style) & (df["split"] == split)].to_dict("records")
    if not rows:
        raise ValueError(
            f"No windows found for style={style!r} split={split!r} in {manifest_path}. "
            f"Available splits: {df['split'].unique().tolist()}"
        )
    return rows


def _build_anticipatory_sequence(
    events: list[int],
    mel_events: list[int],
    control_offset: int,
) -> tuple[list[int], list[int]]:
    """
    Compatibility helper for Anticipation's upstream anticipate ordering.

    anticipation.ops.anticipate returns (interleaved_tokens,
    unconsumed_controls). Controls are masked from the training loss.

    Production window tokenisation uses the same predecessor-event criterion
    as this compatibility helper because the sampler's current time is the
    preceding generated event before its next model call.
    """
    settings = _sequence_settings(control_offset)
    interleaved, remaining = interleave_anticipated_controls(
        events, [t + control_offset for t in mel_events], settings
    )
    if remaining:
        raise UnconsumedControlsError(len(remaining) // 3)
    labels = [
        -100 if token >= control_offset else token
        for token in interleaved
    ]
    return list(interleaved), labels


def _tokenise_window(
    row: dict,
    control_offset: int,
    seq_len: int,
    anticipatory_prob: float,
    rng,
    shift: int = 0,
    performance_notes: list[tuple[float, float, int]] | None = None,
    mode: str = "mixed",
    context_groups: int = 170,
) -> list[tuple[list[int], list[int], list[int]]]:
    """Tokenise a window into context-aligned examples with explicit targets."""
    examples, _ = _tokenise_window_with_stats(
        row, control_offset, seq_len, anticipatory_prob, rng, shift,
        performance_notes, mode, context_groups,
    )
    return examples


def _sequence_settings(control_offset: int) -> SequenceSettings:
    from anticipation import config as acfg
    from anticipation import vocab

    return SequenceSettings(
        control_offset=control_offset,
        rest_token=vocab.REST,
        time_offset=vocab.TIME_OFFSET,
        duration_offset=vocab.DUR_OFFSET,
        anticipate_time_offset=vocab.ATIME_OFFSET,
        anticipate_duration_offset=vocab.ADUR_OFFSET,
        time_resolution=acfg.TIME_RESOLUTION,
        anticipation_delta=acfg.DELTA * acfg.TIME_RESOLUTION,
        padding_density=acfg.TIME_RESOLUTION,
    )


def _events_for_notes(notes) -> list[int]:
    from anticipation.convert import midi_to_events
    from model.amt_utils import write_notes_to_midi

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
        path = handle.name
    try:
        write_notes_to_midi(notes, path)
        return list(midi_to_events(path))
    finally:
        os.unlink(path)


def _tokenise_window_with_stats(
    row: dict,
    control_offset: int,
    seq_len: int,
    anticipatory_prob: float,
    rng,
    shift: int = 0,
    performance_notes: list[tuple[float, float, int]] | None = None,
    mode: str = "mixed",
    context_groups: int = 170,
) -> tuple[list[tuple[list[int], list[int], list[int]]], dict]:
    """Tokenise targets with controls visible at the same sampler time as inference."""
    from anticipation.vocab import ANTICIPATE, AUTOREGRESS
    from model.data.skyline import skyline, Note as SNote

    if mode not in ("mixed", "unconditional", "conditioned"):
        raise ValueError(f"unknown tokenisation mode: {mode}")
    start, end = float(row["start_sec"]), float(row["end_sec"])
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"invalid window interval [{start}, {end}]")
    if performance_notes is None:
        performance_notes = _load_performance_notes(row["midi_path"])
    all_notes = [
        SNote(
            note_start - start,
            note_end - start,
            max(0, min(127, pitch + shift)),
        )
        for note_start, note_end, pitch in performance_notes
        if note_start >= start and note_end <= end
    ]
    selected_mode = (
        "conditioned" if rng.random() < anticipatory_prob else "unconditional"
    ) if mode == "mixed" else mode
    settings = _sequence_settings(control_offset)
    end_time = round((end - start) * settings.time_resolution)
    stats = {
        "mode": selected_mode,
        "source_note_count": len(all_notes),
        "original_time_coverage_sec": end - start,
        "retained_time_start_sec": min((note.start for note in all_notes), default=None),
        "retained_time_end_sec": max((note.end for note in all_notes), default=None),
        "control_groups": 0,
        "unconsumed_controls": 0,
        "dropped_controls": 0,
        "chunked": False,
        "invalid": False,
        "empty": False,
    }

    if selected_mode == "conditioned":
        melody = skyline(all_notes)
        melody_keys = {(round(note.start, 3), note.pitch) for note in melody}
        accompaniment = [
            note for note in all_notes
            if (round(note.start, 3), note.pitch) not in melody_keys
        ]
        target_events = _events_for_notes(accompaniment)
        control_events = _events_for_notes(melody)
        control_events = [token + control_offset for token in control_events]
        mode_token = ANTICIPATE
    else:
        target_events = _events_for_notes(all_notes)
        control_events = []
        mode_token = AUTOREGRESS

    padded_events = pad_event_groups(target_events, end_time, settings)
    padded_rest_groups = sum(
        1 for note in padded_events[2::3] if note == settings.rest_token
    )
    interleaved, remaining = interleave_anticipated_controls(
        padded_events, control_events, settings
    )
    stats["control_groups"] = len(control_events) // 3
    stats["unconsumed_controls"] = len(remaining) // 3
    if remaining:
        raise UnconsumedControlsError(len(remaining) // 3)
    examples = chunk_sequence(
        interleaved, mode_token, seq_len, settings, context_groups
    )
    stats.update(example_counts(examples, settings))
    stats["padded_rest_groups"] = padded_rest_groups
    stats["chunked"] = len(examples) > 1
    stats["empty"] = not examples
    return examples, stats


def _load_performance_notes(midi_path: str) -> list[tuple[float, float, int]]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    return [
        (note.start, note.end, note.pitch)
        for instrument in pm.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]


def _cache_path(
    cache_dir: Path,
    style: str,
    split: str,
    seed: int,
    windows: list[dict],
    control_offset: int,
    seq_len: int,
    mode: str = "mixed",
    anticipatory_prob: float = 0.5,
    context_groups: int = 170,
    source_fingerprints: dict[str, str] | None = None,
    invalid_window_policy: str = INVALID_WINDOW_POLICY,
) -> Path:
    if source_fingerprints is None:
        source_fingerprints = _source_fingerprints(windows)
    signature = [
        (
            row["midi_path"],
            float(row["start_sec"]),
            float(row["end_sec"]),
            row.get("window_id", ""),
            source_fingerprints[str(row["midi_path"])],
        )
        for row in windows
    ]
    settings = {
        "schema": CACHE_SCHEMA,
        "policy": SEQUENCE_POLICY,
        "timing": _sequence_settings(control_offset).__dict__,
        "window_selection": "notes-fully-contained-in-manifest-window-v1",
        "pitch_shift": "seeded-training-only-uniform-integer-minus3-to-plus3",
        "style": style,
        "split": split,
        "seed": seed,
        "control_offset": control_offset,
        "seq_len": seq_len,
        "mode": mode,
        "anticipatory_prob": anticipatory_prob,
        "context_groups": context_groups,
        "invalid_window_policy": invalid_window_policy,
        "window_signature": signature,
    }
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return cache_dir / f"{style}_{split}_{digest}_{CACHE_SCHEMA}.pkl"


def _source_fingerprints(windows: list[dict]) -> dict[str, str]:
    """Hash each distinct source MIDI once, so same-path edits invalidate caches."""
    fingerprints = {}
    for row in windows:
        path = str(row["midi_path"])
        if path in fingerprints:
            continue
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            fingerprints[path] = f"unreadable:{type(exc).__name__}:{exc}"
        else:
            fingerprints[path] = digest.hexdigest()
    return fingerprints


class PreprocessingRejectedError(ValueError):
    """Raised after a diagnostic report records rejected dataset windows."""


def _write_preprocessing_report(path: Path, diagnostics: dict) -> None:
    path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_cache(
    windows: list[dict],
    style: str,
    split: str,
    control_offset: int,
    seq_len: int,
    cache_dir: Path,
    seed: int,
    performance_cache: dict[str, list[tuple[float, float, int]]] | None = None,
    mode: str = "mixed",
    anticipatory_prob: float = 0.5,
    context_groups: int = 170,
    invalid_window_policy: str = INVALID_WINDOW_POLICY,
) -> dict:
    """Cache examples without mutating or reusing an incompatible old schema."""
    if invalid_window_policy != INVALID_WINDOW_POLICY:
        raise ValueError(
            f"unsupported invalid_window_policy {invalid_window_policy!r}; "
            f"only {INVALID_WINDOW_POLICY!r} is allowed"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_fingerprints = _source_fingerprints(windows)
    cp = _cache_path(
        cache_dir, style, split, seed, windows, control_offset, seq_len,
        mode, anticipatory_prob, context_groups,
        source_fingerprints=source_fingerprints,
        invalid_window_policy=invalid_window_policy,
    )
    report_path = cp.with_suffix(".diagnostics.json")
    if cp.exists():
        with cp.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"incompatible cache schema in {cp}")
        diagnostics = cached["diagnostics"]
        if (
            diagnostics.get("invalid_windows", 0)
            or diagnostics.get("empty_windows", 0)
            or diagnostics.get("all_masked_examples", 0)
            or cached.get("settings", {}).get("invalid_window_policy")
            != invalid_window_policy
        ):
            _write_preprocessing_report(report_path, diagnostics)
            raise PreprocessingRejectedError(
                f"cached preprocessing rejected windows; report: {report_path}"
            )
        return cached["diagnostics"]
    rng = random.Random(seed)
    shuffled_windows = list(windows)
    if split == "train":
        rng.shuffle(shuffled_windows)
    else:
        shuffled_windows.sort(key=lambda row: (
            str(row.get("window_id", "")), str(row["midi_path"]),
            (
                os.stat(row["midi_path"]).st_size,
                os.stat(row["midi_path"]).st_mtime_ns,
            ) if os.path.isfile(row["midi_path"]) else None,
            float(row["start_sec"]),
        ))
    performance_cache = performance_cache if performance_cache is not None else {}
    print(f"  Caching {style}/{split} ({len(shuffled_windows)} windows) ...")
    samples = []
    window_stats = []
    invalid_windows = []
    cache_started = time.perf_counter()
    for window_index, row in enumerate(shuffled_windows, start=1):
        stats = None
        try:
            midi_path = str(row["midi_path"])
            if midi_path not in performance_cache:
                performance_cache[midi_path] = _load_performance_notes(midi_path)
            shift = rng.randint(-3, 3) if split == "train" else 0
            row_examples, stats = _tokenise_window_with_stats(
                row, control_offset, seq_len, anticipatory_prob, rng,
                shift=shift, performance_notes=performance_cache[midi_path],
                mode=mode, context_groups=context_groups,
            )
            if stats.get("empty"):
                raise ValueError("window produced no training examples")
            if stats.get("all_masked_examples"):
                raise ValueError("window produced all-masked training examples")
            samples.extend(row_examples)
            window_stats.append(stats)
        except Exception as exc:
            if isinstance(exc, ContextOverflowError):
                category = "context_overflow"
            elif stats and stats.get("empty"):
                category = "empty"
            elif stats and stats.get("all_masked_examples"):
                category = "all_masked"
            else:
                category = "unexpected"
            invalid = {
                "window_id": row.get("window_id", ""),
                "style": style,
                "objective": mode,
                "category": category,
                "error_type": type(exc).__name__,
                "empty_windows": int(bool(stats and stats.get("empty"))),
                "all_masked_examples": int(
                    (stats or {}).get("all_masked_examples", 0)
                ),
                "reason": f"{type(exc).__name__}: {exc}",
                "error": f"{type(exc).__name__}: {exc}",
                "unconsumed_controls": (
                    exc.count if isinstance(exc, UnconsumedControlsError) else 0
                ),
            }
            invalid_windows.append(invalid)
            print(
                f"  INVALID {row.get('window_id', '?')}: "
                f"{type(exc).__name__}: {exc}"
            )
        if window_index == 20:
            elapsed = time.perf_counter() - cache_started
            estimate = elapsed * len(shuffled_windows) / window_index
            print(f"  Estimated cache time: {estimate:.1f}s "
                  f"(measured first 20 windows)")
    if len(shuffled_windows) < 20:
        elapsed = time.perf_counter() - cache_started
        print(f"  Cache time: {elapsed:.1f}s for {len(shuffled_windows)} windows "
              "(fewer than 20; no extrapolated estimate)")
    diagnostics = summarize_preprocessing(
        window_stats, invalid_windows, len(shuffled_windows), mode
    )
    diagnostics["cache_identity"] = cp.stem
    diagnostics["invalid_window_policy"] = invalid_window_policy
    diagnostics["style"] = style
    diagnostics["objective"] = mode
    _write_preprocessing_report(report_path, diagnostics)
    if diagnostics["invalid_windows"]:
        rejected_ids = [
            str(entry.get("window_id", "?")) for entry in invalid_windows
        ]
        raise PreprocessingRejectedError(
            f"preprocessing rejected {diagnostics['invalid_windows']} "
            f"{style}/{mode} window(s) {rejected_ids}; report: {report_path}"
        )
    payload = {
        "schema": CACHE_SCHEMA,
        "settings": {
            "mode": mode,
            "anticipatory_prob": anticipatory_prob,
            "context_groups": context_groups,
            "sequence_length": seq_len,
            "seed": seed,
            "invalid_window_policy": invalid_window_policy,
        },
        "diagnostics": diagnostics,
        "samples": samples,
    }
    with cp.open("wb") as handle:
        pickle.dump(payload, handle)
    print(f"  Cache complete for {style}/{split}")
    return diagnostics


def summarize_preprocessing(
    window_stats: list[dict],
    invalid_windows: list[dict],
    source_windows: int,
    mode: str,
) -> dict:
    examples = sum(int(stats.get("produced_examples", 0)) for stats in window_stats)
    failure_counts: dict[str, int] = {}
    for failure in invalid_windows:
        key = "|".join((
            str(failure.get("style", "")),
            str(failure.get("objective", mode)),
            str(failure.get("reason", "unknown")),
        ))
        failure_counts[key] = failure_counts.get(key, 0) + 1
    return {
        "mode": mode,
        "source_windows": source_windows,
        "valid_windows": len(window_stats),
        "produced_examples": examples,
        "original_time_coverage_sec": sum(
            float(stats.get("original_time_coverage_sec", 0))
            for stats in window_stats
        ),
        "retained_time_coverage_sec": sum(
            max(
                0.0,
                float(stats["retained_time_end_sec"])
                - float(stats["retained_time_start_sec"]),
            )
            for stats in window_stats
            if stats.get("retained_time_start_sec") is not None
            and stats.get("retained_time_end_sec") is not None
        ),
        "supervised_accompaniment_tokens": sum(
            int(stats.get("supervised_accompaniment_tokens", 0))
            for stats in window_stats
        ),
        "supervised_rest_groups": sum(
            int(stats.get("supervised_rest_groups", 0))
            for stats in window_stats
        ),
        "supervised_rest_tokens": sum(
            int(stats.get("supervised_rest_tokens", 0))
            for stats in window_stats
        ),
        "control_groups": sum(
            int(stats.get("control_groups", 0)) for stats in window_stats
        ),
        "unconsumed_controls": sum(
            int(stats.get("unconsumed_controls", 0)) for stats in window_stats
        ) + sum(
            int(entry.get("unconsumed_controls", 0)) for entry in invalid_windows
        ),
        "dropped_controls": sum(
            int(stats.get("dropped_controls", 0)) for stats in window_stats
        ),
        "chunked_windows": sum(bool(stats.get("chunked")) for stats in window_stats),
        "truncated_windows": sum(
            int(stats.get("truncated_groups", 0) > 0) for stats in window_stats
        ),
        "invalid_windows": len(invalid_windows),
        "failures_by_style_objective_reason": failure_counts,
        "empty_windows": sum(bool(stats.get("empty")) for stats in window_stats)
        + sum(int(entry.get("empty_windows", 0)) for entry in invalid_windows),
        "all_masked_examples": sum(
            int(stats.get("all_masked_examples", 0)) for stats in window_stats
        ) + sum(
            int(entry.get("all_masked_examples", 0)) for entry in invalid_windows
        ),
        "context_overflows": sum(
            entry.get("category") == "context_overflow"
            for entry in invalid_windows
        ),
        "invalid_examples": invalid_windows,
    }


def load_cached(
    cache_dir: Path,
    style: str,
    split: str,
    seed: int,
    windows: list[dict],
    control_offset: int,
    seq_len: int,
    mode: str = "mixed",
    anticipatory_prob: float = 0.5,
    context_groups: int = 170,
    invalid_window_policy: str = INVALID_WINDOW_POLICY,
) -> list[tuple]:
    """Load cached samples for a style/split."""
    if invalid_window_policy != INVALID_WINDOW_POLICY:
        raise ValueError(
            f"unsupported invalid_window_policy {invalid_window_policy!r}; "
            f"only {INVALID_WINDOW_POLICY!r} is allowed"
        )
    cp = _cache_path(
        cache_dir, style, split, seed, windows, control_offset, seq_len,
        mode, anticipatory_prob, context_groups,
        invalid_window_policy=invalid_window_policy,
    )
    with cp.open("rb") as handle:
        cached = pickle.load(handle)
    if cached.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"incompatible cache schema in {cp}")
    diagnostics = cached.get("diagnostics", {})
    if (
        diagnostics.get("invalid_windows", 0)
        or diagnostics.get("empty_windows", 0)
        or diagnostics.get("all_masked_examples", 0)
        or cached.get("settings", {}).get("invalid_window_policy")
        != INVALID_WINDOW_POLICY
    ):
        report_path = cp.with_suffix(".diagnostics.json")
        _write_preprocessing_report(report_path, diagnostics)
        raise PreprocessingRejectedError(
            f"cached preprocessing rejected windows; report: {report_path}"
        )
    return cached["samples"]


CACHE_SELECTION_SCHEMA = "neural-alignment-cache-selection-v1"


def _explicit_cache_training_settings(tcfg: dict) -> tuple[int, int, float, int]:
    required = {
        "random_seed", "seq_len", "context_groups", "anticipatory_prob",
        "invalid_window_policy",
    }
    missing = required - set(tcfg)
    if missing:
        raise ValueError(
            "saved training config is missing explicit cache settings: "
            f"{sorted(missing)}"
        )
    if "experiment_dir" not in tcfg and "experiment_name" not in tcfg:
        raise ValueError(
            "saved training config must specify experiment_dir or experiment_name"
        )
    if tcfg["invalid_window_policy"] != INVALID_WINDOW_POLICY:
        raise ValueError(
            f"unsupported invalid_window_policy "
            f"{tcfg['invalid_window_policy']!r}; only "
            f"{INVALID_WINDOW_POLICY!r} is allowed"
        )
    return (
        int(tcfg["random_seed"]),
        int(tcfg["seq_len"]),
        float(tcfg["anticipatory_prob"]),
        int(tcfg["context_groups"]),
    )


def _validate_cached_payload(
    cached: dict,
    cache_path: Path,
    style: str,
    split: str,
    mode: str,
    seed: int,
    seq_len: int,
    anticipatory_prob: float,
    context_groups: int,
    control_offset: int,
) -> list[tuple[list[int], list[int], list[int]]]:
    """Validate a selected cache without consulting a manifest or source MIDI."""
    expected_prefix = f"{style}_{split}_"
    expected_suffix = f"_{CACHE_SCHEMA}.pkl"
    if not cache_path.name.startswith(expected_prefix) or not cache_path.name.endswith(
        expected_suffix
    ):
        raise ValueError(
            f"cache filename does not identify {style}/{split}: {cache_path}"
        )
    if cached.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"incompatible cache schema in {cache_path}")

    settings = cached.get("settings", {})
    expected_settings = {
        "mode": mode,
        "anticipatory_prob": anticipatory_prob,
        "context_groups": context_groups,
        "sequence_length": seq_len,
        "seed": seed,
        "invalid_window_policy": INVALID_WINDOW_POLICY,
    }
    mismatches = {
        name: (settings.get(name), expected)
        for name, expected in expected_settings.items()
        if settings.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            f"incompatible cache settings in {cache_path}: {mismatches}"
        )

    diagnostics = cached.get("diagnostics", {})
    identity = cache_path.name[:-4]
    if (
        diagnostics.get("cache_identity") != identity
        or diagnostics.get("style") != style
        or diagnostics.get("objective") != mode
        or diagnostics.get("mode") != mode
    ):
        raise ValueError(
            f"cache identity/style/objective metadata mismatch in {cache_path}"
        )
    if (
        int(diagnostics.get("source_windows", 0)) <= 0
        or diagnostics.get("source_windows") != diagnostics.get("valid_windows")
        or int(diagnostics.get("invalid_windows", 0)) != 0
        or int(diagnostics.get("empty_windows", 0)) != 0
        or int(diagnostics.get("all_masked_examples", 0)) != 0
        or diagnostics.get("invalid_window_policy") != INVALID_WINDOW_POLICY
    ):
        raise PreprocessingRejectedError(
            f"cache contains rejected, empty, or all-masked windows: {cache_path}"
        )

    samples = cached.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"cache has no samples: {cache_path}")
    if int(diagnostics.get("produced_examples", -1)) != len(samples):
        raise ValueError(f"cache sample count disagrees with diagnostics: {cache_path}")

    supervised_tokens = 0
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, (tuple, list)) or len(sample) != 3:
            raise ValueError(f"malformed sample {sample_index} in {cache_path}")
        ids, labels, attention = sample
        if not all(
            isinstance(values, (tuple, list)) and len(values) == seq_len
            for values in (ids, labels, attention)
        ):
            raise ValueError(
                f"sample {sample_index} does not match sequence length {seq_len}"
            )
        if any(not isinstance(token, int) or token < 0 for token in ids):
            raise ValueError(f"invalid input token in sample {sample_index}")
        if any(
            not isinstance(label, int) or (label < 0 and label != -100)
            for label in labels
        ):
            raise ValueError(f"invalid label in sample {sample_index}")
        if any(mask not in (0, 1) for mask in attention):
            raise ValueError(f"invalid attention mask in sample {sample_index}")
        if any(label != -100 and not mask for label, mask in zip(labels, attention)):
            raise ValueError(
                f"sample {sample_index} supervises an attention-masked token"
            )
        if labels[0] != -100:
            raise ValueError(f"sample {sample_index} supervises its mode token")
        if any(
            label != -100 and label >= control_offset
            for label in labels
        ):
            raise ValueError(f"sample {sample_index} supervises a control token")
        if any(
            label != -100 and label != token
            for label, token in zip(labels, ids)
        ):
            raise ValueError(f"sample {sample_index} labels do not match input tokens")
        for index in range(1, seq_len - 2, 3):
            if ids[index + 2] >= control_offset and any(
                label != -100 for label in labels[index:index + 3]
            ):
                raise ValueError(
                    f"sample {sample_index} supervises an anticipation control group"
                )
        count = sum(
            label != -100 and bool(mask)
            for label, mask in zip(labels[1:], attention[1:])
        )
        if count == 0:
            raise ValueError(f"sample {sample_index} has no causal targets")
        supervised_tokens += count
    if supervised_tokens == 0:
        raise ValueError(f"cache has no supervised targets: {cache_path}")
    return samples


def load_explicit_cache(
    cache_path: str | Path,
    style: str,
    split: str,
    mode: str,
    seed: int,
    seq_len: int,
    anticipatory_prob: float,
    context_groups: int,
    control_offset: int,
) -> list[tuple[list[int], list[int], list[int]]]:
    """Load one explicitly selected cache and verify its embedded identity."""
    samples, _ = _load_explicit_cache_with_diagnostics(
        cache_path, style, split, mode, seed, seq_len, anticipatory_prob,
        context_groups, control_offset,
    )
    return samples


def _load_explicit_cache_with_diagnostics(
    cache_path: str | Path,
    style: str,
    split: str,
    mode: str,
    seed: int,
    seq_len: int,
    anticipatory_prob: float,
    context_groups: int,
    control_offset: int,
) -> tuple[
    list[tuple[list[int], list[int], list[int]]],
    dict,
]:
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"selected cache does not exist: {cache_path}")
    with cache_path.open("rb") as handle:
        cached = pickle.load(handle)
    samples = _validate_cached_payload(
        cached, cache_path, style, split, mode, seed, seq_len,
        anticipatory_prob, context_groups, control_offset,
    )
    return samples, cached["diagnostics"]


def discover_cache_selection(
    cache_dir: str | Path,
    cfg: dict,
    styles: tuple[str, ...] = ("baroque", "romantic"),
) -> dict:
    """Resolve one and only one compatible cache for each training/validation role."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"cache directory does not exist: {cache_dir}")
    tcfg = cfg["training"]
    seed, seq_len, train_probability, context_groups = (
        _explicit_cache_training_settings(tcfg)
    )
    from anticipation.vocab import CONTROL_OFFSET
    entries: dict[str, dict[str, str]] = {}
    for style in styles:
        entries[style] = {}
        roles = (
            ("train", "mixed", "train_mixed", train_probability),
            ("val", "unconditional", "val_unconditional", 0.0),
            ("val", "conditioned", "val_conditioned", 0.0),
        )
        for split, mode, role, probability in roles:
            candidates = sorted(
                path for path in cache_dir.iterdir()
                if path.is_file()
                and path.name.startswith(f"{style}_{split}_")
                and path.name.endswith(f"_{CACHE_SCHEMA}.pkl")
            )
            compatible = []
            failures = []
            for candidate in candidates:
                try:
                    load_explicit_cache(
                        candidate, style, split, mode, seed, seq_len,
                        probability, context_groups, CONTROL_OFFSET,
                    )
                except (ValueError, PreprocessingRejectedError) as exc:
                    failures.append(f"{candidate.name}: {exc}")
                else:
                    compatible.append(candidate)
            if len(compatible) != 1:
                detail = "; ".join(failures) or "no matching cache found"
                raise ValueError(
                    f"expected exactly one compatible {style}/{split}/{mode} "
                    f"cache, found {len(compatible)}; {detail}"
                )
            entries[style][role] = str(compatible[0].resolve())
    return {
        "schema": CACHE_SELECTION_SCHEMA,
        "cache_dir": str(cache_dir.resolve()),
        "entries": entries,
    }


def _load_cache_selection(path: str | Path, cache_dir: Path) -> dict:
    selection_path = Path(path)
    with selection_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("schema") != CACHE_SELECTION_SCHEMA:
        raise ValueError(f"incompatible cache-selection schema in {selection_path}")
    if Path(selection.get("cache_dir", "")).resolve() != cache_dir.resolve():
        raise ValueError(
            f"cache selection points to {selection.get('cache_dir')}, "
            f"but config selects {cache_dir}"
        )
    return selection


def load_selected_cache_set(
    selection: dict,
    cache_dir: Path,
    style: str,
    seed: int,
    seq_len: int,
    anticipatory_prob: float,
    context_groups: int,
    control_offset: int,
) -> tuple[
    list[tuple[list[int], list[int], list[int]]],
    dict[str, list[tuple[list[int], list[int], list[int]]]],
    dict[str, dict],
]:
    if selection.get("schema") != CACHE_SELECTION_SCHEMA:
        raise ValueError("incompatible explicit cache selection schema")
    if Path(selection.get("cache_dir", "")).resolve() != cache_dir.resolve():
        raise ValueError("explicit cache selection does not match configured cache dir")
    entries = selection.get("entries", {}).get(style)
    expected_roles = {"train_mixed", "val_unconditional", "val_conditioned"}
    if not isinstance(entries, dict) or set(entries) != expected_roles:
        raise ValueError(
            f"cache selection for {style} must contain exactly "
            f"{sorted(expected_roles)}"
        )
    for role, cache_file in entries.items():
        if Path(cache_file).resolve().parent != cache_dir.resolve():
            raise ValueError(
                f"selected cache {cache_file} is outside {cache_dir.resolve()}"
            )
    train_samples = load_explicit_cache(
        entries["train_mixed"], style, "train", "mixed", seed, seq_len,
        anticipatory_prob, context_groups, control_offset,
    )
    validation_samples = {}
    validation_diagnostics = {}
    for mode, role in (
        ("unconditional", "val_unconditional"),
        ("conditioned", "val_conditioned"),
    ):
        (
            validation_samples[mode],
            validation_diagnostics[mode],
        ) = _load_explicit_cache_with_diagnostics(
            entries[role], style, "val", mode,
            seed, seq_len, 0.0, context_groups, control_offset,
        )
    return train_samples, validation_samples, validation_diagnostics


def _validate_validation_cache_mappings(
    samples_by_mode: dict,
    diagnostics_by_mode: dict,
    style: str,
) -> None:
    expected_modes = {"unconditional", "conditioned"}
    if set(samples_by_mode) != expected_modes:
        raise ValueError(
            f"{style} validation samples must contain exactly "
            f"{sorted(expected_modes)}; got {sorted(samples_by_mode)}"
        )
    if set(diagnostics_by_mode) != expected_modes:
        raise ValueError(
            f"{style} validation diagnostics must contain exactly "
            f"{sorted(expected_modes)}; got {sorted(diagnostics_by_mode)}"
        )
    for mode in sorted(expected_modes):
        if not samples_by_mode[mode]:
            raise ValueError(f"{style} {mode} validation cache has no samples")
        diagnostics = diagnostics_by_mode[mode]
        if not isinstance(diagnostics, dict) or not diagnostics:
            raise ValueError(
                f"{style} {mode} validation cache has no diagnostics"
            )
        if (
            diagnostics.get("mode") != mode
            or diagnostics.get("objective") != mode
            or diagnostics.get("style") != style
        ):
            raise ValueError(
                f"{style} {mode} validation diagnostics have mismatched "
                f"identity: mode={diagnostics.get('mode')!r}, "
                f"objective={diagnostics.get('objective')!r}, "
                f"style={diagnostics.get('style')!r}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_base_model(model_name: str, device: str):
    """Load the base model in fp16. Falls back to GPT2LMHeadModel if needed."""
    import torch
    import traceback

    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        print("Loaded with AutoModelForCausalLM")
        return model
    except Exception:
        auto_error = traceback.format_exc()
        print("AutoModelForCausalLM failed; trying GPT2LMHeadModel:\n" + auto_error)
        from transformers import GPT2LMHeadModel
        try:
            model = GPT2LMHeadModel.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(device)
        except Exception as fallback_error:
            raise RuntimeError(
                "training base-model load failed in both supported loaders.\n"
                f"AutoModelForCausalLM traceback:\n{auto_error}"
            ) from fallback_error
        print("Loaded with GPT2LMHeadModel")
        return model


def _verify_lora_targets(model, target_modules: list[str]) -> None:
    """Assert that all target_modules exist in the model."""
    module_names = {name for name, _ in model.named_modules()}
    for target in target_modules:
        found = any(target in name for name in module_names)
        if not found:
            print(f"  WARNING: LoRA target '{target}' not found in model modules.")
            print(f"  Available modules (sample): {list(module_names)[:20]}")
        else:
            print(f"  LoRA target '{target}': found ok")


def _assert_trainable_parameters(model) -> None:
    """Require a frozen base and fp32 trainable LoRA weights."""
    import torch

    trainable = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" not in name:
            raise RuntimeError(f"non-LoRA base parameter is trainable: {name}")
        if param.dtype != torch.float32:
            raise RuntimeError(
                f"trainable LoRA parameter {name} is {param.dtype}, expected float32"
            )
        trainable.append(name)
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters were found")
    print(f"  Frozen base; {len(trainable)} trainable fp32 LoRA tensors")


MAX_AMP_UPDATE_RETRIES = 3


def _require_finite_loss(loss: float, scale: float, step: int) -> None:
    if not math.isfinite(loss):
        raise RuntimeError(
            f"optimizer step {step} has non-finite forward loss {loss}; "
            f"current AMP scale={scale}. No backward or optimizer update was run."
        )


def _training_parameters_changed(before, after, torch) -> bool:
    if len(before) != len(after):
        raise ValueError("trainable parameter count changed during optimizer update")
    return any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, after)
    )


def _smoke_test_lr_lambda(lr_lambda, smoke_test: bool):
    offset = int(smoke_test)
    return lambda index: lr_lambda(index + offset)


def _inspect_and_clip_gradients(
    named_parameters,
    torch,
    scale: float,
    step: int,
    max_norm: float = 1.0,
) -> tuple[float | None, list[str]]:
    parameters_with_grad = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    nonfinite_names = [
        name for name, parameter in parameters_with_grad
        if not bool(torch.isfinite(parameter.grad).all().item())
    ]
    if nonfinite_names:
        return None, nonfinite_names
    if not parameters_with_grad:
        raise RuntimeError(
            f"optimizer step {step} has no gradients; current AMP scale={scale}"
        )

    squared_norm = 0.0
    for name, parameter in parameters_with_grad:
        parameter_norm = float(
            torch.linalg.vector_norm(
                parameter.grad.detach().to(dtype=torch.float64), ord=2
            ).item()
        )
        if not math.isfinite(parameter_norm):
            names = [entry_name for entry_name, _ in parameters_with_grad]
            raise RuntimeError(
                f"optimizer step {step} has a non-finite gradient norm reduction "
                "despite finite gradient elements; "
                f"AMP scale={scale}, contributing parameters={names}"
            )
        squared_norm += parameter_norm * parameter_norm
        if not math.isfinite(squared_norm):
            names = [entry_name for entry_name, _ in parameters_with_grad]
            raise RuntimeError(
                f"optimizer step {step} has a non-finite global gradient norm "
                "reduction despite finite gradient elements; "
                f"AMP scale={scale}, contributing parameters={names}"
            )
    gradient_norm = math.sqrt(squared_norm)
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        names = [name for name, _ in parameters_with_grad]
        raise RuntimeError(
            f"optimizer step {step} has invalid gradient norm {gradient_norm}; "
            f"AMP scale={scale}, contributing parameters={names}"
        )
    clip_coefficient = min(1.0, max_norm / gradient_norm)
    if clip_coefficient < 1.0:
        for _, parameter in parameters_with_grad:
            parameter.grad.mul_(clip_coefficient)
    return gradient_norm, []


def _run_optimizer_update_with_retries(
    attempt,
    scaler,
    optimizer,
    scheduler,
    step: int,
    max_retries: int = MAX_AMP_UPDATE_RETRIES,
) -> tuple[float, int]:
    """Run one accumulated update, retrying only GradScaler-skipped attempts."""
    for retry in range(max_retries + 1):
        scale_before = float(scaler.get_scale())
        outcome = attempt(retry, scale_before)
        gradient_names = outcome["nonfinite_gradients"]
        if gradient_names and not outcome["amp_enabled"]:
            optimizer.zero_grad()
            raise RuntimeError(
                f"optimizer step {step} has non-finite gradient elements with AMP "
                f"disabled; scale={scale_before}, parameters={gradient_names}"
            )
        if gradient_names:
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad()
            print(
                f"AMP backoff at optimizer step {step}: attempt {retry + 1}/"
                f"{max_retries + 1}; non-finite gradient elements observed in "
                f"{gradient_names}; scale {scale_before} -> {scale_after}. "
                "GradScaler handled the unsafe update; the scale change alone "
                "does not establish the underlying cause."
            )
            if scale_after >= scale_before:
                raise RuntimeError(
                    f"GradScaler did not lower its scale after non-finite "
                    f"gradients at optimizer step {step}; parameters="
                    f"{gradient_names}, scale={scale_before}->{scale_after}"
                )
            if retry == max_retries:
                raise RuntimeError(
                    f"AMP retries exhausted at optimizer step {step}; "
                    f"non-finite gradient parameters={gradient_names}, "
                    f"scale={scale_before}->{scale_after}"
                )
            continue

        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if scale_after < scale_before:
            optimizer.zero_grad()
            print(
                f"AMP backoff at optimizer step {step}: attempt {retry + 1}/"
                f"{max_retries + 1}; GradScaler skipped the update; "
                f"our gradient scan found no non-finite elements; scale "
                f"{scale_before} -> {scale_after}. The scale change alone "
                "does not establish the underlying cause."
            )
            if retry == max_retries:
                raise RuntimeError(
                    f"AMP retries exhausted at optimizer step {step} after "
                    f"GradScaler skipped the update; scale="
                    f"{scale_before}->{scale_after}"
                )
            continue

        scheduler.step()
        optimizer.zero_grad()
        return float(outcome["loss"]), retry
    raise AssertionError("bounded optimizer retry loop exited unexpectedly")


def _causal_supervised_token_count(
    labels: list[int], attention: list[int]
) -> int:
    if len(labels) != len(attention):
        raise ValueError("labels and attention mask must have equal lengths")
    return sum(
        label != -100 and bool(mask)
        for label, mask in zip(labels[1:], attention[1:])
    )


def _evaluate_validation(
    model,
    samples: list[tuple[list[int], list[int], list[int]]],
    device: str,
    limit: int | None = None,
) -> dict[str, float | int]:
    """Return supervised-token-weighted teacher-forced validation loss."""
    import torch
    from torch.cuda.amp import autocast

    selected = samples if limit is None else samples[:limit]
    weighted_loss = 0.0
    supervised_tokens = 0
    evaluated_examples = 0
    model.eval()
    with torch.no_grad():
        for ids, labels, attention in selected:
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            labels_t = torch.tensor([labels], dtype=torch.long, device=device)
            attention_t = torch.tensor([attention], dtype=torch.long, device=device)
            # Causal LM loss shifts internally, so position zero is never a target.
            count = _causal_supervised_token_count(labels, attention)
            if count == 0:
                continue
            with autocast(enabled=device == "cuda"):
                output = model(
                    input_ids=ids_t,
                    attention_mask=attention_t,
                    labels=labels_t,
                )
            weighted_loss += float(output.loss.item()) * count
            supervised_tokens += count
            evaluated_examples += 1
    return {
        "loss": weighted_loss / supervised_tokens if supervised_tokens else float("nan"),
        "supervised_tokens": supervised_tokens,
        "examples": evaluated_examples,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    style: str,
    manifest_path: str | None,
    drive_dir: str,
    cfg: dict,
    dry_run: bool = False,
    cache_selection: dict | None = None,
    cache_only: bool = False,
    smoke_test: bool = False,
    config_path: str | None = None,
) -> None:
    import torch
    from torch.cuda.amp import GradScaler, autocast
    from peft import LoraConfig, get_peft_model
    import anticipation.vocab as acfg

    tcfg = cfg["training"]
    if dry_run and smoke_test:
        raise ValueError("--dry_run and --smoke-test are mutually exclusive")
    if cache_only and cache_selection is None:
        raise ValueError("--cache-only requires an explicit cache selection")
    experiment_root = new_experiment_root(tcfg, drive_dir)
    if cache_selection is not None:
        cache_seed, cache_seq_len, train_probability, cache_context_groups = (
            _explicit_cache_training_settings(tcfg)
        )
    else:
        cache_seed = int(tcfg["random_seed"])
        cache_seq_len = tcfg["seq_len"]
        train_probability = float(tcfg.get("anticipatory_prob", 0.5))
        cache_context_groups = int(tcfg.get("context_groups", 170))
    control_offset = acfg.CONTROL_OFFSET
    seq_len = cache_seq_len
    invalid_window_policy = tcfg.get(
        "invalid_window_policy", INVALID_WINDOW_POLICY
    )
    if invalid_window_policy != INVALID_WINDOW_POLICY:
        raise ValueError(
            f"unsupported invalid_window_policy {invalid_window_policy!r}; "
            f"only {INVALID_WINDOW_POLICY!r} is allowed"
        )
    cache_dir = experiment_root / "token_cache"
    selected_cache_samples = None
    if cache_selection is not None:
        selected_cache_samples = load_selected_cache_set(
            cache_selection, cache_dir, style, cache_seed,
            seq_len, train_probability, cache_context_groups, control_offset,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if smoke_test and device != "cuda":
        raise RuntimeError("--smoke-test requires a CUDA runtime")

    seed = cfg["training"]["random_seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed)

    max_steps = 1 if smoke_test else (5 if dry_run else tcfg["max_steps"])
    micro_batch = tcfg["micro_batch"]
    grad_accum = tcfg["grad_accum"]
    save_every = tcfg["save_every"]
    print(f"CONTROL_OFFSET = {control_offset}")

    train_windows: list[dict] = []
    val_windows: list[dict] = []
    if selected_cache_samples is not None:
        train_samples, val_samples_by_mode, val_diagnostics = (
            selected_cache_samples
        )
        print(f"Loaded explicitly selected caches from {cache_dir}")
    else:
        if manifest_path is None:
            raise ValueError("manifest_path is required unless explicit caches are selected")
        train_windows = load_windows(manifest_path, style, "train")
        val_windows = load_windows(manifest_path, style, "val")
        rng.shuffle(train_windows)
        window_limit = tcfg.get("max_train_windows_per_style", 600)
        n_train = min(8, len(train_windows)) if dry_run else min(
            window_limit, len(train_windows)
        )
        train_windows = train_windows[:n_train]
        print(
            f"Train windows: {len(train_windows)} (limit {window_limit}), "
            f"Val windows: {len(val_windows)}"
        )
        print("Building token cache (skips if already done)...")
        cache_started = time.perf_counter()
        performance_cache: dict[str, list[tuple[float, float, int]]] = {}
        training_diagnostics = build_cache(
            train_windows, style, "train", control_offset, seq_len, cache_dir,
            seed, performance_cache, mode="mixed",
            anticipatory_prob=tcfg.get("anticipatory_prob", 0.5),
            context_groups=tcfg.get("context_groups", 170),
            invalid_window_policy=invalid_window_policy,
        )
        print(
            f"  Training preprocessing: "
            f"{json.dumps(training_diagnostics, sort_keys=True)}"
        )
        val_diagnostics = {}
        val_samples_by_mode = {}
        for validation_mode in ("unconditional", "conditioned"):
            val_diagnostics[validation_mode] = build_cache(
                val_windows, style, "val", control_offset, seq_len, cache_dir,
                seed, performance_cache, mode=validation_mode,
                anticipatory_prob=0.0,
                context_groups=tcfg.get("context_groups", 170),
                invalid_window_policy=invalid_window_policy,
            )
            val_samples_by_mode[validation_mode] = load_cached(
                cache_dir, style, "val", seed, val_windows, control_offset, seq_len,
                mode=validation_mode, anticipatory_prob=0.0,
                context_groups=tcfg.get("context_groups", 170),
                invalid_window_policy=invalid_window_policy,
            )
        print(
            f"Token cache build/check time: "
            f"{time.perf_counter() - cache_started:.1f}s"
        )
        train_samples = load_cached(
            cache_dir, style, "train", seed, train_windows, control_offset, seq_len,
            mode="mixed", anticipatory_prob=tcfg.get("anticipatory_prob", 0.5),
            context_groups=tcfg.get("context_groups", 170),
            invalid_window_policy=invalid_window_policy,
        )
    _validate_validation_cache_mappings(
        val_samples_by_mode, val_diagnostics, style
    )
    if not train_samples:
        raise ValueError(f"No supervised training examples for {style}; see diagnostics")
    for validation_mode, mode_samples in val_samples_by_mode.items():
        print(
            f"Validation {validation_mode}: {len(mode_samples)} examples; "
            f"diagnostics={json.dumps(val_diagnostics[validation_mode], sort_keys=True)}"
        )
    print(f"Loaded {len(train_samples)} supervised train examples")

    # Do not load the GPU model until both validation objectives and their
    # diagnostic provenance have been checked and reported.
    base = load_base_model(cfg["model"]["name"], device)
    for parameter in base.parameters():
        parameter.requires_grad = False
    if hasattr(base, "config") and hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    for module in base.modules():  # base model was not trained with dropout
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    base.enable_input_require_grads()
    base.gradient_checkpointing_enable()
    if hasattr(base, "is_gradient_checkpointing") and not base.is_gradient_checkpointing:
        raise RuntimeError("gradient checkpointing did not enable on the base model")

    lora_cfg_dict = cfg["lora"]
    _verify_lora_targets(base, lora_cfg_dict["target_modules"])

    lora_config = LoraConfig(
        r=lora_cfg_dict["r"],
        lora_alpha=lora_cfg_dict["lora_alpha"],
        lora_dropout=lora_cfg_dict["lora_dropout"],
        target_modules=lora_cfg_dict["target_modules"],
        fan_in_fan_out=lora_cfg_dict["fan_in_fan_out"],
        bias="none",
    )
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()
    _assert_trainable_parameters(model)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    initial_amp_scale = 1.0
    save_dir = None if smoke_test else prepare_new_adapter_dir(
        experiment_root, style
    )
    if save_dir is not None:
        if config_path is not None:
            config_bytes = Path(config_path).read_bytes()
        else:
            config_bytes = yaml.safe_dump(
                cfg, sort_keys=False, allow_unicode=True
            ).encode("utf-8")
        config_copy = save_dir / "training_config.yaml"
        with config_copy.open("xb") as handle:
            handle.write(config_bytes)
        cache_identities = {}
        if cache_selection is not None:
            selected_entries = cache_selection["entries"][style]
            cache_identities = {
                role: {
                    "path": str(Path(path).resolve()),
                    "sha256": _sha256_file(Path(path)),
                }
                for role, path in selected_entries.items()
            }
        else:
            cache_identities = {
                "train_mixed": {
                    "path": str(_cache_path(
                        cache_dir, style, "train", seed, train_windows,
                        control_offset, seq_len, "mixed",
                        tcfg.get("anticipatory_prob", 0.5),
                        tcfg.get("context_groups", 170),
                    ).resolve()),
                },
                **{
                    f"val_{mode}": {
                        "path": str(_cache_path(
                            cache_dir, style, "val", seed, val_windows,
                            control_offset, seq_len, mode, 0.0,
                            tcfg.get("context_groups", 170),
                        ).resolve()),
                    }
                    for mode in ("unconditional", "conditioned")
                },
            }
            for identity in cache_identities.values():
                identity["sha256"] = _sha256_file(Path(identity["path"]))
        provenance = {
            "style": style,
            "cache_schema": CACHE_SCHEMA,
            "amp_enabled": device == "cuda",
            "amp_initial_scale": initial_amp_scale,
            "cache_identities": cache_identities,
            "training_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "training_config_path": str(config_copy),
            "manifest_and_source_fingerprints": (
                "not available in explicit-cache mode; source files were not read"
                if cache_selection is not None
                else "cache identity includes source MIDI fingerprints"
            ),
            "excluded_window_provenance": (
                "not available from selected caches; consult preserved Drive evidence"
                if cache_selection is not None
                else "see preprocessing diagnostics"
            ),
        }
        provenance_path = save_dir / "training_provenance.json"
        with provenance_path.open("x", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
            handle.write("\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"],
    )
    warmup = tcfg["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _smoke_test_lr_lambda(lr_lambda, smoke_test)
    )
    scaler = GradScaler(enabled=device == "cuda", init_scale=initial_amp_scale)

    log_path = save_dir / "train_log.jsonl" if save_dir is not None else None
    step = 0

    model.train()
    optimizer.zero_grad()
    sample_order = list(range(len(train_samples)))
    rng.shuffle(sample_order)
    sample_idx = 0

    if (dry_run or smoke_test) and device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    initial_trainable = (
        [parameter.detach().clone() for parameter in model.parameters()
         if parameter.requires_grad]
        if smoke_test else []
    )

    while step < max_steps:
        if device == "cuda":
            torch.cuda.synchronize()
        optimizer_step_started = time.perf_counter()
        micro_batches = []
        token_counts = []
        for _ in range(grad_accum):
            batch_ids, batch_labels, batch_mask = [], [], []
            for _ in range(micro_batch):
                if sample_idx == len(sample_order):
                    rng.shuffle(sample_order)
                    sample_idx = 0
                ids, labels, mask = train_samples[sample_order[sample_idx]]
                sample_idx += 1
                batch_ids.append(ids)
                batch_labels.append(labels)
                batch_mask.append(mask)
            valid_tokens = sum(
                _causal_supervised_token_count(labels, masks)
                for labels, masks in zip(batch_labels, batch_mask)
            )
            micro_batches.append((batch_ids, batch_labels, batch_mask))
            token_counts.append(valid_tokens)

        loss_weights = token_weighted_loss_weights(token_counts)
        def attempt_update(_retry: int, scale: float) -> dict:
            optimizer.zero_grad()
            step_loss = 0.0
            for (batch_ids, batch_labels, batch_mask), weight in zip(
                micro_batches, loss_weights
            ):
                ids_t = torch.tensor(batch_ids, dtype=torch.long, device=device)
                lbl_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
                mask_t = torch.tensor(batch_mask, dtype=torch.long, device=device)
                with autocast(enabled=device == "cuda"):
                    output = model(
                        input_ids=ids_t, attention_mask=mask_t, labels=lbl_t
                    )
                    loss_value = float(output.loss.item())
                    if not math.isfinite(loss_value):
                        optimizer.zero_grad()
                    _require_finite_loss(loss_value, scale, step + 1)
                    weighted_loss = output.loss * weight
                scaler.scale(weighted_loss).backward()
                step_loss += loss_value * weight

            scaler.unscale_(optimizer)
            _, nonfinite_gradients = _inspect_and_clip_gradients(
                model.named_parameters(), torch, scale, step + 1, max_norm=1.0
            )
            return {
                "loss": step_loss,
                "nonfinite_gradients": nonfinite_gradients,
                "amp_enabled": device == "cuda",
            }

        step_loss, retries = _run_optimizer_update_with_retries(
            attempt_update, scaler, optimizer, scheduler, step + 1
        )
        step += 1
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - optimizer_step_started

        log_entry: dict = {
            "step": step,
            "train_loss": step_loss,
            "s_per_step": elapsed,
            "amp_retries": retries,
            "amp_initial_scale": initial_amp_scale,
        }

        if dry_run:
            if device == "cuda":
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                log_entry["peak_gpu_gb"] = peak_gb
                memory_report = f"{peak_gb:.2f}GB"
            else:
                memory_report = "UNAVAILABLE (CPU run)"
            print(f"[dry_run] step {step}: loss={log_entry['train_loss']:.4f} "
                  f"s/optimizer_step={elapsed:.2f} "
                  f"peak_gpu={memory_report}")
        elif smoke_test:
            updated_trainable = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
            ]
            changed = _training_parameters_changed(
                initial_trainable, updated_trainable, torch
            )
            if not changed:
                raise RuntimeError("real optimizer step did not update any LoRA parameter")
            model.eval()
            smoke_metrics = {}
            for validation_mode, mode_samples in val_samples_by_mode.items():
                metrics = _evaluate_validation(
                    model, mode_samples, device,
                    tcfg.get("validation_max_examples"),
                )
                if metrics["examples"] <= 0 or not math.isfinite(metrics["loss"]):
                    raise RuntimeError(
                        f"smoke validation {validation_mode} produced no finite loss"
                    )
                smoke_metrics[validation_mode] = metrics
            model.train()
            print(
                "SMOKE TEST PASSED: one LoRA optimizer update changed parameters; "
                f"validation={json.dumps(smoke_metrics, sort_keys=True)}"
            )
            return
        else:
            # Full deterministic validation by default; limit is in examples.
            if step % 10 == 0:
                model.eval()
                validation_limit = tcfg.get("validation_max_examples")
                for validation_mode, mode_samples in val_samples_by_mode.items():
                    metrics = _evaluate_validation(
                        model, mode_samples, device, validation_limit
                    )
                    label = (
                        "val_unconditional_piano"
                        if validation_mode == "unconditional"
                        else "val_conditioned_accompaniment"
                    )
                    log_entry[f"{label}_loss"] = metrics["loss"]
                    log_entry[f"{label}_tokens"] = metrics["supervised_tokens"]
                    log_entry[f"{label}_examples"] = metrics["examples"]
                model.train()

            print(f"step {step}: " + " ".join(
                f"{k}={v:.4f}" for k, v in log_entry.items() if k != "step"
            ))
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if step % save_every == 0 or step == max_steps:
                ckpt_dir = save_dir / f"step_{step}"
                model.save_pretrained(str(ckpt_dir))
                print(f"Saved checkpoint: {ckpt_dir}")

            if step == 50 and val_windows:
                _check_skyline_dominance(
                    model, val_windows[:4], control_offset, seq_len, device
                )

    if not dry_run:
        model.save_pretrained(str(save_dir / "final"))
        print(f"Training complete. Final adapter: {save_dir}/final")
    else:
        print(f"Dry run complete. {step} steps done.")


def _check_skyline_dominance(model, windows, control_offset, seq_len, device):
    """
    Generate a 10-second clip and check skyline dominance.
    Wrapped in try/except so a failure logs a warning and does not kill training.
    """
    print("Checking skyline dominance at step 50...")
    try:
        import torch
        import pretty_midi
        from model.data.skyline import skyline, Note as SNote
        from model.data.features import skyline_dominance, Note as FNote
        from model.amt_utils import write_notes_to_midi, notes_to_events, events_to_pretty_midi
        from model.generation.sampling import constrained_sampling
        from model.generation.anticipation_compat import generate_conditioned

        if not windows:
            print("  No validation windows for dominance check.")
            return

        row = windows[0]
        pm = pretty_midi.PrettyMIDI(row["midi_path"])
        mel_notes: list[SNote] = []
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                if n.start >= row["start_sec"] and n.end <= row["end_sec"]:
                    mel_notes.append(SNote(
                        n.start - row["start_sec"],
                        n.end - row["start_sec"], n.pitch
                    ))
        mel = skyline(mel_notes)
        if not mel:
            print("  Empty melody, skipping.")
            return

        mel_events = notes_to_events(mel)
        controls = [t + control_offset for t in mel_events]

        model.eval()
        try:
            with constrained_sampling(), torch.no_grad():
                output = generate_conditioned(
                    model, start_time=0, end_time=max(n.end for n in mel),
                    inputs=[], controls=controls, top_p=0.95,
                    max_events=100_000,
                )
        finally:
            model.train()

        gen_pm = events_to_pretty_midi(output)
        gen_notes = [FNote(n.start, n.end, n.pitch)
                     for inst in gen_pm.instruments
                     for n in inst.notes]
        mel_fnotes = [FNote(n.start, n.end, n.pitch) for n in mel]
        dom = skyline_dominance(mel_fnotes, gen_notes)
        print(f"  Skyline dominance at step 50: {dom:.3f}")
    except Exception:
        import traceback
        print("  WARNING: skyline dominance check failed:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--style", choices=["baroque", "romantic"], required=True)
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 5 steps on 8 windows to validate pipeline")
    parser.add_argument(
        "--cache-selection",
        help="JSON file selecting exact validated train/validation cache paths",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Require explicit caches; do not read manifests/MIDI or build caches",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one CUDA update and both cached validation objectives; save no adapter",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    drive_dir = cfg["training"]["drive_dir"]
    manifest_path = (
        None if args.cache_only
        else str(Path(cfg["data"]["output_dir"]) / "manifest.csv")
    )

    if args.cache_only and not args.cache_selection:
        parser.error("--cache-only requires --cache-selection")
    if args.cache_selection and not args.cache_only:
        parser.error("--cache-selection must be used with --cache-only")
    selection = None
    if args.cache_selection:
        cache_dir = new_experiment_root(cfg["training"], drive_dir) / "token_cache"
        selection = _load_cache_selection(args.cache_selection, cache_dir)
    train(
        args.style,
        None if args.cache_only else manifest_path,
        drive_dir,
        cfg,
        dry_run=args.dry_run,
        cache_selection=selection,
        cache_only=args.cache_only,
        smoke_test=args.smoke_test,
        config_path=str(config_path),
    )


if __name__ == "__main__":
    main()
