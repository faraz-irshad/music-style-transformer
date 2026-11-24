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
    SEQUENCE_POLICY,
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
    Build an anticipatory interleaved sequence using the anticipation library's
    own routine so the training format matches what generate() expects.

    anticipation.ops.anticipate returns (interleaved_tokens,
    unconsumed_controls). Controls are masked from the training loss.

    Returns (input_ids, labels).
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
) -> Path:
    signature = [
        (
            row["midi_path"],
            float(row["start_sec"]),
            float(row["end_sec"]),
            row.get("window_id", ""),
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
        "window_signature": signature,
    }
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return cache_dir / f"{style}_{split}_{digest}_{CACHE_SCHEMA}.pkl"


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
) -> dict:
    """Cache examples without mutating or reusing an incompatible old schema."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(
        cache_dir, style, split, seed, windows, control_offset, seq_len,
        mode, anticipatory_prob, context_groups,
    )
    if cp.exists():
        with cp.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"incompatible cache schema in {cp}")
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
            samples.extend(row_examples)
            window_stats.append(stats)
        except Exception as exc:
            invalid = {
                "window_id": row.get("window_id", ""),
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
    payload = {
        "schema": CACHE_SCHEMA,
        "settings": {
            "mode": mode,
            "anticipatory_prob": anticipatory_prob,
            "context_groups": context_groups,
            "sequence_length": seq_len,
            "seed": seed,
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
        "empty_windows": sum(bool(stats.get("empty")) for stats in window_stats),
        "all_masked_examples": sum(
            int(stats.get("all_masked_examples", 0)) for stats in window_stats
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
) -> list[tuple]:
    """Load cached samples for a style/split."""
    cp = _cache_path(
        cache_dir, style, split, seed, windows, control_offset, seq_len,
        mode, anticipatory_prob, context_groups,
    )
    with cp.open("rb") as handle:
        cached = pickle.load(handle)
    if cached.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"incompatible cache schema in {cp}")
    return cached["samples"]


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


def _assert_dtypes(model) -> None:
    """Assert base params are fp16 and LoRA params are fp32."""
    import torch

    lora_fp32_ok = True
    base_fp16_ok = True
    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        if is_lora and param.requires_grad:
            if param.dtype != torch.float32:
                print(f"  WARNING: LoRA param {name} is {param.dtype}, expected float32")
                lora_fp32_ok = False
        elif not param.requires_grad:
            if param.dtype != torch.float16:
                # Some non-trainable params may be fp32 (e.g. layer norms) — warn only
                pass
    if lora_fp32_ok:
        print("  LoRA params dtype: float32 ok")
    # Print a sample
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            print(f"  Sample LoRA param: {name} dtype={param.dtype}")
            break
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  Sample frozen param: {name} dtype={param.dtype}")
            break


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
            count = sum(
                label != -100 and mask
                for label, mask in zip(labels[1:], attention[1:])
            )
            if count == 0:
                continue
            with autocast():
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
    manifest_path: str,
    drive_dir: str,
    cfg: dict,
    dry_run: bool = False,
) -> None:
    import torch
    from torch.cuda.amp import GradScaler, autocast
    from peft import LoraConfig, get_peft_model
    import anticipation.vocab as acfg

    tcfg = cfg["training"]
    experiment_root = new_experiment_root(tcfg, drive_dir)
    save_dir = prepare_new_adapter_dir(experiment_root, style)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    seed = cfg["training"]["random_seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Load model
    base = load_base_model(cfg["model"]["name"], device)
    for module in base.modules():  # base model was not trained with dropout
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    base.enable_input_require_grads()
    base.gradient_checkpointing_enable()

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
    _assert_dtypes(model)

    max_steps = 5 if dry_run else tcfg["max_steps"]
    seq_len = tcfg["seq_len"]
    micro_batch = tcfg["micro_batch"]
    grad_accum = tcfg["grad_accum"]
    save_every = tcfg["save_every"]
    control_offset = acfg.CONTROL_OFFSET
    print(f"CONTROL_OFFSET = {control_offset}")

    # Data
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

    # Cache tokenised windows
    cache_dir = experiment_root / "token_cache"
    print("Building token cache (skips if already done)...")
    cache_started = time.perf_counter()
    performance_cache: dict[str, list[tuple[float, float, int]]] = {}
    training_diagnostics = build_cache(
        train_windows, style, "train", control_offset, seq_len, cache_dir,
        seed, performance_cache, mode="mixed",
        anticipatory_prob=tcfg.get("anticipatory_prob", 0.5),
        context_groups=tcfg.get("context_groups", 170),
    )
    print(f"  Training preprocessing: {json.dumps(training_diagnostics, sort_keys=True)}")
    val_diagnostics = {}
    val_samples_by_mode = {}
    for validation_mode in ("unconditional", "conditioned"):
        val_diagnostics[validation_mode] = build_cache(
        val_windows, style, "val", control_offset, seq_len, cache_dir,
            seed, performance_cache, mode=validation_mode,
            anticipatory_prob=0.0,
            context_groups=tcfg.get("context_groups", 170),
        )
        val_samples_by_mode[validation_mode] = load_cached(
            cache_dir, style, "val", seed, val_windows, control_offset, seq_len,
            mode=validation_mode, anticipatory_prob=0.0,
            context_groups=tcfg.get("context_groups", 170),
        )
    print(f"Token cache build/check time: {time.perf_counter() - cache_started:.1f}s")
    train_samples = load_cached(
        cache_dir, style, "train", seed, train_windows, control_offset, seq_len,
        mode="mixed", anticipatory_prob=tcfg.get("anticipatory_prob", 0.5),
        context_groups=tcfg.get("context_groups", 170),
    )
    if not train_samples:
        raise ValueError(f"No supervised training examples for {style}; see diagnostics")
    for validation_mode, mode_samples in val_samples_by_mode.items():
        print(
            f"Validation {validation_mode}: {len(mode_samples)} examples; "
            f"diagnostics={json.dumps(val_diagnostics[validation_mode], sort_keys=True)}"
        )
    print(f"Loaded {len(train_samples)} supervised train examples")

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

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    log_path = save_dir / "train_log.jsonl"
    step = 0

    model.train()
    optimizer.zero_grad()
    sample_order = list(range(len(train_samples)))
    rng.shuffle(sample_order)
    sample_idx = 0

    if dry_run and device == "cuda":
        torch.cuda.reset_peak_memory_stats()

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
                label != -100
                for labels in batch_labels
                for label in labels[1:]
            )
            micro_batches.append((batch_ids, batch_labels, batch_mask))
            token_counts.append(valid_tokens)

        loss_weights = token_weighted_loss_weights(token_counts)
        optimizer.zero_grad()
        step_loss = 0.0
        for (batch_ids, batch_labels, batch_mask), weight in zip(
            micro_batches, loss_weights
        ):
            ids_t = torch.tensor(batch_ids, dtype=torch.long, device=device)
            lbl_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
            mask_t = torch.tensor(batch_mask, dtype=torch.long, device=device)
            with autocast():
                output = model(
                    input_ids=ids_t, attention_mask=mask_t, labels=lbl_t
                )
                weighted_loss = output.loss * weight
            scaler.scale(weighted_loss).backward()
            step_loss += float(output.loss.item()) * weight

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - optimizer_step_started

        log_entry: dict = {
            "step": step,
            "train_loss": step_loss,
            "s_per_step": elapsed,
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

            if step == 50:
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
        with constrained_sampling(), torch.no_grad():
            output = generate_conditioned(
                model, start_time=0, end_time=max(n.end for n in mel),
                inputs=[], controls=controls, top_p=0.95,
            )
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    manifest_path = str(Path(cfg["data"]["output_dir"]) / "manifest.csv")
    drive_dir = cfg["training"]["drive_dir"]

    train(args.style, manifest_path, drive_dir, cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
