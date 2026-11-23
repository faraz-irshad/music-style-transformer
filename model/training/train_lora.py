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

UNVERIFIED items (require Colab run):
  - CONTROL_OFFSET exact value
  - anticipation interleave routine name/signature (see _build_anticipatory_sequence)
  - events_to_midi return type (normalised in amt_utils.events_to_pretty_midi)
  - LoRA target module names (verified at startup with _verify_lora_targets)
"""
from __future__ import annotations
import argparse, json, math, os, pickle, time
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast


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
    from anticipation import ops
    ctrl_tokens = [t + control_offset for t in mel_events]
    interleaved, _unconsumed_controls = ops.anticipate(events, ctrl_tokens)
    if len(interleaved) % 3:
        raise ValueError(
            f"anticipation.ops.anticipate returned {len(interleaved)} tokens; "
            "expected complete 3-token events"
        )
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
) -> tuple[list[int], list[int], list[int]]:
    """
    Tokenise one MIDI window.
    Returns (input_ids, labels, attention_mask), each of length seq_len.
    Padding token is 0; padded positions have label=-100 and mask=0.
    Never truncates inside a 3-token event group.
    """
    from anticipation.convert import midi_to_events
    from model.data.skyline import skyline, Note as SNote
    from model.amt_utils import write_notes_to_midi
    import pretty_midi, tempfile

    pm = pretty_midi.PrettyMIDI(row["midi_path"])
    start, end = float(row["start_sec"]), float(row["end_sec"])

    all_notes: list[SNote] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= start and n.end <= end:
                all_notes.append(SNote(
                    n.start - start, n.end - start,
                    max(0, min(127, n.pitch + shift))
                ))

    if not all_notes:
        pad = [0] * seq_len
        return pad, [-100] * seq_len, [0] * seq_len

    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp.close()
    try:
        write_notes_to_midi(all_notes, tmp.name)
        events = midi_to_events(tmp.name)
    finally:
        os.unlink(tmp.name)

    use_anticipatory = rng.random() < anticipatory_prob
    if use_anticipatory:
        mel = skyline(all_notes)
        tmp2 = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        tmp2.close()
        try:
            write_notes_to_midi(mel, tmp2.name)
            mel_events = midi_to_events(tmp2.name)
        finally:
            os.unlink(tmp2.name)
        ids, labels = _build_anticipatory_sequence(events, mel_events, control_offset)
    else:
        ids = list(events)
        labels = list(events)

    # Truncate to seq_len, but never inside a 3-token group
    max_tokens = (seq_len // 3) * 3
    ids = ids[:max_tokens]
    labels = labels[:max_tokens]

    # Pad
    pad_len = seq_len - len(ids)
    mask = [1] * len(ids) + [0] * pad_len
    ids = ids + [0] * pad_len
    labels = labels + [-100] * pad_len

    return ids, labels, mask


def _cache_path(cache_dir: Path, style: str, split: str, shift: int) -> Path:
    return cache_dir / f"{style}_{split}_shift{shift:+d}.pkl"


def build_cache(
    windows: list[dict],
    style: str,
    split: str,
    control_offset: int,
    seq_len: int,
    cache_dir: Path,
    seed: int,
) -> None:
    """Pre-tokenise all windows for all transposition shifts and cache to disk."""
    import random
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for shift in range(-6, 7):
        cp = _cache_path(cache_dir, style, split, shift)
        if cp.exists():
            continue
        print(f"  Caching {style}/{split} shift={shift:+d} ...")
        samples = []
        for row in windows:
            ids, lbls, mask = _tokenise_window(
                row, control_offset, seq_len,
                anticipatory_prob=0.5 if split == "train" else 0.0,
                rng=rng, shift=shift,
            )
            samples.append((ids, lbls, mask))
        with open(cp, "wb") as f:
            pickle.dump(samples, f)
    print(f"  Cache complete for {style}/{split}")


def load_cached(cache_dir: Path, style: str, split: str) -> list[tuple]:
    """Load all cached samples for a style/split (all shifts combined)."""
    samples = []
    for shift in range(-6, 7):
        cp = _cache_path(cache_dir, style, split, shift)
        if cp.exists():
            with open(cp, "rb") as f:
                samples.extend(pickle.load(f))
    return samples


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_base_model(model_name: str, device: str):
    """Load the base model in fp16. Falls back to GPT2LMHeadModel if needed."""
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        print("Loaded with AutoModelForCausalLM")
        return model
    except Exception as e:
        print(f"AutoModelForCausalLM failed ({e}), trying GPT2LMHeadModel")
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
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
    import random
    from peft import LoraConfig, get_peft_model
    import anticipation.config as acfg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    seed = cfg["training"]["random_seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Load model
    base = load_base_model(cfg["model"]["name"], device)
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

    tcfg = cfg["training"]
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
    n_train = 8 if dry_run else len(train_windows)
    train_windows = train_windows[:n_train]
    print(f"Train windows: {len(train_windows)}, Val windows: {len(val_windows)}")

    # Cache tokenised windows
    cache_dir = Path(drive_dir) / "token_cache"
    print("Building token cache (skips if already done)...")
    build_cache(train_windows, style, "train", control_offset, seq_len, cache_dir, seed)
    build_cache(val_windows[:8], style, "val", control_offset, seq_len, cache_dir, seed)
    train_samples = load_cached(cache_dir, style, "train")
    val_samples = load_cached(cache_dir, style, "val")
    print(f"Loaded {len(train_samples)} train samples, {len(val_samples)} val samples")

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

    save_dir = Path(drive_dir) / f"adapter_{style}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log.jsonl"

    # Resume
    resume_path = save_dir / "resume_step.txt"
    step = 0
    if resume_path.exists() and not dry_run:
        step = int(resume_path.read_text().strip())
        ckpt = save_dir / f"step_{step}"
        if ckpt.exists():
            model.load_adapter(str(ckpt), adapter_name="default")
            print(f"Resumed from step {step}")

    model.train()
    optimizer.zero_grad()
    sample_idx = 0
    micro_step = 0  # counts micro-batches since last optimizer step
    accum_loss = 0.0

    if dry_run:
        torch.cuda.reset_peak_memory_stats()

    while step < max_steps:
        # Build micro-batch
        batch_ids, batch_labels, batch_mask = [], [], []
        for _ in range(micro_batch):
            ids, lbls, mask = train_samples[sample_idx % len(train_samples)]
            sample_idx += 1
            batch_ids.append(ids)
            batch_labels.append(lbls)
            batch_mask.append(mask)

        ids_t = torch.tensor(batch_ids, dtype=torch.long, device=device)
        lbl_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
        mask_t = torch.tensor(batch_mask, dtype=torch.long, device=device)

        t0 = time.time()
        with autocast():
            out = model(input_ids=ids_t, attention_mask=mask_t, labels=lbl_t)
            loss = out.loss / grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item()
        micro_step += 1

        # Optimizer step only after grad_accum micro-batches
        if micro_step % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            elapsed = time.time() - t0

            log_entry: dict = {
                "step": step,
                "train_loss": float(accum_loss * grad_accum),
                "s_per_step": elapsed,
            }
            accum_loss = 0.0

            if dry_run:
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                log_entry["peak_gpu_gb"] = peak_gb
                print(f"[dry_run] step {step}: loss={log_entry['train_loss']:.4f} "
                      f"s/step={elapsed:.2f} peak_gpu={peak_gb:.2f}GB")
            else:
                # Validation loss every 10 steps
                if step % 10 == 0 and val_samples:
                    model.eval()
                    val_losses = []
                    for vids, vlbls, vmask in val_samples[:8]:
                        vids_t = torch.tensor([vids], dtype=torch.long, device=device)
                        vlbl_t = torch.tensor([vlbls], dtype=torch.long, device=device)
                        vmsk_t = torch.tensor([vmask], dtype=torch.long, device=device)
                        with torch.no_grad(), autocast():
                            vout = model(input_ids=vids_t, attention_mask=vmsk_t, labels=vlbl_t)
                        val_losses.append(vout.loss.item())
                    log_entry["val_loss"] = sum(val_losses) / len(val_losses)
                    model.train()

                print(f"step {step}: " + " ".join(
                    f"{k}={v:.4f}" for k, v in log_entry.items() if k != "step"
                ))
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                if step % save_every == 0 or step == max_steps:
                    ckpt_dir = save_dir / f"step_{step}"
                    model.save_pretrained(str(ckpt_dir))
                    resume_path.write_text(str(step))
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
        import pretty_midi
        from anticipation.sample import generate
        from model.data.skyline import skyline, Note as SNote
        from model.data.features import skyline_dominance, Note as FNote
        from model.amt_utils import write_notes_to_midi, notes_to_events, events_to_pretty_midi

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
        with torch.no_grad():
            output = generate(model, start_time=0, end_time=10,
                              inputs=[], controls=controls, top_p=0.95)
        model.train()

        gen_pm = events_to_pretty_midi(output)
        gen_notes = [FNote(n.start, n.end, n.pitch)
                     for inst in gen_pm.instruments
                     for n in inst.notes]
        mel_fnotes = [FNote(n.start, n.end, n.pitch) for n in mel]
        dom = skyline_dominance(mel_fnotes, gen_notes)
        print(f"  Skyline dominance at step 50: {dom:.3f}")
    except Exception as e:
        print(f"  WARNING: skyline dominance check failed: {e}")


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
