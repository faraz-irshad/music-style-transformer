"""
train_lora.py

Fine-tunes LoRA adapters (adapter_baroque, adapter_romantic) on top of the
frozen Anticipatory Music Transformer (stanford-crfm/music-medium-800k).

Run on Colab GPU via colab/05_train_lora.ipynb.
Do NOT run locally — requires GPU and the anticipation library.

Provenance: Layer C (learned).
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_windows(manifest_path: str, style: str, split: str) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(manifest_path)
    return df[(df["style"] == style) & (df["split"] == split)].to_dict("records")


def window_to_tokens(
    row: dict,
    control_offset: int,
    seq_len: int,
    anticipatory_prob: float = 0.5,
    rng: "random.Random | None" = None,
) -> tuple[list[int], list[int]]:
    """
    Convert a MIDI window to (input_ids, labels).
    ~50% of windows use anticipatory interleaved form (melody as controls).
    Labels for control tokens are set to -100 (masked).

    Returns (input_ids, labels) each of length seq_len.
    """
    import random as _random
    from anticipation.convert import midi_to_events
    from anticipation import ops
    from data.skyline import skyline, Note as SNote

    if rng is None:
        rng = _random.Random(42)

    # Load and clip MIDI window
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(row["midi_path"])
    start, end = row["start_sec"], row["end_sec"]

    # Collect all notes in window
    all_notes: list[SNote] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= start and n.end <= end:
                all_notes.append(SNote(n.start - start, n.end - start, n.pitch))

    if not all_notes:
        return [0] * seq_len, [-100] * seq_len

    # 12-semitone transposition augmentation
    shift = rng.randint(-6, 6)
    all_notes = [SNote(n.start, n.end, n.pitch + shift) for n in all_notes]

    # Convert to anticipation event tokens
    # UNVERIFIED: exact API — using midi_to_events on a temp file
    import tempfile, mido as _mido
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp.close()
    try:
        _write_notes_to_midi(all_notes, tmp.name)
        events = midi_to_events(tmp.name)
    finally:
        os.unlink(tmp.name)

    use_anticipatory = rng.random() < anticipatory_prob
    labels = list(events)

    if use_anticipatory:
        # Extract skyline melody, shift to control space, interleave
        from data.skyline import skyline
        mel = skyline(all_notes)
        mel_events = _notes_to_events(mel)
        ctrl_events = [t + control_offset for t in mel_events]
        # Interleave: for each event group (3 tokens), insert matching control if exists
        # UNVERIFIED: exact interleave routine — using simple prepend of controls
        events = ctrl_events + events
        labels = [-100] * len(ctrl_events) + labels

    # Truncate or pad to seq_len
    events = events[:seq_len]
    labels = labels[:seq_len]
    pad_len = seq_len - len(events)
    events = events + [0] * pad_len
    labels = labels + [-100] * pad_len

    return events, labels


def _write_notes_to_midi(notes, path: str, tempo: float = 100.0) -> None:
    import mido
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo_us = int(60_000_000 / tempo)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    events = []
    for n in notes:
        tpb = 480
        tick_s = int(n.start * tempo / 60.0 * tpb)
        tick_e = int(n.end * tempo / 60.0 * tpb)
        p = max(0, min(127, n.pitch))
        events.append((tick_s, "on", p))
        events.append((tick_e, "off", p))
    events.sort(key=lambda x: x[0])
    prev = 0
    for tick, kind, pitch in events:
        delta = tick - prev
        if kind == "on":
            track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
        prev = tick
    mid.save(path)


def _notes_to_events(notes) -> list[int]:
    """Stub: convert notes to anticipation integer tokens via temp MIDI file."""
    import tempfile, os
    from anticipation.convert import midi_to_events
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp.close()
    try:
        _write_notes_to_midi(notes, tmp.name)
        return midi_to_events(tmp.name)
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    style: str,
    manifest_path: str,
    drive_dir: str,
    cfg: dict,
) -> None:
    import random
    from transformers import GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model
    import anticipation.config as acfg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    rng = random.Random(cfg["training"]["random_seed"])
    torch.manual_seed(cfg["training"]["random_seed"])

    # Load model
    model_name = cfg["model"]["name"]
    try:
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
    except Exception:
        base = GPT2LMHeadModel.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)

    base.enable_input_require_grads()
    base.gradient_checkpointing_enable()

    lora_cfg_dict = cfg["lora"]
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

    # Optimizer with cosine LR
    tcfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"],
    )
    max_steps = tcfg["max_steps"]
    warmup = tcfg["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # Data
    train_windows = load_windows(manifest_path, style, "train")
    val_windows = load_windows(manifest_path, style, "val")
    print(f"Train windows: {len(train_windows)}, Val windows: {len(val_windows)}")

    control_offset = acfg.CONTROL_OFFSET
    seq_len = tcfg["seq_len"]
    micro_batch = tcfg["micro_batch"]
    grad_accum = tcfg["grad_accum"]
    save_every = tcfg["save_every"]

    save_dir = Path(drive_dir) / f"adapter_{style}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log.jsonl"

    model.train()
    step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    # Resume if checkpoint exists
    resume_path = save_dir / "resume_step.txt"
    if resume_path.exists():
        step = int(resume_path.read_text().strip())
        ckpt = save_dir / f"step_{step}"
        if ckpt.exists():
            model.load_adapter(str(ckpt), adapter_name="default")
            print(f"Resumed from step {step}")

    window_idx = 0
    while step < max_steps:
        # Build micro-batch
        batch_ids, batch_labels = [], []
        for _ in range(micro_batch):
            row = train_windows[window_idx % len(train_windows)]
            window_idx += 1
            ids, lbls = window_to_tokens(row, control_offset, seq_len, rng=rng)
            batch_ids.append(ids)
            batch_labels.append(lbls)

        ids_t = torch.tensor(batch_ids, dtype=torch.long, device=device)
        lbl_t = torch.tensor(batch_labels, dtype=torch.long, device=device)

        with autocast():
            out = model(input_ids=ids_t, labels=lbl_t)
            loss = out.loss / grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item()

        if (window_idx % grad_accum) == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            log_entry = {"step": step, "train_loss": accum_loss * grad_accum}
            accum_loss = 0.0

            # Validation loss every 10 steps
            if step % 10 == 0 and val_windows:
                model.eval()
                val_losses = []
                for vrow in val_windows[:8]:
                    vids, vlbls = window_to_tokens(
                        vrow, control_offset, seq_len, anticipatory_prob=0.0
                    )
                    vids_t = torch.tensor([vids], dtype=torch.long, device=device)
                    vlbl_t = torch.tensor([vlbls], dtype=torch.long, device=device)
                    with torch.no_grad(), autocast():
                        vout = model(input_ids=vids_t, labels=vlbl_t)
                    val_losses.append(vout.loss.item())
                log_entry["val_loss"] = sum(val_losses) / len(val_losses)
                model.train()

            print(f"step {step}: " + " ".join(f"{k}={v:.4f}" for k, v in log_entry.items() if k != "step"))
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Save checkpoint
            if step % save_every == 0 or step == max_steps:
                ckpt_dir = save_dir / f"step_{step}"
                model.save_pretrained(str(ckpt_dir))
                resume_path.write_text(str(step))
                print(f"Saved checkpoint: {ckpt_dir}")

            # Step-50 skyline dominance check
            if step == 50:
                _check_skyline_dominance(model, val_windows[:4], control_offset,
                                         seq_len, device, cfg)

    # Save final adapter
    model.save_pretrained(str(save_dir / "final"))
    print(f"Training complete. Final adapter: {save_dir}/final")


def _check_skyline_dominance(model, windows, control_offset, seq_len, device, cfg):
    """Generate a short clip and check that melody remains dominant."""
    print("Checking skyline dominance at step 50...")
    from anticipation.sample import generate
    from anticipation.convert import events_to_midi
    import anticipation.config as acfg
    from data.skyline import skyline, Note as SNote
    from data.features import skyline_dominance, Note as FNote

    if not windows:
        print("  No validation windows for dominance check.")
        return

    row = windows[0]
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(row["midi_path"])
    mel_notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= row["start_sec"] and n.end <= row["end_sec"]:
                mel_notes.append(SNote(n.start - row["start_sec"],
                                       n.end - row["start_sec"], n.pitch))
    mel = skyline(mel_notes)
    if not mel:
        print("  Empty melody, skipping.")
        return

    mel_events = _notes_to_events(mel)
    controls = [t + control_offset for t in mel_events]

    model.eval()
    with torch.no_grad():
        output = generate(model, start_time=0, end_time=10,
                          inputs=[], controls=controls, top_p=0.95)
    model.train()

    from anticipation.convert import events_to_midi
    mid = events_to_midi(output)
    gen_notes = []
    for inst in mid.instruments:
        for n in inst.notes:
            gen_notes.append(FNote(n.start, n.end, n.pitch))

    mel_fnotes = [FNote(n.start, n.end, n.pitch) for n in mel]
    dom = skyline_dominance(mel_fnotes, gen_notes)
    print(f"  Skyline dominance at step 50: {dom:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--style", choices=["baroque", "romantic"], required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    manifest_path = str(Path(cfg["data"]["output_dir"]) / "manifest.csv")
    drive_dir = cfg["training"]["drive_dir"]

    train(args.style, manifest_path, drive_dir, cfg)


if __name__ == "__main__":
    main()
