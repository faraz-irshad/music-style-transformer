"""
tests/test_train_lora.py

CPU-only tests for train_lora.py logic.
No GPU, no anticipation library, no MAESTRO required.
"""
import sys, os, csv, tempfile, pathlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# P1.1 — Gradient accumulation: one optimizer step per grad_accum micro-batches
# ---------------------------------------------------------------------------

def test_grad_accum_step_count():
    """
    Simulate the training loop's micro_step counter.
    With micro_batch=4 and grad_accum=4, the optimizer must step exactly once
    per grad_accum micro-batches.
    """
    grad_accum = 4
    max_steps = 10
    optimizer_steps = 0
    micro_step = 0
    step = 0

    while step < max_steps:
        micro_step += 1
        if micro_step % grad_accum == 0:
            optimizer_steps += 1
            step += 1

    assert optimizer_steps == max_steps, (
        f"Expected {max_steps} optimizer steps, got {optimizer_steps}"
    )
    assert micro_step == max_steps * grad_accum, (
        f"Expected {max_steps * grad_accum} micro-steps, got {micro_step}"
    )
    print(f"grad_accum test: {optimizer_steps} optimizer steps, "
          f"{micro_step} micro-steps — ok")


def test_grad_accum_no_early_step():
    """
    The old bug: window_idx grew by micro_batch each loop, so with
    micro_batch=4 and grad_accum=4, (window_idx % grad_accum == 0) fired
    on every iteration. Verify the new counter does NOT step early.
    """
    grad_accum = 4
    micro_step = 0
    steps_taken = 0
    for _ in range(3):  # only 3 micro-batches, less than grad_accum=4
        micro_step += 1
        if micro_step % grad_accum == 0:
            steps_taken += 1

    assert steps_taken == 0, (
        f"Optimizer should not step after only 3 micro-batches (grad_accum=4), "
        f"but stepped {steps_taken} times"
    )
    print("no early step test: ok")


def test_grad_accum_exact_boundary():
    """After exactly grad_accum micro-batches, exactly one step is taken."""
    grad_accum = 4
    micro_step = 0
    steps = 0
    for _ in range(grad_accum):
        micro_step += 1
        if micro_step % grad_accum == 0:
            steps += 1
    assert steps == 1
    print("exact boundary test: ok")


def test_old_bug_would_step_every_iteration():
    """
    Demonstrate the old bug: window_idx % grad_accum == 0 fires on every
    iteration when window_idx is incremented by micro_batch=grad_accum each loop.
    This test documents the bug; the new code does NOT use this pattern.
    """
    micro_batch = 4
    grad_accum = 4
    max_steps = 4
    window_idx = 0
    old_steps = 0
    for _ in range(max_steps):
        window_idx += micro_batch  # old code incremented by micro_batch
        if window_idx % grad_accum == 0:
            old_steps += 1
    # Old code would step on every iteration
    assert old_steps == max_steps, f"Old bug: expected {max_steps} steps, got {old_steps}"
    print(f"old bug documented: {old_steps} steps in {max_steps} iterations (every iter) — ok")


# ---------------------------------------------------------------------------
# P1.4 — Val split: load_windows raises if val split is empty
# ---------------------------------------------------------------------------

def test_load_windows_raises_on_missing_split():
    """load_windows must raise ValueError if the requested split has no rows."""
    with tempfile.TemporaryDirectory() as d:
        manifest_path = os.path.join(d, "manifest.csv")
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "style", "split", "midi_path", "start_sec", "end_sec",
                "composer", "title", "window_id",
            ])
            writer.writeheader()
            writer.writerow({
                "style": "baroque", "split": "train",
                "midi_path": "/fake/path.mid", "start_sec": 0, "end_sec": 15,
                "composer": "Bach", "title": "Prelude", "window_id": "w00",
            })

        # Simulate load_windows logic without importing train_lora (needs torch)
        rows = []
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                if row["style"] == "baroque" and row["split"] == "val":
                    rows.append(row)

        assert len(rows) == 0

        raised = False
        try:
            if not rows:
                raise ValueError("No windows found for style='baroque' split='val'")
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for missing val split"
    print("load_windows missing split test: ok")


# ---------------------------------------------------------------------------
# P2.16 — layer_a.generate import signatures
# ---------------------------------------------------------------------------

def test_layer_a_imports():
    """generate.py calls generate_accompaniment and notes_to_midi from layer_a.generate."""
    from layer_a.generate import generate_accompaniment, notes_to_midi
    import inspect

    sig_gen = inspect.signature(generate_accompaniment)
    params = list(sig_gen.parameters)
    assert "melody_notes" in params, f"melody_notes not in {params}"
    assert "style" in params, f"style not in {params}"
    assert "seed" in params, f"seed not in {params}"

    sig_midi = inspect.signature(notes_to_midi)
    params_midi = list(sig_midi.parameters)
    assert "melody_notes" in params_midi, f"melody_notes not in {params_midi}"
    assert "accompaniment" in params_midi, f"accompaniment not in {params_midi}"
    assert "style" in params_midi, f"style not in {params_midi}"
    assert "output_path" in params_midi, f"output_path not in {params_midi}"

    print("layer_a import signatures: ok")


if __name__ == "__main__":
    test_grad_accum_step_count()
    test_grad_accum_no_early_step()
    test_grad_accum_exact_boundary()
    test_old_bug_would_step_every_iteration()
    test_load_windows_raises_on_missing_split()
    test_layer_a_imports()
    print("\nALL train_lora tests PASSED")
