"""
nll_eval.py

Held-out cross-style negative log-likelihood matrix.
Scores Baroque and Romantic test windows under:
  - base model
  - adapter_baroque
  - adapter_romantic

Decomposes NLL into time, duration, and note terms where feasible.

Run on Colab GPU after training is complete.
Usage:
    python nll_eval.py --config config.yaml
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
import yaml


# AMT token grammar: each event = 3 tokens (time_bin, duration_bin, pitch).
# NLL decomposition: average loss over positions 0,3,6,... = time terms;
# positions 1,4,7,... = duration terms; positions 2,5,8,... = note terms.
# UNVERIFIED: exact token position semantics — depends on anticipation tokeniser.

def _nll_decomposed(
    model, ids_t: torch.Tensor, lbl_t: torch.Tensor, mask_t: torch.Tensor
) -> dict[str, float]:
    """
    Compute per-token NLL and decompose into time/duration/note terms.
    Returns dict with keys: total, time, duration, note.
    """
    from torch.cuda.amp import autocast
    with torch.no_grad(), autocast():
        out = model(input_ids=ids_t, attention_mask=mask_t, labels=lbl_t)
    total_nll = float(out.loss.item())

    # Decompose: get per-token logits
    logits = out.logits  # (B, T, V)
    B, T, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Shift: labels are shifted by 1 in HuggingFace causal LM
    shift_labels = lbl_t[:, 1:].contiguous()
    shift_log_probs = log_probs[:, :-1, :].contiguous()
    shift_mask = mask_t[:, 1:].contiguous()

    per_token_nll = -shift_log_probs.gather(
        2, shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)

    # Mask padding and -100 labels
    valid = (shift_labels >= 0) & (shift_mask == 1)
    per_token_nll = per_token_nll * valid.float()

    # Decompose by position within 3-token groups
    # UNVERIFIED: position 0=time, 1=duration, 2=note
    results = {"total": total_nll}
    for pos_name, pos_offset in [("time", 0), ("duration", 1), ("note", 2)]:
        mask_pos = torch.zeros_like(valid)
        for i in range(pos_offset, T - 1, 3):
            mask_pos[:, i] = True
        mask_pos = mask_pos & valid
        n = mask_pos.float().sum()
        if n > 0:
            results[pos_name] = float((per_token_nll * mask_pos.float()).sum() / n)
        else:
            results[pos_name] = float("nan")

    return results


def score_windows(
    model, windows: list[dict], control_offset: int, seq_len: int, device: str
) -> list[dict[str, float]]:
    """Score a list of manifest windows under the given model."""
    from model.training.train_lora import _tokenise_window
    import random
    rng = random.Random(42)
    scores = []
    for row in windows:
        try:
            # Share the training tokenization format, including its AUTOREGRESS
            # prefix and masked prefix label, for comparable teacher-forced NLL.
            ids, lbls, mask = _tokenise_window(
                row, control_offset, seq_len,
                anticipatory_prob=0.0, rng=rng, shift=0
            )
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            lbl_t = torch.tensor([lbls], dtype=torch.long, device=device)
            msk_t = torch.tensor([mask], dtype=torch.long, device=device)
            s = _nll_decomposed(model, ids_t, lbl_t, msk_t)
            scores.append(s)
        except Exception as e:
            print(f"  skip {row.get('window_id', '?')}: {e}")
    return scores


def mean_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: float(sum(s[k] for s in scores if not (s[k] != s[k])) /
                     max(1, sum(1 for s in scores if not (s[k] != s[k]))))
            for k in keys}


def main(config_path: str) -> None:
    import pandas as pd
    import anticipation.vocab as acfg
    import random
    from model.training.train_lora import load_windows
    from model.generation.generate import load_clean_base

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    manifest_path = str(Path(cfg["data"]["output_dir"]) / "manifest.csv")
    drive_dir = cfg["training"]["drive_dir"]
    model_name = cfg["model"]["name"]
    seq_len = cfg["training"]["seq_len"]
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    control_offset = acfg.CONTROL_OFFSET

    # Load test windows
    rng = random.Random(42)
    baroque_windows = load_windows(manifest_path, "baroque", "test")
    romantic_windows = load_windows(manifest_path, "romantic", "test")
    baroque_test = rng.sample(baroque_windows, min(20, len(baroque_windows)))
    romantic_test = rng.sample(romantic_windows, min(20, len(romantic_windows)))
    print(f"Baroque test: {len(baroque_test)}, Romantic test: {len(romantic_test)}")

    matrix = {}

    # Score under base model
    print("\nScoring under base model...")
    base = load_clean_base(model_name, device)
    base.eval()
    matrix["base_baroque"] = mean_scores(
        score_windows(base, baroque_test, control_offset, seq_len, device)
    )
    matrix["base_romantic"] = mean_scores(
        score_windows(base, romantic_test, control_offset, seq_len, device)
    )
    del base

    # Score under each adapter
    for style in ("baroque", "romantic"):
        adapter_path = str(Path(drive_dir) / f"adapter_{style}" / "final")
        if not Path(adapter_path).exists():
            print(f"  adapter_{style} not found, skipping")
            continue
        print(f"\nScoring under adapter_{style}...")
        from peft import PeftModel
        fresh = load_clean_base(model_name, device)
        model = PeftModel.from_pretrained(fresh, adapter_path)
        model.eval()
        matrix[f"adapter_{style}_baroque"] = mean_scores(
            score_windows(model, baroque_test, control_offset, seq_len, device)
        )
        matrix[f"adapter_{style}_romantic"] = mean_scores(
            score_windows(model, romantic_test, control_offset, seq_len, device)
        )
        del model, fresh

    # Print matrix
    print("\n=== NLL Matrix (total NLL) ===")
    print(f"{'':30s} {'baroque_test':>14} {'romantic_test':>14}")
    for model_key in ["base", "adapter_baroque", "adapter_romantic"]:
        b = matrix.get(f"{model_key}_baroque", {}).get("total", float("nan"))
        r = matrix.get(f"{model_key}_romantic", {}).get("total", float("nan"))
        print(f"  {model_key:28s} {b:14.4f} {r:14.4f}")

    print("\n=== NLL Decomposition (time / duration / note) ===")
    for key, scores in matrix.items():
        print(f"  {key}: " + " ".join(
            f"{k}={v:.4f}" for k, v in scores.items()
        ))

    with open(results_dir / "nll_matrix.json", "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"\nSaved to {results_dir}/nll_matrix.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
