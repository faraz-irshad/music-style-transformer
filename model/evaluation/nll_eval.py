"""
Held-out NLL evaluation for unconditional piano modelling and conditioned
accompaniment modelling. Metrics use the aligned-v1 preprocessing objective;
they are not comparable to historical prefix-truncated NLL files.

Run on Colab after training. No model is downloaded by importing this module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


def _nll_decomposed(
    model, ids_t: torch.Tensor, lbl_t: torch.Tensor, mask_t: torch.Tensor
) -> dict[str, float | int]:
    """Return causal teacher-forced NLL and event-field decompositions."""
    from torch.cuda.amp import autocast

    with torch.no_grad(), autocast():
        out = model(input_ids=ids_t, attention_mask=mask_t, labels=lbl_t)

    logits = out.logits
    _, length, _ = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    shift_labels = lbl_t[:, 1:].contiguous()
    valid = (shift_labels >= 0) & (mask_t[:, 1:] == 1)
    per_token_nll = -log_probs.gather(
        2, shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    per_token_nll = per_token_nll * valid.float()
    results: dict[str, float | int] = {
        "total": float(out.loss.item()),
        "valid_tokens": int(valid.sum().item()),
    }

    # The mode token occupies position zero; shifted labels then follow the
    # source event grammar: time, duration, pitch, repeated.
    for metric, phase in (("time", 0), ("duration", 1), ("note", 2)):
        phase_mask = torch.zeros_like(valid)
        phase_mask[:, phase:length - 1:3] = True
        phase_mask &= valid
        count = int(phase_mask.sum().item())
        results[f"{metric}_tokens"] = count
        results[metric] = (
            float((per_token_nll * phase_mask.float()).sum().item() / count)
            if count else float("nan")
        )
    return results


def score_windows(
    model,
    windows: list[dict],
    control_offset: int,
    seq_len: int,
    device: str,
    mode: str,
) -> list[dict[str, float | int]]:
    """Score all context-aligned examples, preserving the selected objective."""
    import random
    from model.training.train_lora import _tokenise_window

    rng = random.Random(42)
    scores = []
    for row in windows:
        try:
            examples = _tokenise_window(
                row,
                control_offset,
                seq_len,
                anticipatory_prob=0.0,
                rng=rng,
                shift=0,
                mode=mode,
            )
            for ids, labels, attention in examples:
                ids_t = torch.tensor([ids], dtype=torch.long, device=device)
                labels_t = torch.tensor([labels], dtype=torch.long, device=device)
                mask_t = torch.tensor([attention], dtype=torch.long, device=device)
                score = _nll_decomposed(model, ids_t, labels_t, mask_t)
                if score["valid_tokens"]:
                    scores.append(score)
        except Exception:
            import traceback
            print(
                f"  skip {row.get('window_id', '?')} ({mode}):\n"
                + traceback.format_exc()
            )
    return scores


def mean_scores(
    scores: list[dict[str, float | int]],
) -> dict[str, float | int]:
    """Aggregate unequal examples by their number of supervised tokens."""
    if not scores:
        return {}
    result: dict[str, float | int] = {
        "valid_tokens": sum(int(score["valid_tokens"]) for score in scores)
    }
    for metric, count_key in (
        ("total", "valid_tokens"),
        ("time", "time_tokens"),
        ("duration", "duration_tokens"),
        ("note", "note_tokens"),
    ):
        count = sum(int(score.get(count_key, 0)) for score in scores)
        result[f"{metric}_tokens"] = count
        result[metric] = (
            sum(float(score[metric]) * int(score[count_key]) for score in scores
                if int(score.get(count_key, 0)) > 0) / count
            if count else float("nan")
        )
    return result


def _adapter_path(cfg: dict, style: str) -> Path:
    from model.generation.adapter_paths import resolve_adapter_path

    drive_dir = Path(cfg["training"]["drive_dir"])
    experiment_root = Path(
        cfg["training"].get("experiment_dir")
        or drive_dir / "experiments" / cfg["training"].get(
            "experiment_name", "neural_alignment_v1"
        )
    )
    return resolve_adapter_path(drive_dir, experiment_root, style)


def main(config_path: str) -> None:
    import anticipation.vocab as acfg
    from model.training.train_lora import load_windows
    from model.generation.generate import load_clean_base

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    manifest_path = str(Path(cfg["data"]["output_dir"]) / "manifest.csv")
    model_name = cfg["model"]["name"]
    seq_len = cfg["training"]["seq_len"]
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "nll_matrix_alignment_v1.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    control_offset = acfg.CONTROL_OFFSET
    limit = cfg["evaluation"].get("nll_test_max_windows")
    windows_by_style = {}
    for style in ("baroque", "romantic"):
        windows = sorted(
            load_windows(manifest_path, style, "test"),
            key=lambda row: (str(row.get("window_id", "")), str(row["midi_path"])),
        )
        windows_by_style[style] = windows if limit is None else windows[:int(limit)]
    print("Test windows: " + json.dumps({
        style: len(windows) for style, windows in windows_by_style.items()
    }))

    matrix = {}
    base = load_clean_base(model_name, device)
    base.eval()
    for style, windows in windows_by_style.items():
        for mode in ("unconditional", "conditioned"):
            matrix[f"base_{style}_{mode}"] = mean_scores(
                score_windows(
                    base, windows, control_offset, seq_len, device, mode
                )
            )
    del base

    for adapter_style in ("baroque", "romantic"):
        adapter_path = _adapter_path(cfg, adapter_style)
        if not adapter_path.exists():
            print(f"  adapter_{adapter_style} not found at {adapter_path}; skipping")
            continue
        from peft import PeftModel

        fresh = load_clean_base(model_name, device)
        model = PeftModel.from_pretrained(fresh, str(adapter_path))
        model.eval()
        for style, windows in windows_by_style.items():
            for mode in ("unconditional", "conditioned"):
                matrix[f"adapter_{adapter_style}_{style}_{mode}"] = mean_scores(
                    score_windows(
                        model, windows, control_offset, seq_len, device, mode
                    )
                )
        del model, fresh

    for key, scores in matrix.items():
        print(f"{key}: {scores}")
    output_path.write_text(json.dumps({
        "objective_version": "aligned-v1-unconditional-and-conditioned",
        "aggregation": "supervised-token-weighted",
        "validation_window_limit": limit,
        "results": matrix,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
