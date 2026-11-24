"""
Held-out NLL evaluation for unconditional piano modelling and conditioned
accompaniment modelling. Metrics use the aligned-v1 preprocessing objective;
they are not comparable to historical prefix-truncated NLL files.

Run on Colab after training. No model is downloaded by importing this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from model.training.sequence import (
    CACHE_SCHEMA,
    INVALID_WINDOW_POLICY,
    SEQUENCE_POLICY,
)


def _nll_decomposed(
    model, ids_t: torch.Tensor, lbl_t: torch.Tensor, mask_t: torch.Tensor
) -> dict[str, float | int]:
    """Return causal teacher-forced NLL and event-field decompositions."""
    import torch
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
    valid_count = int(valid.sum().item())
    results: dict[str, float | int] = {
        "total": (
            float(per_token_nll.sum().item() / valid_count)
            if valid_count else float("nan")
        ),
        "valid_tokens": valid_count,
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


class IncompleteComparisonError(RuntimeError):
    pass


def prepare_examples(
    windows: list[dict],
    control_offset: int,
    seq_len: int,
    mode: str,
    context_groups: int = 170,
) -> list[dict]:
    """Prepare a stable, shared example set once for every model comparison."""
    import random
    from model.training.train_lora import (
        _load_performance_notes,
        _tokenise_window,
    )

    rng = random.Random(42)
    performance_cache = {}
    prepared = []
    for row in windows:
        path = str(row.get("midi_path", "<missing>"))
        try:
            path = str(row["midi_path"])
            if path not in performance_cache:
                performance_cache[path] = _load_performance_notes(path)
            examples = _tokenise_window(
                row,
                control_offset,
                seq_len,
                anticipatory_prob=0.0,
                rng=rng,
                shift=0,
                performance_notes=performance_cache[path],
                mode=mode,
                context_groups=context_groups,
            )
        except Exception as exc:
            raise IncompleteComparisonError(
                f"could not prepare {row.get('window_id', path)} "
                f"for objective {mode}: {type(exc).__name__}: {exc}"
            ) from exc
        if not examples:
            raise IncompleteComparisonError(
                f"window {row.get('window_id', path)} produced no examples "
                f"for objective {mode}"
            )
        window_id = str(row.get("window_id") or (
            f"{path}:{row.get('start_sec')}:{row.get('end_sec')}"
        ))
        window_identity = json.dumps(
            [
                window_id,
                path,
                float(row["start_sec"]),
                float(row["end_sec"]),
            ],
            separators=(",", ":"),
        )
        stable_window_id = (
            f"{window_id}::{hashlib.sha256(window_identity.encode()).hexdigest()[:12]}"
        )
        for chunk_id, (ids, labels, attention) in enumerate(examples):
            valid_tokens = sum(
                label >= 0 and mask
                for label, mask in zip(labels[1:], attention[1:])
            )
            if not valid_tokens:
                raise IncompleteComparisonError(
                    f"{window_id} chunk {chunk_id} has no supervised tokens"
                )
            prepared.append({
                "example_id": f"{stable_window_id}::chunk-{chunk_id:05d}",
                "window_id": window_id,
                "chunk_id": chunk_id,
                "ids": ids,
                "labels": labels,
                "attention": attention,
                "valid_tokens": valid_tokens,
            })
    if not prepared:
        raise IncompleteComparisonError(
            f"no examples prepared for objective {mode}"
        )
    return prepared


def score_prepared_examples(
    model,
    examples: list[dict],
    device: str,
    model_label: str,
) -> list[dict[str, float | int | str]]:
    """Score every prepared example or reject the model comparison."""
    import torch
    scores = []
    for example in examples:
        example_id = example["example_id"]
        try:
            ids_t = torch.tensor([example["ids"]], dtype=torch.long, device=device)
            labels_t = torch.tensor(
                [example["labels"]], dtype=torch.long, device=device
            )
            mask_t = torch.tensor(
                [example["attention"]], dtype=torch.long, device=device
            )
            score = _nll_decomposed(model, ids_t, labels_t, mask_t)
            if score["valid_tokens"] != example["valid_tokens"]:
                raise ValueError(
                    f"token count changed from {example['valid_tokens']} "
                    f"to {score['valid_tokens']}"
                )
        except Exception as exc:
            raise IncompleteComparisonError(
                f"{model_label} failed on {example_id}; comparison incomplete: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        scores.append({
            "example_id": example_id,
            "window_id": example["window_id"],
            "chunk_id": example["chunk_id"],
            **score,
        })
    expected_ids = [example["example_id"] for example in examples]
    actual_ids = [score["example_id"] for score in scores]
    if actual_ids != expected_ids:
        raise IncompleteComparisonError(
            f"{model_label} score IDs do not match prepared comparison examples"
        )
    return scores


def require_matching_examples(
    reference: list[dict], comparison: list[dict], label: str
) -> None:
    reference_ids = [row["example_id"] for row in reference]
    comparison_ids = [row["example_id"] for row in comparison]
    if reference_ids != comparison_ids:
        raise IncompleteComparisonError(
            f"{label} has {len(comparison_ids)} scored examples; expected "
            f"{len(reference_ids)} with identical ordered IDs"
        )


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
    import torch
    import anticipation.vocab as acfg
    from model.training.train_lora import _sequence_settings, load_windows
    from model.generation.generate import load_clean_base

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    manifest_path = str(Path(cfg["data"]["output_dir"]) / "manifest.csv")
    model_name = cfg["model"]["name"]
    seq_len = cfg["training"]["seq_len"]
    context_groups = cfg["training"].get("context_groups", 170)
    invalid_window_policy = cfg["training"].get(
        "invalid_window_policy", INVALID_WINDOW_POLICY
    )
    if invalid_window_policy != INVALID_WINDOW_POLICY:
        raise ValueError(
            f"unsupported invalid_window_policy {invalid_window_policy!r}"
        )
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "nll_matrix_alignment_v2.json"
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

    prepared_by_objective = {}
    for style, windows in windows_by_style.items():
        for mode in ("unconditional", "conditioned"):
            prepared_by_objective[(style, mode)] = prepare_examples(
                windows, control_offset, seq_len, mode, context_groups
            )

    results = {}
    per_example_scores = {}
    base_scores = {}
    base = load_clean_base(model_name, device)
    base.eval()
    for style in windows_by_style:
        for mode in ("unconditional", "conditioned"):
            key = f"base_{style}_{mode}"
            scores = score_prepared_examples(
                base,
                prepared_by_objective[(style, mode)],
                device,
                model_label=key,
            )
            base_scores[(style, mode)] = scores
            results[key] = mean_scores(scores)
            per_example_scores[key] = scores
    del base

    model_paths = {"base": {"model_identifier": model_name, "adapter_path": None}}
    missing_adapters = []
    for adapter_style in ("baroque", "romantic"):
        adapter_path = _adapter_path(cfg, adapter_style)
        model_paths[f"adapter_{adapter_style}"] = {
            "model_identifier": model_name,
            "adapter_path": str(adapter_path),
        }
        if not adapter_path.exists():
            print(f"  adapter_{adapter_style} not found at {adapter_path}; skipping")
            missing_adapters.append(str(adapter_path))
            continue
        from peft import PeftModel

        fresh = load_clean_base(model_name, device)
        model = PeftModel.from_pretrained(fresh, str(adapter_path))
        model.eval()
        for style in windows_by_style:
            for mode in ("unconditional", "conditioned"):
                key = f"adapter_{adapter_style}_{style}_{mode}"
                scores = score_prepared_examples(
                    model,
                    prepared_by_objective[(style, mode)],
                    device,
                    model_label=key,
                )
                require_matching_examples(
                    base_scores[(style, mode)], scores, key
                )
                results[key] = mean_scores(scores)
                per_example_scores[key] = scores
        del model, fresh

    for key, aggregate in results.items():
        print(f"{key}: {aggregate}")
    comparison_status = (
        "incomplete_missing_adapters" if missing_adapters else "complete"
    )
    output_path.write_text(json.dumps({
        "objective_version": "aligned-v2-paired-unconditional-and-conditioned",
        "aggregation": "supervised-token-weighted",
        "validation_window_limit": limit,
        "model_identifier": model_name,
        "models": model_paths,
        "comparison_status": comparison_status,
        "missing_adapters": missing_adapters,
        "preprocessing": {
            "cache_schema": CACHE_SCHEMA,
            "sequence_policy": SEQUENCE_POLICY,
            "invalid_window_policy": invalid_window_policy,
            "sequence_length": seq_len,
            "context_groups": context_groups,
            "timing": _sequence_settings(control_offset).__dict__,
            "window_selection": "notes-fully-contained-in-manifest-window-v1",
            "pitch_shift": 0,
            "anticipatory_probability": 0.0,
            "seed": 42,
        },
        "example_counts": {
            f"{style}_{mode}": len(examples)
            for (style, mode), examples in prepared_by_objective.items()
        },
        "results": results,
        "per_example_scores": per_example_scores,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
