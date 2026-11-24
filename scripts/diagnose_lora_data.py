#!/usr/bin/env python3
"""CPU-only preprocessing audit; reads MIDI but never loads model weights."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.training.sequence import CACHE_SCHEMA, SEQUENCE_POLICY, UnconsumedControlsError
from model.training.train_lora import (
    _cache_path,
    _load_performance_notes,
    _sequence_settings,
    _tokenise_window_with_stats,
    load_windows,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--style", choices=("baroque", "romantic"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    return parser.parse_args()


def _audit_mode(
    windows: list[dict],
    style: str,
    split: str,
    control_offset: int,
    seq_len: int,
    seed: int,
    context_groups: int,
    mode: str,
) -> dict:
    rng = random.Random(seed)
    performance_cache = {}
    window_stats = []
    invalid = []
    for row in windows:
        try:
            path = str(row["midi_path"])
            if path not in performance_cache:
                performance_cache[path] = _load_performance_notes(path)
            _examples, stats = _tokenise_window_with_stats(
                row,
                control_offset,
                seq_len,
                anticipatory_prob=0.0,
                rng=rng,
                performance_notes=performance_cache[path],
                mode=mode,
                context_groups=context_groups,
            )
            window_stats.append(stats)
        except Exception as exc:
            invalid.append({
                "window_id": row.get("window_id", ""),
                "error": f"{type(exc).__name__}: {exc}",
                "unconsumed_controls": (
                    exc.count if isinstance(exc, UnconsumedControlsError) else 0
                ),
            })

    cache_identity = _cache_path(
        Path(".diagnostic-cache-identity"),
        style,
        split,
        seed,
        windows,
        control_offset,
        seq_len,
        mode=mode,
        anticipatory_prob=0.0,
        context_groups=context_groups,
    ).stem
    return {
        "mode": mode,
        "source_windows": len(windows),
        "valid_windows": len(window_stats),
        "produced_examples": sum(s.get("produced_examples", 0) for s in window_stats),
        "original_time_coverage_sec": sum(
            s.get("original_time_coverage_sec", 0.0) for s in window_stats
        ),
        "retained_time_coverage_sec": sum(
            max(0.0, s["retained_time_end_sec"] - s["retained_time_start_sec"])
            for s in window_stats
            if s.get("retained_time_start_sec") is not None
            and s.get("retained_time_end_sec") is not None
        ),
        "supervised_accompaniment_tokens": sum(
            s.get("supervised_accompaniment_tokens", 0) for s in window_stats
        ),
        "supervised_rest_tokens": sum(
            s.get("supervised_rest_tokens", 0) for s in window_stats
        ),
        "supervised_rest_groups": sum(
            s.get("supervised_rest_groups", 0) for s in window_stats
        ),
        "control_groups": sum(s.get("control_groups", 0) for s in window_stats),
        "unconsumed_controls": sum(
            s.get("unconsumed_controls", 0) for s in window_stats
        ) + sum(entry["unconsumed_controls"] for entry in invalid),
        "dropped_controls": sum(s.get("dropped_controls", 0) for s in window_stats),
        "chunked_windows": sum(bool(s.get("chunked")) for s in window_stats),
        "truncated_windows": sum(
            int(s.get("truncated_groups", 0) > 0) for s in window_stats
        ),
        "empty_windows": sum(bool(s.get("empty")) for s in window_stats),
        "all_masked_examples": sum(
            s.get("all_masked_examples", 0) for s in window_stats
        ),
        "invalid_windows": len(invalid),
        "invalid_examples": invalid[:20],
        "cache_identity": cache_identity,
    }


def main() -> int:
    args = _arguments()
    import anticipation.vocab as vocab
    import yaml
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    manifest = str(Path(cfg["data"]["output_dir"]) / "manifest.csv")
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    settings = _sequence_settings(vocab.CONTROL_OFFSET)
    report = {
        "cache_schema": CACHE_SCHEMA,
        "sequence_policy": SEQUENCE_POLICY,
        "upstream_anticipation_commit": "af37397922665a0fb8d474d7988b0f3755a38d45",
        "settings": {
            "sequence_length": cfg["training"]["seq_len"],
            "context_groups": cfg["training"].get("context_groups", 170),
            "anticipation_delta_ticks": settings.anticipation_delta,
            "rest_padding_density_ticks": settings.padding_density,
            "time_resolution": settings.time_resolution,
            "validation_limit": cfg["training"].get("validation_max_examples"),
            "seed": cfg["training"]["random_seed"],
        },
        "splits": {},
    }
    for split in splits:
        windows = load_windows(manifest, args.style, split)
        report["splits"][split] = {
            mode: _audit_mode(
                windows,
                args.style,
                split,
                vocab.CONTROL_OFFSET,
                cfg["training"]["seq_len"],
                cfg["training"]["random_seed"],
                cfg["training"].get("context_groups", 170),
                mode,
            )
            for mode in ("unconditional", "conditioned")
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if any(
        result["invalid_windows"]
        for split in report["splits"].values()
        for result in split.values()
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
