"""
scripts/build_demo_index.py

Generates view/public/demo/index.json from:
  - results/generation_log.csv
  - results/evaluation_results.csv  (if present)
  - results/evaluation_summary.json (if present)
  - demo/                            (audio files copied to view/public/demo/)

The JSON is consumed by the React results browser (view/).

Usage:
    python scripts/build_demo_index.py --config config.yaml
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


LAYER_LABELS = {
    "base_amt": "Pretrained AMT",
    "layer_a_scripted": "Scripted — not a learned model",
    "layer_c_lora": "Learned — LoRA adapter",
}


def main(config_path: str) -> None:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["evaluation"]["results_dir"])
    wav_dir = results_dir / "wav"
    out_dir = Path("view/public/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load generation log
    gen_log_path = results_dir / "generation_log.csv"
    if not gen_log_path.exists():
        print(f"generation_log.csv not found — writing empty index")
        _write_empty_index(out_dir)
        return

    import csv
    gen_rows = []
    with open(gen_log_path) as f:
        for row in csv.DictReader(f):
            if row["status"] in ("ok", "partial"):
                gen_rows.append(row)

    # Load evaluation results if available
    eval_path = results_dir / "evaluation_results.csv"
    eval_by_file: dict[str, dict] = {}
    if eval_path.exists():
        with open(eval_path) as f:
            for row in csv.DictReader(f):
                eval_by_file[row["file"]] = row

    # Load summary if available
    summary_path = results_dir / "evaluation_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    # Group by melody_id
    melodies: dict[str, dict] = {}
    for row in gen_rows:
        mel_id = row["melody_id"]
        if mel_id not in melodies:
            melodies[mel_id] = {"melody_id": mel_id, "outputs": []}

        stem = row["file"]
        wav_src = wav_dir / f"{stem}.wav"
        wav_dst = out_dir / f"{stem}.wav"

        # Copy WAV to public/demo/
        audio_url = None
        if wav_src.exists():
            shutil.copy2(str(wav_src), str(wav_dst))
            audio_url = f"/demo/{stem}.wav"

        # Copy melody WAV if present
        mel_wav_src = wav_dir / f"melody_{mel_id}.wav"
        mel_wav_dst = out_dir / f"melody_{mel_id}.wav"
        melody_url = None
        if mel_wav_src.exists():
            shutil.copy2(str(mel_wav_src), str(mel_wav_dst))
            melody_url = f"/demo/melody_{mel_id}.wav"
        melodies[mel_id]["melody_audio"] = melody_url

        metrics = {}
        if stem in eval_by_file:
            er = eval_by_file[stem]
            for key in ("clf_prob_baroque", "clf_prob_romantic",
                        "skyline_dom", "clash_rate", "pc_entropy", "fourgram_rep"):
                try:
                    metrics[key] = float(er[key])
                except (KeyError, ValueError):
                    pass

        melodies[mel_id]["outputs"].append({
            "file": stem,
            "layer": row["layer"],
            "layer_label": LAYER_LABELS.get(row["layer"], row["layer"]),
            "style": row["style"],
            "audio_url": audio_url,
            "metrics": metrics,
        })

    index = {
        "melodies": list(melodies.values()),
        "summary": summary,
        "layers": [
            {"id": k, "label": v} for k, v in LAYER_LABELS.items()
        ],
    }

    out_path = out_dir / "index.json"
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {out_path} ({len(melodies)} melodies, {len(gen_rows)} outputs)")


def _write_empty_index(out_dir: Path) -> None:
    index = {
        "melodies": [],
        "summary": {},
        "layers": [{"id": k, "label": v} for k, v in LAYER_LABELS.items()],
        "_empty": True,
    }
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print("Wrote empty index.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
