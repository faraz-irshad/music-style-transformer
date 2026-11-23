"""
evaluate.py

Applies the classifier and all metrics to generated outputs.
Produces results/*.csv and results/figures/*.png.

Run after generate.py has completed.
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Load generated MIDI
# ---------------------------------------------------------------------------

def load_notes_from_midi(midi_path: str) -> list:
    """Load all non-drum notes from a MIDI file."""
    import pretty_midi
    from data.features import Note
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append(Note(n.start, n.end, n.pitch, n.velocity))
    return notes


def split_melody_accompaniment(notes: list, melody_track_idx: int = 0) -> tuple[list, list]:
    """
    For Layer A outputs, track 0 is melody, track 1 is accompaniment.
    For AMT outputs, use skyline to separate.
    """
    from data.skyline import skyline, Note as SNote
    from data.features import Note as FNote
    snotes = [SNote(n.start, n.end, n.pitch) for n in notes]
    mel = skyline(snotes)
    mel_set = {(n.start, n.pitch) for n in mel}
    melody = [n for n in notes if (n.start, n.pitch) in mel_set]
    accompaniment = [n for n in notes if (n.start, n.pitch) not in mel_set]
    return melody, accompaniment


# ---------------------------------------------------------------------------
# Memorization check
# ---------------------------------------------------------------------------

def longest_shared_interval_ngram(
    gen_notes: list, corpus_notes: list, max_n: int = 20
) -> int:
    """
    Longest transposition-invariant pitch-interval n-gram shared between
    generated clip and corpus. Returns length of longest shared n-gram.
    """
    def to_intervals(notes):
        pitches = [n.pitch for n in sorted(notes, key=lambda n: n.start)]
        return tuple(pitches[i+1] - pitches[i] for i in range(len(pitches)-1))

    gen_iv = to_intervals(gen_notes)
    corp_iv = to_intervals(corpus_notes)
    if not gen_iv or not corp_iv:
        return 0

    best = 0
    for n in range(min(max_n, len(gen_iv), len(corp_iv)), 0, -1):
        gen_grams = {gen_iv[i:i+n] for i in range(len(gen_iv)-n+1)}
        corp_grams = {corp_iv[i:i+n] for i in range(len(corp_iv)-n+1)}
        if gen_grams & corp_grams:
            best = n
            break
    return best


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> float:
    """Exact McNemar test p-value (two-sided)."""
    from scipy.stats import binom
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n = b + c
    if n == 0:
        return 1.0
    p = float(binom.cdf(min(b, c), n, 0.5)) * 2
    return min(p, 1.0)


def wilcoxon_test(x: list[float], y: list[float]) -> float:
    """Paired Wilcoxon signed-rank test p-value."""
    from scipy.stats import wilcoxon
    if len(x) < 2:
        return 1.0
    try:
        _, p = wilcoxon(x, y)
        return float(p)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["evaluation"]["results_dir"])
    midi_dir = results_dir / "midi"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    classifier_path = results_dir / "classifier.pkl"
    if not classifier_path.exists():
        print("Classifier not found. Run train_classifier.py first.")
        return

    with open(classifier_path, "rb") as f:
        clf = pickle.load(f)

    from data.features import (
        extract_all, Note,
        pitch_class_entropy, fourgram_repetition_rate,
        skyline_dominance, melody_clash_rate,
        jensen_shannon_divergence, wasserstein1,
    )

    midi_files = sorted(midi_dir.glob("*.mid"))
    if not midi_files:
        print(f"No MIDI files in {midi_dir}")
        return

    rows = []
    for midi_path in midi_files:
        name = midi_path.stem
        parts = name.split("_")
        # Filename: {layer}_{mel_id}_{style}
        # layer may contain underscores, style is last, mel_id is second-to-last
        style = parts[-1]
        mel_id = parts[-2]
        layer = "_".join(parts[:-2])

        notes = load_notes_from_midi(str(midi_path))
        if not notes:
            continue
        melody, accompaniment = split_melody_accompaniment(notes)
        if not accompaniment:
            accompaniment = notes

        feats = extract_all(accompaniment, melody_notes=melody, block="all")
        clf_pred = clf.predict([feats])[0]
        clf_prob = clf.predict_proba([feats])[0]

        row = {
            "file": name,
            "layer": layer,
            "mel_id": mel_id,
            "style": style,
            "clf_pred": "baroque" if clf_pred == 0 else "romantic",
            "clf_prob_baroque": float(clf_prob[0]),
            "clf_prob_romantic": float(clf_prob[1]),
            "clf_correct": (clf_pred == 0) == (style == "baroque"),
            "pc_entropy": pitch_class_entropy(accompaniment),
            "fourgram_rep": fourgram_repetition_rate(accompaniment),
            "skyline_dom": skyline_dominance(melody, accompaniment) if melody else 1.0,
            "clash_rate": melody_clash_rate(melody, accompaniment) if melody else 0.0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "evaluation_results.csv", index=False)
    print(f"Saved {len(df)} rows to {results_dir}/evaluation_results.csv")

    # Per-layer accuracy
    print("\n--- Classifier accuracy by layer ---")
    for layer in df["layer"].unique():
        sub = df[df["layer"] == layer]
        acc = sub["clf_correct"].mean()
        k, n = sub["clf_correct"].sum(), len(sub)
        lo, hi = wilson_ci(int(k), n)
        print(f"  {layer:30s}: acc={acc:.3f} 95%CI=[{lo:.3f},{hi:.3f}] n={n}")

    # Paired tests: base_amt vs layer_c_lora
    for style in ("baroque", "romantic"):
        base = df[(df["layer"] == "base_amt") & (df["style"] == style)].sort_values("mel_id")
        lc = df[(df["layer"] == "layer_c_lora") & (df["style"] == style)].sort_values("mel_id")
        common = set(base["mel_id"]) & set(lc["mel_id"])
        if len(common) < 2:
            continue
        base_c = base[base["mel_id"].isin(common)].sort_values("mel_id")
        lc_c = lc[lc["mel_id"].isin(common)].sort_values("mel_id")
        p_mc = mcnemar_test(
            list(base_c["clf_correct"]), list(lc_c["clf_correct"])
        )
        p_wx = wilcoxon_test(
            list(base_c[f"clf_prob_{style}"]),
            list(lc_c[f"clf_prob_{style}"]),
        )
        print(f"\n  {style}: base_amt vs layer_c_lora (n={len(common)})")
        print(f"    McNemar p={p_mc:.4f}, Wilcoxon p={p_wx:.4f}")

    # JSD / Wasserstein between generated and real distributions
    print("\n--- Distribution distances (pc_entropy) ---")
    for style in ("baroque", "romantic"):
        for layer in df["layer"].unique():
            sub = df[(df["layer"] == layer) & (df["style"] == style)]["pc_entropy"].values
            if len(sub) < 2:
                continue
            print(f"  {layer} {style}: mean_pc_entropy={sub.mean():.3f}")

    # Memorization check (sample — full corpus check needs MAESTRO loaded)
    print("\n--- Memorization check (inter-file, generated only) ---")
    gen_files = [r for r in rows if r["layer"] != "layer_a_scripted"]
    max_shared = 0
    for i in range(len(gen_files)):
        for j in range(i + 1, len(gen_files)):
            ni = load_notes_from_midi(str(midi_dir / (gen_files[i]["file"] + ".mid")))
            nj = load_notes_from_midi(str(midi_dir / (gen_files[j]["file"] + ".mid")))
            shared = longest_shared_interval_ngram(ni, nj)
            max_shared = max(max_shared, shared)
    print(f"  Max shared interval n-gram (between generated clips): {max_shared}")
    print("  NOTE: Full memorization check against MAESTRO corpus requires corpus loaded.")

    # Save summary
    summary = {
        "n_files": len(df),
        "overall_accuracy": float(df["clf_correct"].mean()),
        "per_layer": {
            layer: float(df[df["layer"] == layer]["clf_correct"].mean())
            for layer in df["layer"].unique()
        },
        "max_shared_ngram_between_generated": max_shared,
    }
    with open(results_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {results_dir}/evaluation_summary.json")

    _make_plots(df, fig_dir)


def _make_plots(df: pd.DataFrame, fig_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, metric in zip(axes, ["clf_prob_baroque", "clf_prob_romantic"]):
            for layer in df["layer"].unique():
                for style in ("baroque", "romantic"):
                    sub = df[(df["layer"] == layer) & (df["style"] == style)][metric]
                    if len(sub) == 0:
                        continue
                    ax.scatter([f"{layer}\n{style}"] * len(sub), sub, alpha=0.5, s=20)
            ax.set_title(metric)
            ax.set_ylabel("probability")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(str(fig_dir / "clf_probs.png"), dpi=120)
        plt.close()
        print(f"Saved {fig_dir}/clf_probs.png")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
