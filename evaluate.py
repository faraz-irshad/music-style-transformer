"""
evaluate.py

Applies the classifier and all metrics to generated outputs.
Reads metadata from results/generation_log.csv (not filename parsing).
Uses the saved melody_{mel_id}.mid for melody/accompaniment separation.

Produces:
  results/evaluation_results.csv
  results/evaluation_summary.json
  results/figures/clf_probs.png

Run after generate.py has completed.
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Note loading
# ---------------------------------------------------------------------------

def load_notes(midi_path: str) -> list:
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


def split_by_known_melody(
    all_notes: list, melody_notes: list
) -> tuple[list, list]:
    """
    Split notes into melody and accompaniment using the KNOWN input melody
    (not a skyline of the output, which misclassifies arpeggios above the tune).
    """
    mel_set = {(round(n.start, 4), n.pitch) for n in melody_notes}
    melody = [n for n in all_notes if (round(n.start, 4), n.pitch) in mel_set]
    accompaniment = [n for n in all_notes if (round(n.start, 4), n.pitch) not in mel_set]
    return melody, accompaniment


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float], stat_fn=np.mean, n: int = 1000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    stats = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ---------------------------------------------------------------------------
# Memorization: hash-indexed corpus n-gram check
# ---------------------------------------------------------------------------

def build_corpus_ngram_index(
    corpus_notes_list: list[list], n: int = 8
) -> set[tuple[int, ...]]:
    """
    Build a hash set of all pitch-interval n-grams from the corpus.
    Runs in O(total_notes) rather than O(n_generated * n_corpus).
    """
    index: set[tuple[int, ...]] = set()
    for notes in corpus_notes_list:
        pitches = [note.pitch for note in sorted(notes, key=lambda x: x.start)]
        ivs = tuple(pitches[i+1] - pitches[i] for i in range(len(pitches)-1))
        for i in range(len(ivs) - n + 1):
            index.add(ivs[i:i+n])
    return index


def longest_match_against_index(
    gen_notes: list, index: set[tuple[int, ...]], max_n: int = 20
) -> int:
    """Find the longest n-gram in gen_notes that appears in the corpus index."""
    pitches = [n.pitch for n in sorted(gen_notes, key=lambda x: x.start)]
    ivs = tuple(pitches[i+1] - pitches[i] for i in range(len(pitches)-1))
    for n in range(min(max_n, len(ivs)), 0, -1):
        for i in range(len(ivs) - n + 1):
            if ivs[i:i+n] in index:
                return n
    return 0


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> float:
    from scipy.stats import binom
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n = b + c
    if n == 0:
        return 1.0
    return min(float(binom.cdf(min(b, c), n, 0.5)) * 2, 1.0)


def wilcoxon_test(x: list[float], y: list[float]) -> float:
    from scipy.stats import wilcoxon
    if len(x) < 2:
        return 1.0
    try:
        _, p = wilcoxon(x, y)
        return float(p)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Distribution distances
# ---------------------------------------------------------------------------

def compute_jsd_wasserstein(
    gen_notes_list: list[list],
    real_notes_list: list[list],
) -> dict[str, float]:
    """
    Compute JSD and 1-Wasserstein between generated and real distributions
    for: key-normalised pitch class, chord spread, log-IOI ratio, polyphony.
    """
    from data.features import (
        Note, harmony_features, texture_features, rhythm_features,
        jensen_shannon_divergence, wasserstein1, _tonic_pc,
    )
    import numpy as np

    def pc_hist(notes_list):
        hists = []
        for notes in notes_list:
            if not notes:
                continue
            tonic = _tonic_pc(notes)
            counts = np.zeros(12)
            for n in notes:
                counts[(n.pitch - tonic) % 12] += 1
            s = counts.sum()
            if s > 0:
                hists.append(counts / s)
        return np.array(hists) if hists else np.zeros((1, 12))

    def spread_samples(notes_list):
        vals = []
        for notes in notes_list:
            tf = texture_features(notes)
            vals.append(tf["mean_chord_spread"])
        return np.array(vals)

    def ioi_samples(notes_list):
        vals = []
        for notes in notes_list:
            rf = rhythm_features(notes)
            vals.append(rf["ioi_cv"])
        return np.array(vals)

    def poly_samples(notes_list):
        vals = []
        for notes in notes_list:
            tf = texture_features(notes)
            vals.append(tf["mean_polyphony"])
        return np.array(vals)

    results = {}

    # Pitch-class JSD (average over all pairs of histograms)
    gen_pc = pc_hist(gen_notes_list)
    real_pc = pc_hist(real_notes_list)
    mean_gen = gen_pc.mean(axis=0)
    mean_real = real_pc.mean(axis=0)
    results["jsd_pitch_class"] = jensen_shannon_divergence(mean_gen, mean_real)

    # 1-Wasserstein on scalar features
    for name, fn in [("chord_spread", spread_samples),
                     ("ioi_cv", ioi_samples),
                     ("polyphony", poly_samples)]:
        g = fn(gen_notes_list)
        r = fn(real_notes_list)
        if len(g) > 0 and len(r) > 0:
            results[f"wasserstein_{name}"] = wasserstein1(g, r)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    import pandas as pd

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["evaluation"]["results_dir"])
    midi_dir = results_dir / "midi"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_n = cfg["evaluation"]["bootstrap_n"]

    # Load generation manifest
    gen_log_path = results_dir / "generation_log.csv"
    if not gen_log_path.exists():
        print(f"generation_log.csv not found at {gen_log_path}. Run generate.py first.")
        return
    gen_log = pd.read_csv(gen_log_path)
    gen_log = gen_log[gen_log["status"] == "ok"].reset_index(drop=True)

    # Load classifier (with class names stored inside)
    clf_path = results_dir / "classifier.pkl"
    if not clf_path.exists():
        print("Classifier not found. Run train_classifier.py first.")
        return
    with open(clf_path, "rb") as f:
        clf_bundle = pickle.load(f)
    clf = clf_bundle["model"]
    class_names = clf_bundle["class_names"]  # e.g. ["baroque", "romantic"]
    baroque_idx = class_names.index("baroque")
    romantic_idx = class_names.index("romantic")

    from data.features import (
        extract_all, Note,
        pitch_class_entropy, fourgram_repetition_rate,
        skyline_dominance, melody_clash_rate,
    )

    rows = []
    for _, log_row in gen_log.iterrows():
        stem = log_row["file"]
        mel_id = log_row["melody_id"]
        layer = log_row["layer"]
        style = log_row["style"]

        midi_path = midi_dir / f"{stem}.mid"
        melody_path = midi_dir / f"melody_{mel_id}.mid"

        if not midi_path.exists():
            continue

        all_notes = load_notes(str(midi_path))
        if not all_notes:
            continue

        # Use known melody for separation (not skyline of output)
        if melody_path.exists():
            melody_notes = load_notes(str(melody_path))
            melody, accompaniment = split_by_known_melody(all_notes, melody_notes)
        else:
            from data.skyline import skyline, Note as SNote
            snotes = [SNote(n.start, n.end, n.pitch) for n in all_notes]
            mel = skyline(snotes)
            mel_set = {(n.start, n.pitch) for n in mel}
            melody = [n for n in all_notes if (n.start, n.pitch) in mel_set]
            accompaniment = [n for n in all_notes if (n.start, n.pitch) not in mel_set]

        if not accompaniment:
            accompaniment = all_notes

        feats = extract_all(accompaniment, melody_notes=melody, block="all")
        clf_pred_idx = clf.predict([feats])[0]
        clf_prob = clf.predict_proba([feats])[0]
        clf_pred_name = class_names[clf_pred_idx]

        # base_amt has style=none — don't score as correct/incorrect
        if style in ("baroque", "romantic"):
            clf_correct = clf_pred_name == style
        else:
            clf_correct = None

        row = {
            "file": stem,
            "layer": layer,
            "mel_id": mel_id,
            "style": style,
            "clf_pred": clf_pred_name,
            "clf_prob_baroque": float(clf_prob[baroque_idx]),
            "clf_prob_romantic": float(clf_prob[romantic_idx]),
            "clf_correct": clf_correct,
            "pc_entropy": pitch_class_entropy(accompaniment),
            "fourgram_rep": fourgram_repetition_rate(accompaniment),
            "skyline_dom": skyline_dominance(melody, accompaniment) if melody else 1.0,
            "clash_rate": melody_clash_rate(melody, accompaniment) if melody else 0.0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "evaluation_results.csv", index=False)
    print(f"Saved {len(df)} rows to {results_dir}/evaluation_results.csv")

    # ------------------------------------------------------------------
    # Per-layer accuracy (only for layers with a style label)
    # ------------------------------------------------------------------
    print("\n--- Classifier accuracy by layer (styled outputs only) ---")
    styled = df[df["clf_correct"].notna()].copy()
    styled["clf_correct"] = styled["clf_correct"].astype(bool)
    summary_per_layer = {}
    for layer in styled["layer"].unique():
        sub = styled[styled["layer"] == layer]
        acc = sub["clf_correct"].mean()
        k, n = int(sub["clf_correct"].sum()), len(sub)
        lo, hi = wilson_ci(k, n)
        bs_lo, bs_hi = bootstrap_ci(list(sub["clf_correct"].astype(float)), n=bootstrap_n)
        print(f"  {layer:30s}: acc={acc:.3f} Wilson=[{lo:.3f},{hi:.3f}] "
              f"Bootstrap=[{bs_lo:.3f},{bs_hi:.3f}] n={n}")
        summary_per_layer[layer] = {"accuracy": acc, "wilson_lo": lo, "wilson_hi": hi,
                                    "bootstrap_lo": bs_lo, "bootstrap_hi": bs_hi, "n": n}

    # ------------------------------------------------------------------
    # Paired tests: base_amt vs layer_c_lora (on style probability)
    # ------------------------------------------------------------------
    print("\n--- Paired tests: base_amt vs layer_c_lora ---")
    paired_results = {}
    for style in ("baroque", "romantic"):
        base = df[(df["layer"] == "base_amt")].sort_values("mel_id")
        lc = df[(df["layer"] == "layer_c_lora") & (df["style"] == style)].sort_values("mel_id")
        common = set(base["mel_id"]) & set(lc["mel_id"])
        if len(common) < 2:
            print(f"  {style}: not enough paired samples")
            continue
        base_c = base[base["mel_id"].isin(common)].sort_values("mel_id")
        lc_c = lc[lc["mel_id"].isin(common)].sort_values("mel_id")
        # base_amt has no style label so we can't do McNemar on correctness;
        # use Wilcoxon on style probability shift
        p_wx = wilcoxon_test(
            list(base_c[f"clf_prob_{style}"]),
            list(lc_c[f"clf_prob_{style}"]),
        )
        mean_shift = lc_c[f"clf_prob_{style}"].mean() - base_c[f"clf_prob_{style}"].mean()
        print(f"  {style}: Wilcoxon p={p_wx:.4f}  mean prob shift={mean_shift:+.3f} (n={len(common)})")
        paired_results[style] = {"wilcoxon_p": p_wx, "mean_prob_shift": mean_shift}

    # McNemar between layer_a_scripted and layer_c_lora (both have style labels)
    print("\n--- McNemar: layer_a_scripted vs layer_c_lora ---")
    for style in ("baroque", "romantic"):
        la = styled[(styled["layer"] == "layer_a_scripted") & (styled["style"] == style)].sort_values("mel_id")
        lc = styled[(styled["layer"] == "layer_c_lora") & (styled["style"] == style)].sort_values("mel_id")
        common = set(la["mel_id"]) & set(lc["mel_id"])
        if len(common) < 2:
            continue
        la_c = la[la["mel_id"].isin(common)].sort_values("mel_id")
        lc_c = lc[lc["mel_id"].isin(common)].sort_values("mel_id")
        p_mc = mcnemar_test(list(la_c["clf_correct"]), list(lc_c["clf_correct"]))
        print(f"  {style}: McNemar p={p_mc:.4f} (n={len(common)})")

    # ------------------------------------------------------------------
    # JSD / Wasserstein vs real MAESTRO test distribution
    # ------------------------------------------------------------------
    print("\n--- Distribution distances vs real MAESTRO test set ---")
    manifest_path = Path(cfg["data"]["output_dir"]) / "manifest.csv"
    dist_results = {}
    if manifest_path.exists():
        import pandas as _pd
        manifest = _pd.read_csv(manifest_path)
        for style in ("baroque", "romantic"):
            real_rows = manifest[(manifest["style"] == style) & (manifest["split"] == "test")]
            real_notes_list = []
            for _, rrow in real_rows.iterrows():
                try:
                    notes = load_notes(rrow["midi_path"])
                    real_notes_list.append(notes)
                except Exception:
                    pass
            if not real_notes_list:
                continue
            for layer in df["layer"].unique():
                sub = df[(df["layer"] == layer) & (df["style"].isin([style, "none"]))]
                gen_notes_list = []
                for _, grow in sub.iterrows():
                    mp = midi_dir / f"{grow['file']}.mid"
                    if mp.exists():
                        try:
                            gen_notes_list.append(load_notes(str(mp)))
                        except Exception:
                            pass
                if not gen_notes_list:
                    continue
                dists = compute_jsd_wasserstein(gen_notes_list, real_notes_list)
                key = f"{layer}_{style}"
                dist_results[key] = dists
                print(f"  {layer:30s} {style}: " +
                      " ".join(f"{k}={v:.4f}" for k, v in dists.items()))
    else:
        print("  manifest.csv not found — skipping distribution distances")

    # ------------------------------------------------------------------
    # Memorization check against corpus
    # ------------------------------------------------------------------
    print("\n--- Memorization check ---")
    mem_results = {}
    if manifest_path.exists():
        import pandas as _pd
        manifest = _pd.read_csv(manifest_path)
        corpus_notes_list = []
        for _, rrow in manifest[manifest["split"] == "train"].iterrows():
            try:
                corpus_notes_list.append(load_notes(rrow["midi_path"]))
            except Exception:
                pass
        if corpus_notes_list:
            print(f"  Building corpus n-gram index ({len(corpus_notes_list)} windows)...")
            corpus_index = build_corpus_ngram_index(corpus_notes_list, n=8)
            print(f"  Index size: {len(corpus_index)} 8-grams")
            gen_files = df[df["layer"] != "layer_a_scripted"]
            matches = []
            for _, grow in gen_files.iterrows():
                mp = midi_dir / f"{grow['file']}.mid"
                if not mp.exists():
                    continue
                try:
                    gen_notes = load_notes(str(mp))
                    m = longest_match_against_index(gen_notes, corpus_index)
                    matches.append(m)
                except Exception:
                    pass
            if matches:
                mem_results = {
                    "max_shared_ngram": int(max(matches)),
                    "p95_shared_ngram": float(np.percentile(matches, 95)),
                    "n_clips": len(matches),
                }
                print(f"  Max shared n-gram: {mem_results['max_shared_ngram']}")
                print(f"  95th percentile: {mem_results['p95_shared_ngram']:.1f}")
        else:
            print("  No corpus windows loaded")
    else:
        print("  manifest.csv not found — skipping corpus memorization check")

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary = {
        "n_files": len(df),
        "per_layer_accuracy": summary_per_layer,
        "paired_tests": paired_results,
        "distribution_distances": dist_results,
        "memorization": mem_results,
    }
    with open(results_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {results_dir}/evaluation_summary.json")

    _make_plots(df, fig_dir)


def _make_plots(df, fig_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric in zip(axes, ["clf_prob_baroque", "clf_prob_romantic"]):
            for layer in df["layer"].unique():
                for style in ("baroque", "romantic", "none"):
                    sub = df[(df["layer"] == layer) & (df["style"] == style)][metric]
                    if len(sub) == 0:
                        continue
                    label = f"{layer}\n({style})"
                    ax.scatter([label] * len(sub), sub, alpha=0.5, s=20)
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
