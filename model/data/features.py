"""
data/features.py

Shortcut-resistant feature extraction for Baroque vs. Romantic classification.
Evaluates the ACCOMPANIMENT ONLY (melody stripped before calling these functions).

Three feature blocks:
  1. Harmony  — key-invariant pitch-class features
  2. Texture  — chord spread and polyphony
  3. Rhythm   — tempo-invariant IOI statistics

Deliberately excluded (shortcuts):
  - Raw BPM / absolute IOI histograms
  - Velocities
  - Absolute-key pitch-class histograms
  - Beat-grid syncopation (no reliable beat annotations in MAESTRO)
"""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Note container (reuse or import from skyline)
# ---------------------------------------------------------------------------

class Note:
    __slots__ = ("start", "end", "pitch", "velocity")

    def __init__(self, start: float, end: float, pitch: int, velocity: int = 64):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity


def notes_from_pretty_midi(pm, exclude_drums: bool = True) -> list[Note]:
    notes: list[Note] = []
    for inst in pm.instruments:
        if exclude_drums and inst.is_drum:
            continue
        for n in inst.notes:
            notes.append(Note(n.start, n.end, n.pitch, n.velocity))
    return sorted(notes, key=lambda n: n.start)


def split_skyline(notes: list[Note]) -> tuple[list[Note], list[Note]]:
    """Return skyline melody notes and the remaining accompaniment notes."""
    from model.data.skyline import Note as SkylineNote, skyline

    skyline_notes = skyline([
        SkylineNote(note.start, note.end, note.pitch) for note in notes
    ])
    melody_keys = {
        (note.start, note.end, note.pitch) for note in skyline_notes
    }
    melody = [
        note for note in notes
        if (note.start, note.end, note.pitch) in melody_keys
    ]
    accompaniment = [
        note for note in notes
        if (note.start, note.end, note.pitch) not in melody_keys
    ]
    return melody, accompaniment


# ---------------------------------------------------------------------------
# Block 1: Key-invariant harmony
# ---------------------------------------------------------------------------

def _tonic_pc(notes: list[Note]) -> int:
    """Estimate tonic pitch class via Krumhansl major profile."""
    profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
               2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    counts = [0.0] * 12
    for n in notes:
        counts[n.pitch % 12] += 1.0
    best_k, best_s = 0, -1e9
    for k in range(12):
        s = sum(counts[(k + i) % 12] * profile[i] for i in range(12))
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def harmony_features(notes: list[Note]) -> dict[str, float]:
    """
    Key-invariant harmony features.
    All pitch-class histograms are rotated so tonic = index 0.
    """
    if not notes:
        return {k: 0.0 for k in [
            "pc_entropy", "non_diatonic_ratio",
            "major_third_rate", "minor_third_rate",
            "major_sixth_rate", "minor_sixth_rate",
            "dom7_rate", "dim7_rate",
        ]}

    tonic = _tonic_pc(notes)
    major_scale = {0, 2, 4, 5, 7, 9, 11}

    counts = np.zeros(12)
    for n in notes:
        counts[(n.pitch - tonic) % 12] += 1.0
    counts /= counts.sum() + 1e-9

    pc_entropy = float(-np.sum(counts * np.log2(counts + 1e-9)))
    non_diatonic = float(sum(counts[i] for i in range(12) if i not in major_scale))

    # Vertical interval rates: look at simultaneous note pairs
    frame_sec = 0.05
    t_end = max(n.end for n in notes)
    frames = int(t_end / frame_sec) + 1
    intervals: list[int] = []
    for f in range(frames):
        t = f * frame_sec
        active = [n.pitch for n in notes if n.start <= t < n.end]
        active.sort()
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                intervals.append((active[j] - active[i]) % 12)

    total = len(intervals) + 1e-9
    major_third = sum(1 for iv in intervals if iv == 4) / total
    minor_third = sum(1 for iv in intervals if iv == 3) / total
    major_sixth = sum(1 for iv in intervals if iv == 9) / total
    minor_sixth = sum(1 for iv in intervals if iv == 8) / total
    dom7 = sum(1 for iv in intervals if iv == 10) / total
    dim7 = sum(1 for iv in intervals if iv == 9) / total  # enharmonic with maj6

    return {
        "pc_entropy": pc_entropy,
        "non_diatonic_ratio": non_diatonic,
        "major_third_rate": major_third,
        "minor_third_rate": minor_third,
        "major_sixth_rate": major_sixth,
        "minor_sixth_rate": minor_sixth,
        "dom7_rate": dom7,
        "dim7_rate": dim7,
    }


# ---------------------------------------------------------------------------
# Block 2: Texture
# ---------------------------------------------------------------------------

def texture_features(notes: list[Note], melody_notes: list[Note] | None = None) -> dict[str, float]:
    """
    Chord spread, polyphony, and fraction of accompaniment above melody.
    melody_notes: if provided, used to compute skyline_dominance and clash_rate.
    """
    if not notes:
        return {k: 0.0 for k in [
            "mean_chord_spread", "p90_chord_spread",
            "mean_polyphony", "frac_above_melody",
        ]}

    frame_sec = 0.05
    t_end = max(n.end for n in notes)
    frames = int(t_end / frame_sec) + 1

    spreads, polyphonies = [], []
    for f in range(frames):
        t = f * frame_sec
        active = [n.pitch for n in notes if n.start <= t < n.end]
        if active:
            spreads.append(max(active) - min(active))
            polyphonies.append(len(active))

    mean_spread = float(np.mean(spreads)) if spreads else 0.0
    p90_spread = float(np.percentile(spreads, 90)) if spreads else 0.0
    mean_poly = float(np.mean(polyphonies)) if polyphonies else 0.0

    frac_above = 0.0
    if melody_notes:
        above = sum(1 for n in notes
                    for m in melody_notes
                    if abs(n.start - m.start) < frame_sec and n.pitch > m.pitch)
        frac_above = above / (len(notes) + 1e-9)

    return {
        "mean_chord_spread": mean_spread,
        "p90_chord_spread": p90_spread,
        "mean_polyphony": mean_poly,
        "frac_above_melody": frac_above,
    }


# ---------------------------------------------------------------------------
# Block 3: Tempo-invariant rhythm and articulation
# ---------------------------------------------------------------------------

def rhythm_features(notes: list[Note]) -> dict[str, float]:
    """
    Coefficient of variation of IOIs, log-IOI ratio histogram (5 bins),
    and legato/overlap ratio. All tempo-invariant.
    """
    if len(notes) < 2:
        return {k: 0.0 for k in [
            "ioi_cv", "log_ioi_ratio_b0", "log_ioi_ratio_b1",
            "log_ioi_ratio_b2", "log_ioi_ratio_b3", "log_ioi_ratio_b4",
            "legato_ratio",
        ]}

    onsets = sorted(n.start for n in notes)
    iois = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    iois = [x for x in iois if x > 1e-4]  # drop near-zero

    if not iois:
        return {k: 0.0 for k in [
            "ioi_cv", "log_ioi_ratio_b0", "log_ioi_ratio_b1",
            "log_ioi_ratio_b2", "log_ioi_ratio_b3", "log_ioi_ratio_b4",
            "legato_ratio",
        ]}

    ioi_arr = np.array(iois)
    ioi_cv = float(np.std(ioi_arr) / (np.mean(ioi_arr) + 1e-9))

    log_ratios = [math.log(iois[i + 1] / (iois[i] + 1e-9))
                  for i in range(len(iois) - 1)]
    hist, _ = np.histogram(log_ratios, bins=5, range=(-2.0, 2.0), density=True)
    hist = hist / (hist.sum() + 1e-9)

    # Legato: fraction of note pairs where note ends after next note starts
    sorted_notes = sorted(notes, key=lambda n: n.start)
    overlaps = sum(
        1 for i in range(len(sorted_notes) - 1)
        if sorted_notes[i].end > sorted_notes[i + 1].start
    )
    legato_ratio = overlaps / (len(sorted_notes) - 1)

    return {
        "ioi_cv": ioi_cv,
        "log_ioi_ratio_b0": float(hist[0]),
        "log_ioi_ratio_b1": float(hist[1]),
        "log_ioi_ratio_b2": float(hist[2]),
        "log_ioi_ratio_b3": float(hist[3]),
        "log_ioi_ratio_b4": float(hist[4]),
        "legato_ratio": legato_ratio,
    }


# ---------------------------------------------------------------------------
# Combined feature vector
# ---------------------------------------------------------------------------

def extract_all(
    notes: list[Note],
    melody_notes: list[Note] | None = None,
    block: str = "all",
) -> np.ndarray:
    """
    Extract feature vector from accompaniment notes.
    block: 'harmony', 'texture', 'rhythm', or 'all'
    """
    feats: dict[str, float] = {}
    if block in ("harmony", "all"):
        feats.update(harmony_features(notes))
    if block in ("texture", "all"):
        feats.update(texture_features(notes, melody_notes))
    if block in ("rhythm", "all"):
        feats.update(rhythm_features(notes))
    return np.array(list(feats.values()), dtype=np.float32)


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JSD between two probability distributions (already normalized)."""
    p = p / (p.sum() + 1e-9)
    q = q / (q.sum() + 1e-9)
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * np.log2(a / (b + 1e-9) + 1e-9)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def wasserstein1(samples_p: np.ndarray, samples_q: np.ndarray) -> float:
    """1-Wasserstein distance between two 1D sample sets."""
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(samples_p, samples_q))


# ---------------------------------------------------------------------------
# Additional metrics
# ---------------------------------------------------------------------------

def pitch_class_entropy(notes: list[Note]) -> float:
    counts = np.zeros(12)
    for n in notes:
        counts[n.pitch % 12] += 1.0
    counts /= counts.sum() + 1e-9
    return float(-np.sum(counts * np.log2(counts + 1e-9)))


def fourgram_repetition_rate(notes: list[Note]) -> float:
    """Fraction of 4-grams of pitch intervals that are repeated."""
    pitches = [n.pitch for n in sorted(notes, key=lambda n: n.start)]
    if len(pitches) < 5:
        return 0.0
    intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    grams = [tuple(intervals[i:i + 4]) for i in range(len(intervals) - 3)]
    if not grams:
        return 0.0
    from collections import Counter
    counts = Counter(grams)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / len(grams)


def skyline_dominance(
    melody_notes: list[Note], accompaniment_notes: list[Note]
) -> float:
    """
    Fraction of melody onsets where the melody pitch is the highest sounding pitch.
    """
    if not melody_notes:
        return 1.0
    dominant = 0
    for m in melody_notes:
        acc_active = [n.pitch for n in accompaniment_notes
                      if n.start <= m.start < n.end]
        if not acc_active or m.pitch >= max(acc_active):
            dominant += 1
    return dominant / len(melody_notes)


def melody_clash_rate(
    melody_notes: list[Note], accompaniment_notes: list[Note],
    semitone_threshold: int = 1,
) -> float:
    """
    Fraction of accompaniment notes that clash with the melody
    (within semitone_threshold semitones, sounding simultaneously).
    """
    if not accompaniment_notes or not melody_notes:
        return 0.0
    clashes = 0
    for a in accompaniment_notes:
        for m in melody_notes:
            if a.start < m.end and a.end > m.start:
                if abs(a.pitch - m.pitch) <= semitone_threshold:
                    clashes += 1
                    break
    return clashes / len(accompaniment_notes)
