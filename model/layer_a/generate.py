"""
layer_a/generate.py

Rule-based piano accompaniment generator.
Provenance label: layer_a_scripted

This is NOT a learned model. It uses hand-coded rules to produce
Baroque and Romantic accompaniments for an arbitrary melody.
Never present its output as AI/ML output.

Baroque rules:
  - Steady walking bass (quarter notes, stepwise or chord-tone motion)
  - Short crisp notes (duration = 0.4 * beat)
  - Constant pulse, slow harmonic rhythm (one chord per bar)
  - Inner voices fill chord tones

Romantic rules:
  - Rolling arpeggios (broken chord, bottom to top)
  - Octave bass on beat 1
  - Sustained notes (duration = 0.95 * beat)
  - Slight tempo give (rubato: ±5% beat jitter)
  - Rising dynamics (velocity increases over time)

Dependencies: standard library for generation; mido for MIDI export and
pretty_midi for MIDI loading/WAV rendering.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Note:
    start: float    # seconds
    end: float      # seconds
    pitch: int      # MIDI 0–127
    velocity: int   # 0–127


Style = Literal["baroque", "romantic"]

# ---------------------------------------------------------------------------
# Music theory helpers
# ---------------------------------------------------------------------------

# Diatonic triads in major key (scale degrees 0–6), intervals above root
_TRIAD = {
    0: (0, 4, 7),   # I   major
    1: (0, 3, 7),   # ii  minor
    2: (0, 3, 7),   # iii minor
    3: (0, 4, 7),   # IV  major
    4: (0, 4, 7),   # V   major
    5: (0, 3, 7),   # vi  minor
    6: (0, 3, 6),   # vii diminished
}

_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]


def _key_from_melody(pitches: list[int]) -> int:
    """
    Estimate the key (tonic pitch class 0–11) from melody pitches using
    Krumhansl-Schmuckler key profiles (major only, simplified).
    Returns the pitch class of the estimated tonic.
    """
    if not pitches:
        return 0
    # Major key profile (Krumhansl 1990, simplified)
    profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
               2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    counts = [0.0] * 12
    for p in pitches:
        counts[p % 12] += 1.0
    best_key, best_score = 0, -1e9
    for k in range(12):
        score = sum(counts[(k + i) % 12] * profile[i] for i in range(12))
        if score > best_score:
            best_score, best_key = score, k
    return best_key


def _chord_for_degree(degree: int, tonic: int) -> tuple[int, ...]:
    """Build a diatonic triad from a major-key scale degree."""
    intervals = _TRIAD[degree]
    root_pc = (tonic + _MAJOR_SCALE[degree]) % 12
    # Bass in octave 3
    bass = root_pc + 48
    if bass > 59:
        bass -= 12
    # Inner voices in octave 4
    inner = tuple(bass + 12 + iv for iv in intervals[1:])
    return (bass,) + inner


def _chord_for_beat(beat_idx: int, tonic: int, melody_pitch: int) -> tuple[int, ...]:
    """Compatibility helper using a functional cycle, not the melody as root."""
    del melody_pitch
    degree = (0, 3, 4, 0, 5, 3, 4, 0)[beat_idx % 8]
    return _chord_for_degree(degree, tonic)


def _functional_chord_degrees(
    melody_notes: list[tuple[float, float, int]],
    tonic: int,
    beat: float,
) -> list[int]:
    """Choose I, IV, V, or vi per bar from duration-weighted melody pitch classes."""
    if not melody_notes:
        return [0]
    first_beat = min(start for start, _, _ in melody_notes)
    last_end = max(end for _, end, _ in melody_notes)
    bar_duration = 4 * beat
    n_bars = max(1, int((last_end - first_beat + bar_duration - 1e-9) // bar_duration))
    candidates = (0, 3, 4, 5)
    preference = {degree: rank for rank, degree in enumerate((0, 4, 3, 5))}
    result = []
    for bar in range(n_bars):
        bar_start = first_beat + bar * bar_duration
        bar_end = bar_start + bar_duration
        weights = [0.0] * 12
        for start, end, pitch in melody_notes:
            overlap = max(0.0, min(end, bar_end) - max(start, bar_start))
            weights[pitch % 12] += overlap
        if not any(weights):
            result.append(0)
            continue
        def score(degree: int) -> tuple[float, int]:
            root = (tonic + _MAJOR_SCALE[degree]) % 12
            chord_pcs = {(root + interval) % 12 for interval in _TRIAD[degree]}
            return sum(weights[pc] for pc in chord_pcs), -preference[degree]
        result.append(max(candidates, key=score))
    return result


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_accompaniment(
    melody_notes: list[tuple[float, float, int]],  # (start_sec, end_sec, pitch)
    style: Style,
    tempo_bpm: float = 100.0,
    seed: int = 42,
) -> list[Note]:
    """
    Generate a piano accompaniment for the given melody.

    Args:
        melody_notes: list of (start_sec, end_sec, pitch) tuples.
        style: 'baroque' or 'romantic'.
        tempo_bpm: tempo in beats per minute.
        seed: random seed for determinism.

    Returns:
        List of Note objects for the accompaniment (melody NOT included).
    """
    rng = random.Random(seed)
    beat = 60.0 / tempo_bpm  # seconds per beat

    if not melody_notes:
        return []

    pitches = [p for _, _, p in melody_notes]
    tonic = _key_from_melody(pitches)
    chord_degrees = _functional_chord_degrees(melody_notes, tonic, beat)

    # Determine time range
    t_start = melody_notes[0][0]
    t_end = max(e for _, e, _ in melody_notes)

    # Build beat grid
    beats: list[float] = []
    t = t_start
    while t < t_end + beat:
        beats.append(t)
        t += beat

    accompaniment: list[Note] = []

    if style == "baroque":
        accompaniment = _baroque(
            melody_notes, beats, tonic, beat, rng, chord_degrees
        )
    else:
        accompaniment = _romantic(
            melody_notes, beats, tonic, beat, rng, chord_degrees
        )

    return accompaniment


def _melody_pitch_at(t: float, melody_notes: list[tuple[float, float, int]]) -> int:
    """Return the melody pitch sounding at time t, or 60 (C4) if none."""
    for s, e, p in melody_notes:
        if s <= t < e:
            return p
    # Nearest onset
    if melody_notes:
        return min(melody_notes, key=lambda n: abs(n[0] - t))[2]
    return 60


def _baroque(
    melody_notes: list[tuple[float, float, int]],
    beats: list[float],
    tonic: int,
    beat: float,
    rng: random.Random,
    chord_degrees: list[int],
) -> list[Note]:
    """Baroque: walking bass + short chord stabs on beats 2 and 4."""
    notes: list[Note] = []
    staccato = 0.4 * beat
    vel_base = 64

    prev_bass = None
    for i, t in enumerate(beats):
        if i >= len(beats) - 1:
            break
        mel_p = _melody_pitch_at(t, melody_notes)
        chord = _chord_for_degree(chord_degrees[min(i // 4, len(chord_degrees) - 1)], tonic)
        bass = chord[0]

        # Walking bass: stepwise motion toward chord root
        if prev_bass is not None:
            diff = bass - prev_bass
            if abs(diff) > 2:
                # Step toward target
                step = 1 if diff > 0 else -1
                bass = prev_bass + step
        prev_bass = bass

        # Bass note on every beat
        notes.append(Note(t, t + staccato, bass, vel_base))

        # Inner voices on beats 2 and 4 (every other beat)
        if i % 2 == 1:
            for iv in chord[1:]:
                # Keep inner voices below melody
                while iv >= mel_p - 2 and iv > 48:
                    iv -= 12
                notes.append(Note(t, t + staccato, iv, vel_base - 8))

    return notes


def _romantic(
    melody_notes: list[tuple[float, float, int]],
    beats: list[float],
    tonic: int,
    beat: float,
    rng: random.Random,
    chord_degrees: list[int],
) -> list[Note]:
    """Romantic: octave bass on beat 1, rolling arpeggios on beats 2–4."""
    notes: list[Note] = []
    legato = 0.95 * beat
    n_beats = len(beats)

    for i, t in enumerate(beats):
        if i >= n_beats - 1:
            break
        # Rubato: slight timing jitter ±5%
        jitter = rng.uniform(-0.05, 0.05) * beat
        t_actual = t + jitter

        mel_p = _melody_pitch_at(t, melody_notes)
        chord = _chord_for_degree(chord_degrees[min(i // 4, len(chord_degrees) - 1)], tonic)
        bass = chord[0]

        # Rising dynamics
        progress = i / max(n_beats - 1, 1)
        vel = int(50 + 40 * progress)

        beat_in_bar = i % 4

        if beat_in_bar == 0:
            # Beat 1: octave bass
            notes.append(Note(t_actual, t_actual + legato, bass, vel))
            notes.append(Note(t_actual, t_actual + legato, bass + 12, vel - 8))
        else:
            # Beats 2–4: arpeggio (bottom to top, evenly spaced within beat)
            chord_tones = list(chord)
            # Keep below melody
            chord_tones = [c for c in chord_tones if c < mel_p - 1]
            if not chord_tones:
                chord_tones = [bass]
            n = len(chord_tones)
            for j, ct in enumerate(sorted(chord_tones)):
                t_arp = t_actual + j * (beat / max(n, 1)) * 0.9
                notes.append(Note(t_arp, t_arp + legato, ct, vel - j * 4))

    return notes


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

def notes_to_midi(
    melody_notes: list[tuple[float, float, int]],
    accompaniment: list[Note],
    style: Style,
    output_path: str,
    tempo_bpm: float = 100.0,
) -> None:
    """
    Write melody + accompaniment to a MIDI file.
    Track 0: melody (program 0, channel 0)
    Track 1: accompaniment (program 0, channel 1)
    Filename should include 'layer_a_scripted' per provenance rules.
    """
    import mido
    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    tempo_us = int(60_000_000 / tempo_bpm)

    def _sec_to_tick(sec: float) -> int:
        return int(sec * tempo_bpm / 60.0 * 480)

    # Tempo track
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    tempo_track.append(mido.MetaMessage("track_name", name=f"layer_a_scripted_{style}", time=0))
    mid.tracks.append(tempo_track)

    def _make_track(note_list: list, channel: int, name: str) -> mido.MidiTrack:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        events: list[tuple[int, int, int, int]] = []  # (tick, type, pitch, vel)
        for item in note_list:
            if isinstance(item, Note):
                s, e, p, v = item.start, item.end, item.pitch, item.velocity
            else:
                s, e, p = item
                v = 80
            events.append((_sec_to_tick(s), 1, p, v))
            events.append((_sec_to_tick(e), 0, p, 0))
        events.sort(key=lambda x: (x[0], x[1]))
        prev_tick = 0
        for tick, on, pitch, vel in events:
            delta = tick - prev_tick
            if on:
                track.append(mido.Message("note_on", channel=channel,
                                          note=pitch, velocity=vel, time=delta))
            else:
                track.append(mido.Message("note_off", channel=channel,
                                          note=pitch, velocity=0, time=delta))
            prev_tick = tick
        return track

    mid.tracks.append(_make_track(melody_notes, 0, "melody"))
    mid.tracks.append(_make_track(accompaniment, 1, f"layer_a_scripted_{style}"))
    mid.save(output_path)


# ---------------------------------------------------------------------------
# Demo: Twinkle Twinkle Little Star
# ---------------------------------------------------------------------------

TWINKLE_MELODY: list[tuple[float, float, int]] = [
    # (start_sec, end_sec, pitch)  — C major, 100 BPM, beat = 0.6 s
    (0.0,  0.6,  60),  # C4
    (0.6,  1.2,  60),  # C4
    (1.2,  1.8,  67),  # G4
    (1.8,  2.4,  67),  # G4
    (2.4,  3.0,  69),  # A4
    (3.0,  3.6,  69),  # A4
    (3.6,  4.8,  67),  # G4 (half note)
    (4.8,  5.4,  65),  # F4
    (5.4,  6.0,  65),  # F4
    (6.0,  6.6,  64),  # E4
    (6.6,  7.2,  64),  # E4
    (7.2,  7.8,  62),  # D4
    (7.8,  8.4,  62),  # D4
    (8.4,  9.6,  60),  # C4 (half note)
]


if __name__ == "__main__":
    import sys, os
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    for style in ("baroque", "romantic"):
        acc = generate_accompaniment(TWINKLE_MELODY, style, tempo_bpm=100.0, seed=42)
        path = os.path.join(out_dir, f"twinkle_layer_a_scripted_{style}.mid")
        notes_to_midi(TWINKLE_MELODY, acc, style, path, tempo_bpm=100.0)
        print(f"Saved {path} ({len(acc)} accompaniment notes)")
