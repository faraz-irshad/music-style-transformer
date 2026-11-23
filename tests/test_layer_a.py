"""
tests/test_layer_a.py

Tests for layer_a/generate.py.
All tests run on CPU with no GPU or network access required.
"""
import os, sys, tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layer_a.generate import (
    Note, generate_accompaniment, notes_to_midi,
    TWINKLE_MELODY, _key_from_melody, _chord_for_beat,
    _chord_for_degree, _functional_chord_degrees,
)


# ---------------------------------------------------------------------------
# Key detection
# ---------------------------------------------------------------------------

def test_key_detection_c_major():
    # C major scale pitches should resolve to tonic 0 (C)
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C4–C5
    assert _key_from_melody(pitches) == 0


def test_key_detection_g_major():
    # G major scale
    pitches = [67, 69, 71, 72, 74, 76, 78, 79]
    assert _key_from_melody(pitches) == 7


def test_key_detection_empty():
    assert _key_from_melody([]) == 0


def test_functional_harmony_selects_chords_from_bar_weights():
    c_major_bar = [
        (0.0, 0.6, 60), (0.6, 1.2, 64), (1.2, 1.8, 67),
        (1.8, 2.4, 60),
    ]
    f_major_bar = [
        (0.0, 0.6, 65), (0.6, 1.2, 69), (1.2, 1.8, 72),
        (1.8, 2.4, 65),
    ]

    assert _functional_chord_degrees(c_major_bar, tonic=0, beat=0.6) == [0]
    assert _functional_chord_degrees(f_major_bar, tonic=0, beat=0.6) == [3]
    assert _chord_for_degree(0, tonic=0)[0] % 12 == 0
    assert _chord_for_degree(3, tonic=0)[0] % 12 == 5


# ---------------------------------------------------------------------------
# Accompaniment generation
# ---------------------------------------------------------------------------

def test_baroque_returns_notes():
    acc = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
    assert len(acc) > 0


def test_romantic_returns_notes():
    acc = generate_accompaniment(TWINKLE_MELODY, "romantic", seed=42)
    assert len(acc) > 0


def test_styles_differ():
    """Baroque and Romantic accompaniments must not be identical."""
    baroque = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
    romantic = generate_accompaniment(TWINKLE_MELODY, "romantic", seed=42)
    baroque_pitches = sorted(n.pitch for n in baroque)
    romantic_pitches = sorted(n.pitch for n in romantic)
    assert baroque_pitches != romantic_pitches, "Baroque and Romantic outputs are identical"


def test_deterministic_with_seed():
    """Same seed must produce identical output."""
    acc1 = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
    acc2 = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
    assert [(n.start, n.pitch) for n in acc1] == [(n.start, n.pitch) for n in acc2]


def test_different_seeds_differ():
    acc1 = generate_accompaniment(TWINKLE_MELODY, "romantic", seed=42)
    acc2 = generate_accompaniment(TWINKLE_MELODY, "romantic", seed=99)
    # Romantic has rubato jitter, so timing should differ
    starts1 = [n.start for n in acc1]
    starts2 = [n.start for n in acc2]
    assert starts1 != starts2


def test_accompaniment_below_melody():
    """Accompaniment notes should generally stay below the melody."""
    melody_max = max(p for _, _, p in TWINKLE_MELODY)
    acc = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
    # Allow a small fraction above (inner voices may occasionally cross)
    above = sum(1 for n in acc if n.pitch > melody_max)
    assert above / len(acc) < 0.1, f"{above}/{len(acc)} notes above melody max {melody_max}"


def test_note_durations_positive():
    for style in ("baroque", "romantic"):
        acc = generate_accompaniment(TWINKLE_MELODY, style, seed=42)
        for n in acc:
            assert n.end > n.start, f"Non-positive duration: {n}"


def test_velocities_in_range():
    for style in ("baroque", "romantic"):
        acc = generate_accompaniment(TWINKLE_MELODY, style, seed=42)
        for n in acc:
            assert 0 <= n.velocity <= 127, f"Velocity out of range: {n.velocity}"


def test_baroque_staccato_shorter_than_romantic():
    """Verify coded note-duration behavior; this is not measured style evidence."""
    baroque = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
    romantic = generate_accompaniment(TWINKLE_MELODY, "romantic", seed=42)
    mean_baroque = sum(n.end - n.start for n in baroque) / len(baroque)
    mean_romantic = sum(n.end - n.start for n in romantic) / len(romantic)
    assert mean_baroque < mean_romantic, (
        f"Expected baroque ({mean_baroque:.3f}s) < romantic ({mean_romantic:.3f}s)"
    )


def test_romantic_rising_dynamics():
    """Romantic velocity should increase over time (rising dynamics)."""
    acc = generate_accompaniment(TWINKLE_MELODY, "romantic", seed=42)
    sorted_acc = sorted(acc, key=lambda n: n.start)
    n = len(sorted_acc)
    first_half_vel = sum(n_.velocity for n_ in sorted_acc[:n//2]) / (n//2)
    second_half_vel = sum(n_.velocity for n_ in sorted_acc[n//2:]) / (n - n//2)
    assert second_half_vel > first_half_vel, (
        f"Expected rising dynamics: first={first_half_vel:.1f}, second={second_half_vel:.1f}"
    )


def test_empty_melody():
    acc = generate_accompaniment([], "baroque", seed=42)
    assert acc == []


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

def test_midi_file_created():
    """notes_to_midi must produce a valid MIDI file."""
    import mido
    acc = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        path = f.name
    try:
        notes_to_midi(TWINKLE_MELODY, acc, "baroque", path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        mid = mido.MidiFile(path)
        assert len(mid.tracks) >= 2
    finally:
        os.unlink(path)


def test_midi_melody_preserved():
    """Melody track in MIDI must contain the correct pitches."""
    import mido
    acc = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        path = f.name
    try:
        notes_to_midi(TWINKLE_MELODY, acc, "baroque", path)
        mid = mido.MidiFile(path)
        # Track 1 is melody (track 0 is tempo)
        melody_track = mid.tracks[1]
        note_ons = [msg.note for msg in melody_track
                    if msg.type == "note_on" and msg.velocity > 0]
        expected = [p for _, _, p in TWINKLE_MELODY]
        assert note_ons == expected, f"Melody pitches mismatch: {note_ons} vs {expected}"
    finally:
        os.unlink(path)


def test_provenance_in_track_name():
    """Track names must contain 'layer_a_scripted'."""
    import mido
    acc = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        path = f.name
    try:
        notes_to_midi(TWINKLE_MELODY, acc, "baroque", path)
        mid = mido.MidiFile(path)
        names = [msg.name for track in mid.tracks
                 for msg in track if msg.type == "track_name"]
        assert any("layer_a_scripted" in n for n in names), f"No provenance label in {names}"
    finally:
        os.unlink(path)
