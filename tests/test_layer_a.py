"""Focused tests for the rule-based Layer A accompaniment generator."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layer_a.generate import (
    Note,
    TWINKLE_MELODY,
    _chord_for_degree,
    _functional_chord_degrees,
    _key_from_melody,
    generate_accompaniment,
    notes_to_midi,
    plan_harmony,
)


def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def test_legacy_key_and_functional_harmony_helpers():
    assert _key_from_melody([60, 62, 64, 65, 67, 69, 71, 72]) == 0
    assert _key_from_melody([67, 69, 71, 72, 74, 76, 78, 79]) == 7
    assert _key_from_melody([]) == 0
    c_major = [(0.0, 0.6, 60), (0.6, 1.2, 64), (1.2, 1.8, 67)]
    f_major = [(0.0, 0.6, 65), (0.6, 1.2, 69), (1.2, 1.8, 72)]
    assert _functional_chord_degrees(c_major, tonic=0, beat=0.6) == [0]
    assert _functional_chord_degrees(f_major, tonic=0, beat=0.6) == [3]
    assert _chord_for_degree(0, tonic=0)[0] % 12 == 0
    assert _chord_for_degree(3, tonic=0)[0] % 12 == 5


@pytest.mark.parametrize("style", ["baroque", "romantic"])
def test_generation_is_deterministic_and_nonempty(style):
    first = generate_accompaniment(TWINKLE_MELODY, style, seed=42)
    second = generate_accompaniment(TWINKLE_MELODY, style, seed=42)
    assert first
    assert first == second


def test_rule_based_seed_argument_is_compatible_but_does_not_add_jitter():
    a = generate_accompaniment(TWINKLE_MELODY, "romantic", seed=42)
    b = generate_accompaniment(TWINKLE_MELODY, "romantic", seed=99)
    assert a == b


def test_empty_melody():
    assert generate_accompaniment([], "baroque") == []
    assert generate_accompaniment([], "romantic") == []
    assert plan_harmony([]).chords == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"style": "invalid"},
        {"tempo_bpm": 0},
        {"tempo_bpm": float("nan")},
        {"beats_per_bar": 0},
        {"beats_per_bar": 4.0},
        {"downbeat_sec": float("inf")},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    style = kwargs.pop("style", "baroque")
    with pytest.raises(ValueError):
        generate_accompaniment(TWINKLE_MELODY, style, **kwargs)


def test_pickup_and_downbeat_define_bar_boundaries_and_pulse_grid():
    melody = [(0.0, 0.3, 60), (0.3, 0.6, 62), (0.6, 1.0, 64), (1.0, 1.5, 65)]
    plan = plan_harmony(melody, tempo_bpm=100, beats_per_bar=4, downbeat_sec=0.6)
    assert [chord.bar_index for chord in plan.chords] == [-1, 0]
    assert plan.chords[0].end == pytest.approx(0.6)
    assert plan.chords[1].start == pytest.approx(0.6)
    accompaniment = generate_accompaniment(
        melody, "baroque", downbeat_sec=0.6
    )
    assert 0.0 in [note.start for note in accompaniment]
    assert 0.6 in [note.start for note in accompaniment]


def test_timing_pitch_velocity_and_accompaniment_endpoint_safety():
    melody_end = max(end for _, end, _ in TWINKLE_MELODY)
    for style in ("baroque", "romantic"):
        accompaniment = generate_accompaniment(TWINKLE_MELODY, style)
        assert accompaniment
        for note in accompaniment:
            assert 0 <= note.start < note.end <= melody_end
            assert 0 <= note.pitch <= 127
            assert 1 <= note.velocity <= 127
            assert note.velocity < 80
            assert note.pitch <= max(pitch for _, _, pitch in TWINKLE_MELODY)


def test_accompaniment_has_no_overlapping_duplicate_pitches():
    for style in ("baroque", "romantic"):
        accompaniment = generate_accompaniment(TWINKLE_MELODY, style)
        for pitch in {note.pitch for note in accompaniment}:
            same_pitch = sorted(
                (note for note in accompaniment if note.pitch == pitch),
                key=lambda note: note.start,
            )
            for earlier, later in zip(same_pitch, same_pitch[1:]):
                assert earlier.end <= later.start


def test_bass_is_chord_tone_and_plan_is_shared_across_styles():
    plan = plan_harmony(TWINKLE_MELODY)
    assert plan.key
    assert plan.mode in ("major", "minor")
    assert plan.chords
    for style in ("baroque", "romantic"):
        accompaniment = generate_accompaniment(
            TWINKLE_MELODY, style, harmonic_plan=plan
        )
        bass_notes = [note for note in accompaniment if 36 <= note.pitch <= 52]
        assert bass_notes
        for note in bass_notes:
            chord = next(c for c in plan.chords if c.start <= note.start < c.end)
            assert note.pitch % 12 in chord.pitch_classes
    assert plan_harmony(TWINKLE_MELODY) == plan
    assert plan_harmony(TWINKLE_MELODY, tempo_bpm=100) == plan


def test_minor_key_candidates_are_inspectable():
    a_minor = [
        (0.0, 0.6, 69), (0.6, 1.2, 71), (1.2, 1.8, 72),
        (1.8, 2.4, 74), (2.4, 3.0, 76), (3.0, 3.6, 77),
        (3.6, 4.2, 69),
    ]
    plan = plan_harmony(a_minor)
    assert plan.tonic == 9
    assert plan.mode == "minor"
    assert plan.as_dict()["key"] == "A minor"


def test_styles_have_different_rhythmic_and_voicing_patterns():
    baroque = generate_accompaniment(TWINKLE_MELODY, "baroque")
    romantic = generate_accompaniment(TWINKLE_MELODY, "romantic")
    assert [(n.start, n.pitch) for n in baroque] != [
        (n.start, n.pitch) for n in romantic
    ]
    assert sorted(n.pitch for n in baroque) != sorted(n.pitch for n in romantic)
    assert sum(n.end - n.start for n in baroque) / len(baroque) < (
        sum(n.end - n.start for n in romantic) / len(romantic)
    )


def test_phrase_dynamics_are_bounded_not_an_uninterrupted_crescendo():
    notes = generate_accompaniment(TWINKLE_MELODY, "romantic")
    velocities = [note.velocity for note in notes]
    assert min(velocities) >= 1
    assert max(velocities) <= 127
    assert max(velocities) - min(velocities) <= 13
    assert velocities[-1] < max(velocities)


def _midi_notes_by_pitch(track):
    absolute = 0
    active = {}
    found = []
    for message in track:
        absolute += message.time
        if message.type == "note_on" and message.velocity:
            active.setdefault(message.note, []).append(absolute)
        elif message.type in ("note_off", "note_on"):
            starts = active.get(message.note, [])
            if starts:
                found.append((message.note, starts.pop(0), absolute))
    return sorted(found, key=lambda item: (item[1], item[0]))


def test_midi_export_has_nonnegative_deltas_and_preserves_melody_ticks():
    import mido

    accompaniment = generate_accompaniment(TWINKLE_MELODY, "romantic")
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
        path = handle.name
    try:
        notes_to_midi(TWINKLE_MELODY, accompaniment, "romantic", path)
        midi = mido.MidiFile(path)
        assert len(midi.tracks) >= 3
        assert all(message.time >= 0 for track in midi.tracks for message in track)
        actual = _midi_notes_by_pitch(midi.tracks[1])
        expected = sorted(
            [
                (
                pitch,
                round(start * 100 / 60 * midi.ticks_per_beat),
                round(end * 100 / 60 * midi.ticks_per_beat),
                )
                for start, end, pitch in TWINKLE_MELODY
            ],
            key=lambda item: (item[1], item[0]),
        )
        assert len(actual) == len(expected)
        for observed, wanted in zip(actual, expected):
            assert observed[0] == wanted[0]
            assert abs(observed[1] - wanted[1]) <= 1
            assert abs(observed[2] - wanted[2]) <= 1
    finally:
        os.unlink(path)


def test_export_rejects_negative_time_instead_of_writing_negative_delta():
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
        path = handle.name
    try:
        with pytest.raises(ValueError):
            notes_to_midi([(-0.1, 0.5, 60)], [], "baroque", path)
    finally:
        os.unlink(path)


def test_existing_positional_callers_remain_compatible():
    accompaniment = generate_accompaniment(TWINKLE_MELODY, "baroque", 100.0, 42)
    assert accompaniment
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
        path = handle.name
    try:
        notes_to_midi(TWINKLE_MELODY, accompaniment, "baroque", path, 100.0)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)
