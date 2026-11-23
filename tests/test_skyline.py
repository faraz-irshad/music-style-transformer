"""
tests/test_skyline.py

Unit tests for data/skyline.py.
All tests run on CPU with no GPU or network access required.
"""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.skyline import Note, skyline


def test_single_note():
    notes = [Note(0.0, 1.0, 60)]
    result = skyline(notes)
    assert len(result) == 1
    assert result[0].pitch == 60


def test_empty():
    assert skyline([]) == []


def test_two_simultaneous_keeps_higher():
    """When two notes start at the same time, only the higher pitch is kept."""
    notes = [Note(0.0, 1.0, 60), Note(0.0, 1.0, 67)]
    result = skyline(notes)
    assert len(result) == 1
    assert result[0].pitch == 67


def test_sequential_notes_all_kept():
    """Non-overlapping notes are all kept."""
    notes = [Note(0.0, 0.5, 60), Note(0.5, 1.0, 64), Note(1.0, 1.5, 67)]
    result = skyline(notes)
    assert len(result) == 3
    assert [n.pitch for n in result] == [60, 64, 67]


def test_lower_note_during_higher_dropped():
    """A lower note starting while a higher note is sounding is dropped."""
    notes = [
        Note(0.0, 2.0, 72),   # high note, long
        Note(0.5, 1.0, 60),   # low note starts during high note → dropped
    ]
    result = skyline(notes)
    pitches = [n.pitch for n in result]
    assert 72 in pitches
    assert 60 not in pitches


def test_melody_above_bass():
    """Typical piano: melody in treble, bass in bass — skyline keeps melody."""
    notes = [
        Note(0.0, 0.5, 60),   # bass C3
        Note(0.0, 0.5, 72),   # melody C5
        Note(0.5, 1.0, 62),   # bass D3
        Note(0.5, 1.0, 74),   # melody D5
    ]
    result = skyline(notes)
    assert [n.pitch for n in result] == [72, 74]


def test_output_sorted_by_start():
    notes = [Note(1.0, 1.5, 64), Note(0.0, 0.5, 60), Note(0.5, 1.0, 62)]
    result = skyline(notes)
    starts = [n.start for n in result]
    assert starts == sorted(starts)


def test_twinkle_skyline():
    """Twinkle Twinkle melody should survive skyline extraction unchanged."""
    from layer_a.generate import TWINKLE_MELODY
    notes = [Note(s, e, p) for s, e, p in TWINKLE_MELODY]
    result = skyline(notes)
    # All melody notes should be in the skyline (no lower note to mask them)
    result_pitches = [n.pitch for n in result]
    melody_pitches = [p for _, _, p in TWINKLE_MELODY]
    assert result_pitches == melody_pitches
