"""
data/skyline.py

Skyline melody extractor: at each onset, keep only the highest sounding pitch.
Works on pretty_midi.PrettyMIDI objects or lists of (start, end, pitch) tuples.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence


@dataclass
class Note:
    start: float   # seconds
    end: float     # seconds
    pitch: int     # MIDI pitch 0–127


def skyline(notes: Sequence[Note], frame_sec: float = 0.01) -> list[Note]:
    """
    Extract the melody as the highest sounding pitch at each onset.

    Algorithm: for each note onset, if no higher-pitched note is already
    sounding, include this note in the melody. Ties broken by pitch (higher wins).

    Args:
        notes: sequence of Note objects, any order.
        frame_sec: unused (kept for API compatibility); extraction is onset-based.

    Returns:
        List of Note objects forming the skyline melody, sorted by start time.
    """
    if not notes:
        return []

    # Sort by start time, then by pitch descending (highest first at same onset)
    sorted_notes = sorted(notes, key=lambda n: (n.start, -n.pitch))

    melody: list[Note] = []
    # Track the highest pitch currently sounding at each onset
    # We keep a running "ceiling" of active notes
    active: list[Note] = []  # notes currently sounding

    for note in sorted_notes:
        # Remove notes that have ended before this onset
        active = [a for a in active if a.end > note.start]
        # Include this note only if it is the highest at its onset
        if not active or note.pitch >= max(a.pitch for a in active):
            melody.append(note)
        active.append(note)

    return sorted(melody, key=lambda n: n.start)


def skyline_from_pretty_midi(pm_instrument) -> list[Note]:
    """Extract skyline from a pretty_midi Instrument object."""
    notes = [Note(n.start, n.end, n.pitch) for n in pm_instrument.notes]
    return skyline(notes)


def skyline_from_midi_file(midi_path: str) -> list[Note]:
    """
    Load a MIDI file and extract the skyline across all non-drum tracks.
    Returns notes sorted by start time.
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_notes: list[Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        all_notes.extend(Note(n.start, n.end, n.pitch) for n in inst.notes)
    return skyline(all_notes)


def notes_to_pretty_midi(notes: list[Note], tempo: float = 120.0) -> "pretty_midi.PrettyMIDI":
    """Convert a list of Note objects to a single-track PrettyMIDI object."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0, name="melody")
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=80, pitch=n.pitch, start=n.start, end=n.end
        ))
    pm.instruments.append(inst)
    return pm
