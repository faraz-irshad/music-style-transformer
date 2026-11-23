"""
amt_utils.py

Shared helpers for the Anticipatory Music Transformer pipeline.
Used by train_lora.py and generate.py.

UNVERIFIED items (require Colab run to confirm):
  - CONTROL_OFFSET exact value (read from anticipation.config at runtime)
  - anticipation.ops interleave routine name and signature
  - events_to_midi return type (mido.MidiFile vs pretty_midi.PrettyMIDI)
"""
from __future__ import annotations
import os
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.data.skyline import Note as SNote


# ---------------------------------------------------------------------------
# MIDI ↔ anticipation event token helpers
# ---------------------------------------------------------------------------

def write_notes_to_midi(notes: list, path: str, tempo: float = 100.0) -> None:
    """
    Write a list of Note-like objects (with .start, .end, .pitch) to a MIDI file.
    Uses mido directly so there is no dependency on pretty_midi here.
    """
    import mido
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo_us = int(60_000_000 / tempo)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    events: list[tuple[int, str, int]] = []
    for n in notes:
        tpb = 480
        tick_s = int(n.start * tempo / 60.0 * tpb)
        tick_e = int(n.end * tempo / 60.0 * tpb)
        p = max(0, min(127, n.pitch))
        events.append((tick_s, "on", p))
        events.append((tick_e, "off", p))
    events.sort(key=lambda x: x[0])
    prev = 0
    for tick, kind, pitch in events:
        delta = tick - prev
        if kind == "on":
            track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
        prev = tick
    mid.save(path)


def notes_to_events(notes: list) -> list[int]:
    """
    Convert a list of Note-like objects to anticipation integer event tokens
    by writing a temp MIDI file and calling midi_to_events.

    UNVERIFIED: midi_to_events exact signature and return format.
    """
    from anticipation.convert import midi_to_events
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp.close()
    try:
        write_notes_to_midi(notes, tmp.name)
        return midi_to_events(tmp.name)
    finally:
        os.unlink(tmp.name)


def melody_notes_to_controls(melody_notes: list, control_offset: int) -> list[int]:
    """
    Convert melody notes (list of (start, end, pitch) tuples or Note objects)
    to anticipation control tokens by adding control_offset to each event token.
    """
    from model.data.skyline import Note as SNote
    if melody_notes and isinstance(melody_notes[0], tuple):
        snotes = [SNote(s, e, p) for s, e, p in melody_notes]
    else:
        snotes = melody_notes
    events = notes_to_events(snotes)
    return [t + control_offset for t in events]


# ---------------------------------------------------------------------------
# events_to_midi return-type normalisation
# UNVERIFIED: anticipation.convert.events_to_midi may return mido.MidiFile
# or pretty_midi.PrettyMIDI depending on library version.
# We normalise to pretty_midi.PrettyMIDI here.
# ---------------------------------------------------------------------------

def events_to_pretty_midi(events: list):
    """
    Call anticipation.convert.events_to_midi and normalise the result to
    a pretty_midi.PrettyMIDI object regardless of what the library returns.

    UNVERIFIED: return type of events_to_midi.
    """
    import pretty_midi
    from anticipation.convert import events_to_midi
    result = events_to_midi(events)
    # If it's already a PrettyMIDI object, return directly
    if isinstance(result, pretty_midi.PrettyMIDI):
        return result
    # If it's a mido.MidiFile, save to temp and reload as PrettyMIDI
    import mido
    if isinstance(result, mido.MidiFile):
        tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        tmp.close()
        try:
            result.save(tmp.name)
            return pretty_midi.PrettyMIDI(tmp.name)
        finally:
            os.unlink(tmp.name)
    # Unknown type — try saving with .save() then .write()
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp.close()
    try:
        if hasattr(result, "save"):
            result.save(tmp.name)
        elif hasattr(result, "write"):
            result.write(tmp.name)
        else:
            raise TypeError(f"events_to_midi returned unknown type: {type(result)}")
        return pretty_midi.PrettyMIDI(tmp.name)
    finally:
        os.unlink(tmp.name)


def save_midi_file(pm_or_mido, path: str) -> None:
    """
    Save a pretty_midi.PrettyMIDI or mido.MidiFile to disk.
    Uses .write() for PrettyMIDI and .save() for mido.
    """
    import pretty_midi, mido
    if isinstance(pm_or_mido, pretty_midi.PrettyMIDI):
        pm_or_mido.write(path)
    elif isinstance(pm_or_mido, mido.MidiFile):
        pm_or_mido.save(path)
    else:
        raise TypeError(f"Cannot save object of type {type(pm_or_mido)}")


# ---------------------------------------------------------------------------
# Pure-numpy WAV synthesiser (fallback when fluidsynth is unavailable)
# ---------------------------------------------------------------------------

def synthesise_wav_numpy(pm, path: str, sr: int = 22050) -> None:
    """
    Render a pretty_midi.PrettyMIDI to WAV using a simple additive sine
    synthesiser. Quality is low but sufficient for demo purposes.
    Does not require fluidsynth or pyfluidsynth.
    """
    import numpy as np
    import scipy.io.wavfile as wf

    duration = pm.get_end_time()
    if duration <= 0:
        duration = 1.0
    n_samples = int(duration * sr) + sr  # +1 s tail
    audio = np.zeros(n_samples, dtype=np.float32)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            start = int(note.start * sr)
            end = int(note.end * sr)
            end = min(end, n_samples)
            if start >= end:
                continue
            t = np.arange(end - start, dtype=np.float32) / sr
            # Sine with simple ADSR envelope
            env = np.ones_like(t)
            attack = int(0.01 * sr)
            release = int(0.05 * sr)
            if attack > 0:
                env[:min(attack, len(t))] = np.linspace(0, 1, min(attack, len(t)))
            if release > 0 and len(t) > release:
                env[-release:] = np.linspace(1, 0, release)
            vel = (note.velocity / 127.0) * 0.15
            audio[start:end] += np.sin(2 * np.pi * freq * t) * env * vel

    # Normalise
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    wf.write(path, sr, (audio * 32767).astype(np.int16))


def render_wav(pm, path: str, sr: int = 44100) -> bool:
    """
    Render a pretty_midi.PrettyMIDI to WAV.
    Tries fluidsynth first; falls back to pure-numpy synthesiser.
    Returns True on success.
    """
    try:
        audio = pm.fluidsynth(fs=sr)
        import numpy as np, scipy.io.wavfile as wf
        wf.write(path, sr, (audio * 32767).astype(np.int16))
        return True
    except Exception:
        pass
    try:
        synthesise_wav_numpy(pm, path, sr=sr)
        return True
    except Exception as e:
        print(f"  WAV render failed entirely for {path}: {e}")
        return False
