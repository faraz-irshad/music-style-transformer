"""
amt_utils.py

Shared helpers for the Anticipatory Music Transformer pipeline.
Used by train_lora.py and generate.py.

UNVERIFIED items (require Colab run to confirm):
  - CONTROL_OFFSET exact value (read from anticipation.vocab at runtime)
  - anticipation.ops interleave routine name and signature
  - events_to_midi return type (mido.MidiFile vs pretty_midi.PrettyMIDI)
"""
from __future__ import annotations
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
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
# WAV rendering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WavRenderResult:
    success: bool
    backend: str
    soundfont: str
    path: str
    error: str = ""

    def __bool__(self) -> bool:
        return self.success

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _synthesise_audio_numpy(pm, sr: int):
    """Return additive-sine audio for an explicitly selected fallback backend."""
    import numpy as np

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

    return audio


def _write_wav(audio, path: str, sr: int) -> None:
    import numpy as np
    import scipy.io.wavfile as wf
    import uuid

    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        raise ValueError("renderer produced empty audio")
    if not np.isfinite(samples).all():
        raise ValueError("renderer produced non-finite audio")
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-8:
        raise ValueError("renderer produced silent audio")
    if peak > 1.0:
        samples = samples / peak
    pcm = np.rint(samples * 32767).clip(-32768, 32767).astype(np.int16)
    if not np.any(pcm):
        raise ValueError("renderer produced silent int16 audio")
    destination = Path(path)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp.wav"
    )
    try:
        wf.write(str(temporary), sr, pcm)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def synthesise_wav_numpy(pm, path: str, sr: int = 22050) -> None:
    """Render additive-sine WAV; callers must explicitly select this backend."""
    _write_wav(_synthesise_audio_numpy(pm, sr), path, sr)


def render_wav(
    pm,
    path: str,
    sr: int = 44100,
    soundfont_path: str | None = None,
    backend: str = "fluidsynth",
) -> WavRenderResult:
    """Render with the selected backend and return its explicit outcome."""
    if backend not in ("fluidsynth", "numpy"):
        raise ValueError("backend must be 'fluidsynth' or 'numpy'")
    if sr <= 0:
        raise ValueError("sample rate must be positive")
    if backend == "numpy" and soundfont_path is not None:
        raise ValueError("SoundFont selection applies only to the fluidsynth backend")
    if soundfont_path is not None and not Path(soundfont_path).is_file():
        raise FileNotFoundError(f"SoundFont does not exist: {soundfont_path}")
    if soundfont_path:
        soundfont = str(Path(soundfont_path).resolve())
    elif backend == "fluidsynth":
        try:
            import pretty_midi.fluidsynth as fluidsynth
            soundfont = f"library-default:{fluidsynth.DEFAULT_SF2}"
        except Exception:
            soundfont = "library-default:unresolved"
    else:
        soundfont = "not-used"
    try:
        if backend == "fluidsynth":
            audio = pm.fluidsynth(fs=sr, sf2_path=soundfont_path)
        else:
            audio = _synthesise_audio_numpy(pm, sr)
        _write_wav(audio, path, sr)
        return WavRenderResult(True, backend, soundfont, str(path))
    except Exception as exc:
        return WavRenderResult(
            False, backend, soundfont, str(path),
            f"{type(exc).__name__}: {exc}",
        )


def balance_melody_accompaniment(
    pm,
    melody_notes: list[tuple[float, float, int]],
    melody_velocity: int = 80,
    accompaniment_gain: float = 0.55,
    onset_tolerance_sec: float = 0.03,
    duration_tolerance_sec: float = 0.05,
) -> tuple[object, dict[str, object]]:
    """Return a velocity-balanced copy; source timing and pitches are untouched."""
    import copy
    import math

    if not 1 <= melody_velocity <= 127:
        raise ValueError("melody_velocity must be in [1, 127]")
    if not math.isfinite(accompaniment_gain) or not 0 <= accompaniment_gain <= 1:
        raise ValueError("accompaniment_gain must be finite and in [0, 1]")
    if (
        not math.isfinite(onset_tolerance_sec)
        or not math.isfinite(duration_tolerance_sec)
        or onset_tolerance_sec < 0
        or duration_tolerance_sec < 0
    ):
        raise ValueError("matching tolerances must be finite and non-negative")

    balanced = copy.deepcopy(pm)
    generated = [
        note
        for instrument in balanced.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]
    unmatched = set(range(len(generated)))
    matched = 0
    for start, end, pitch in sorted(melody_notes):
        candidates = [
            index for index in unmatched
            if generated[index].pitch == pitch
            and abs(generated[index].start - start) <= onset_tolerance_sec
            and abs(generated[index].end - end) <= duration_tolerance_sec
        ]
        if not candidates:
            continue
        index = min(
            candidates,
            key=lambda i: (
                abs(generated[i].start - start)
                + abs(generated[i].end - end),
                i,
            ),
        )
        generated[index].velocity = melody_velocity
        unmatched.remove(index)
        matched += 1
    for index in unmatched:
        generated[index].velocity = max(
            0, min(127, round(generated[index].velocity * accompaniment_gain))
        )
    return balanced, {
        "enabled": True,
        "melody_notes_expected": len(melody_notes),
        "melody_notes_matched": matched,
        "melody_velocity": melody_velocity,
        "accompaniment_gain": accompaniment_gain,
        "onset_tolerance_sec": onset_tolerance_sec,
        "duration_tolerance_sec": duration_tolerance_sec,
        "transformed_fields": ["velocity"],
    }
