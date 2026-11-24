"""
Rule-based piano accompaniment generator.
Provenance label: layer_a_scripted

This is not a learned model and produces Baroque-inspired and
Romantic-inspired arrangements, not historically authentic compositions.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int


@dataclass(frozen=True)
class PlannedChord:
    bar_index: int
    start: float
    end: float
    degree: int
    root_pc: int
    quality: str
    pitch_classes: tuple[int, ...]


@dataclass(frozen=True)
class HarmonicPlan:
    tonic: int
    mode: Literal["major", "minor"]
    chords: tuple[PlannedChord, ...]

    @property
    def key(self) -> str:
        return f"{_PITCH_NAMES[self.tonic]} {self.mode}"

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "tonic": self.tonic,
            "mode": self.mode,
            "chords": [
                {
                    "bar_index": chord.bar_index,
                    "start_sec": chord.start,
                    "end_sec": chord.end,
                    "degree": chord.degree,
                    "root": _PITCH_NAMES[chord.root_pc],
                    "quality": chord.quality,
                    "pitch_classes": list(chord.pitch_classes),
                }
                for chord in self.chords
            ],
        }


Style = Literal["baroque", "romantic"]

_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
_TRIADS = {
    "major": ((0, 4, 7), (0, 3, 7), (0, 3, 7), (0, 4, 7),
              (0, 4, 7), (0, 3, 7), (0, 3, 6)),
    "minor": ((0, 3, 7), (0, 3, 6), (0, 4, 7), (0, 3, 7),
              (0, 3, 7), (0, 4, 7), (0, 4, 7)),
}
_MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def _validate_timing(tempo_bpm: float, beats_per_bar: int, downbeat_sec: float) -> None:
    if isinstance(tempo_bpm, bool) or not isinstance(tempo_bpm, (int, float)):
        raise ValueError("tempo_bpm must be a finite positive number")
    if not math.isfinite(tempo_bpm) or tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be a finite positive number")
    if isinstance(beats_per_bar, bool) or not isinstance(beats_per_bar, int):
        raise ValueError("beats_per_bar must be an integer from 1 to 16")
    if not 1 <= beats_per_bar <= 16:
        raise ValueError("beats_per_bar must be an integer from 1 to 16")
    if isinstance(downbeat_sec, bool) or not isinstance(downbeat_sec, (int, float)):
        raise ValueError("downbeat_sec must be a finite number")
    if not math.isfinite(downbeat_sec):
        raise ValueError("downbeat_sec must be a finite number")


def _validate_style(style: str) -> None:
    if style not in ("baroque", "romantic"):
        raise ValueError("style must be 'baroque' or 'romantic'")


def _validate_melody(melody_notes: list[tuple[float, float, int]]) -> None:
    for note in melody_notes:
        if not isinstance(note, (tuple, list)) or len(note) != 3:
            raise ValueError("each melody note must be (start_sec, end_sec, pitch)")
        start, end, pitch = note
        if (isinstance(start, bool) or not isinstance(start, (int, float))
                or isinstance(end, bool) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start):
            raise ValueError("melody note times must be finite with 0 <= start < end")
        if isinstance(pitch, bool) or not isinstance(pitch, int) or not 0 <= pitch <= 127:
            raise ValueError("melody pitches must be integers from 0 to 127")


def _key_from_melody(pitches: list[int]) -> int:
    """Compatibility helper returning the major-profile tonic pitch class."""
    if not pitches:
        return 0
    counts = [0.0] * 12
    for pitch in pitches:
        counts[pitch % 12] += 1
    return max(
        range(12),
        key=lambda tonic: sum(counts[(tonic + i) % 12] * _MAJOR_PROFILE[i]
                              for i in range(12)),
    )


def _estimate_key(melody_notes: list[tuple[float, float, int]]) -> tuple[int, Literal["major", "minor"]]:
    if not melody_notes:
        return 0, "major"
    weights = [0.0] * 12
    for start, end, pitch in melody_notes:
        weights[pitch % 12] += end - start
    mean = sum(weights) / 12
    centered = [weight - mean for weight in weights]
    denominator = math.sqrt(sum(value * value for value in centered))
    candidates: list[tuple[float, int, int, Literal["major", "minor"]]] = []
    for mode_order, (mode, profile) in enumerate(
        (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE))
    ):
        profile_mean = sum(profile) / 12
        centered_profile = [value - profile_mean for value in profile]
        profile_norm = math.sqrt(sum(value * value for value in centered_profile))
        for tonic in range(12):
            rotated = [centered[(tonic + i) % 12] for i in range(12)]
            score = (
                sum(value * profile_value
                    for value, profile_value in zip(rotated, centered_profile))
                / (denominator * profile_norm)
                if denominator else 0.0
            )
            candidates.append((score, -mode_order, -tonic, mode))
    best = max(candidates)
    return -best[2], best[3]


def _chord_for_degree(degree: int, tonic: int) -> tuple[int, ...]:
    """Compatibility helper returning a major-key chord voicing."""
    if not 0 <= degree <= 6:
        raise ValueError("degree must be from 0 to 6")
    root_pc = (tonic + _MAJOR_SCALE[degree]) % 12
    intervals = _TRIADS["major"][degree]
    bass = 48 + root_pc
    while bass > 59:
        bass -= 12
    return (bass,) + tuple(bass + 12 + interval for interval in intervals[1:])


def _chord_for_beat(beat_idx: int, tonic: int, melody_pitch: int) -> tuple[int, ...]:
    """Compatibility helper using a functional cycle, not the melody as root."""
    del melody_pitch
    return _chord_for_degree((0, 3, 4, 0, 5, 3, 4, 0)[beat_idx % 8], tonic)


def _functional_chord_degrees(
    melody_notes: list[tuple[float, float, int]], tonic: int, beat: float
) -> list[int]:
    """Compatibility helper selecting major-key chords by bar overlap."""
    if not melody_notes:
        return [0]
    bar = 4 * beat
    first = min(start for start, _, _ in melody_notes)
    last = max(end for _, end, _ in melody_notes)
    n_bars = max(1, math.ceil((last - first) / bar - 1e-12))
    result = []
    for index in range(n_bars):
        weights = [0.0] * 12
        for start, end, pitch in melody_notes:
            overlap = max(0.0, min(end, first + (index + 1) * bar) - max(start, first + index * bar))
            weights[pitch % 12] += overlap
        result.append(max(
            range(7),
            key=lambda degree: sum(
                weights[(tonic + _MAJOR_SCALE[degree] + interval) % 12]
                for interval in _TRIADS["major"][degree]
            ),
        ))
    return result


def plan_harmony(
    melody_notes: list[tuple[float, float, int]],
    tempo_bpm: float = 100.0,
    beats_per_bar: int = 4,
    downbeat_sec: float = 0.0,
) -> HarmonicPlan:
    """Return a deterministic duration-weighted key estimate and shared chord plan.

    ``downbeat_sec`` is the absolute timestamp (in the input melody's seconds)
    of a known bar downbeat. Notes before it are planned in the preceding bar,
    which supports explicitly timed pickups; no automatic meter or downbeat
    inference is attempted.
    """
    _validate_timing(tempo_bpm, beats_per_bar, downbeat_sec)
    _validate_melody(melody_notes)
    tonic, mode = _estimate_key(melody_notes)
    if not melody_notes:
        return HarmonicPlan(tonic, mode, ())

    beat = 60.0 / tempo_bpm
    bar_duration = beats_per_bar * beat
    first_bar = math.floor((min(s for s, _, _ in melody_notes) - downbeat_sec) / bar_duration)
    last_end = max(e for _, e, _ in melody_notes)
    last_bar = math.floor((last_end - downbeat_sec - 1e-12) / bar_duration)
    scale = _MAJOR_SCALE if mode == "major" else _MINOR_SCALE
    candidates = []
    for degree in range(7):
        root = (tonic + scale[degree]) % 12
        intervals = _TRIADS[mode][degree]
        candidates.append((
            degree,
            root,
            tuple(sorted({(root + interval) % 12 for interval in intervals})),
            "dim" if intervals == (0, 3, 6) else ("maj" if intervals == (0, 4, 7) else "min"),
        ))

    bars: list[tuple[int, float, float, list[float]]] = []
    for bar_index in range(first_bar, last_bar + 1):
        start = downbeat_sec + bar_index * bar_duration
        end = start + bar_duration
        weights = [0.0] * 12
        for note_start, note_end, pitch in melody_notes:
            overlap = max(0.0, min(note_end, end) - max(note_start, start))
            weights[pitch % 12] += overlap
        total = sum(weights)
        scores = [
            sum(weights[pitch_class] for pitch_class in chord[2]) / total
            if total else 0.0
            for chord in candidates
        ]
        bars.append((bar_index, start, end, scores))

    costs: list[list[float]] = []
    previous: list[list[int]] = []
    for bar_number, (_, _, _, scores) in enumerate(bars):
        row = []
        back = []
        for candidate_index, score in enumerate(scores):
            if bar_number == 0:
                row.append(score)
                back.append(-1)
                continue
            options = []
            root = candidates[candidate_index][1]
            for prior_index, prior_score in enumerate(costs[-1]):
                prior_root = candidates[prior_index][1]
                if root == prior_root:
                    transition_penalty = 0.0
                else:
                    interval = (root - prior_root) % 12
                    transition_penalty = 0.10 + (0.04 if interval not in (5, 7) else 0.0)
                options.append((prior_score - transition_penalty, -prior_index, prior_index))
            best_prior = max(options)
            row.append(score + best_prior[0])
            back.append(best_prior[2])
        costs.append(row)
        previous.append(back)

    selected = [max(range(7), key=lambda index: (costs[-1][index], -index))]
    for bar_number in range(len(bars) - 1, 0, -1):
        selected.append(previous[bar_number][selected[-1]])
    selected.reverse()
    chords = tuple(
        PlannedChord(bar[0], bar[1], bar[2], candidates[index][0],
                    candidates[index][1], candidates[index][3], candidates[index][2])
        for bar, index in zip(bars, selected)
    )
    return HarmonicPlan(tonic, mode, chords)


def _chord_at(plan: HarmonicPlan, time: float) -> PlannedChord:
    for chord in plan.chords:
        if chord.start <= time < chord.end:
            return chord
    return min(plan.chords, key=lambda chord: abs(chord.start - time))


def _beat_grid(
    melody_notes: list[tuple[float, float, int]], beat: float, downbeat_sec: float
) -> list[float]:
    start = min(s for s, _, _ in melody_notes)
    end = max(e for _, e, _ in melody_notes)
    first = math.ceil((start - downbeat_sec) / beat - 1e-12)
    last = math.ceil((end - downbeat_sec) / beat - 1e-12)
    return [
        downbeat_sec + index * beat
        for index in range(first, last)
        if downbeat_sec + index * beat >= start - 1e-9
    ]


def _melody_pitch_at(time: float, melody_notes: list[tuple[float, float, int]]) -> int:
    sounding = [pitch for start, end, pitch in melody_notes if start <= time < end]
    if sounding:
        return max(sounding)
    return min(melody_notes, key=lambda note: abs(note[0] - time))[2]


def _bounded_note(
    notes: list[Note], start: float, duration: float, pitch: int, velocity: int, melody_end: float
) -> None:
    start = max(start, 0.0)
    end = min(start + duration, melody_end)
    if start < end and 0 <= pitch <= 127:
        notes.append(Note(start, end, pitch, max(1, min(127, velocity))))


def _bass_pitch(chord: PlannedChord, previous: int | None) -> int:
    choices = [
        pitch for pitch in range(36, 53)
        if pitch % 12 in chord.pitch_classes
    ]
    target = previous if previous is not None else 43
    return min(choices, key=lambda pitch: (abs(pitch - target), pitch % 12 != chord.root_pc, pitch))


def _voicing(chord: PlannedChord, melody_pitch: int, bass: int) -> list[int]:
    selected = [bass]
    for target in (55, 64, 72):
        candidates = [
            pitch for pitch in range(36, 80)
            if pitch % 12 in chord.pitch_classes and pitch not in selected
        ]
        below_melody = [pitch for pitch in candidates if pitch <= melody_pitch - 2]
        if below_melody:
            candidates = below_melody
        if not candidates:
            continue
        selected.append(min(candidates, key=lambda pitch: (abs(pitch - target), pitch)))
    return sorted(set(selected))


def _phrase_velocity(base: int, beat_index: int, beats_per_bar: int) -> int:
    phrase_position = (beat_index % (4 * beats_per_bar)) / (4 * beats_per_bar)
    return round(base + 7 * math.sin(math.pi * phrase_position))


def _remove_same_pitch_overlaps(notes: list[Note]) -> list[Note]:
    result: list[Note] = []
    last_for_pitch: dict[int, Note] = {}
    for note in sorted(notes, key=lambda item: (item.start, item.pitch, -item.velocity)):
        previous = last_for_pitch.get(note.pitch)
        if previous is not None and note.start < previous.end:
            if note.start == previous.start:
                previous.end = max(previous.end, note.end)
                previous.velocity = max(previous.velocity, note.velocity)
                continue
            previous.end = note.start
        if note.end > note.start:
            result.append(note)
            last_for_pitch[note.pitch] = note
    return sorted(result, key=lambda item: (item.start, item.pitch))


def generate_accompaniment(
    melody_notes: list[tuple[float, float, int]],
    style: Style,
    tempo_bpm: float = 100.0,
    seed: int = 42,
    beats_per_bar: int = 4,
    downbeat_sec: float = 0.0,
    harmonic_plan: HarmonicPlan | None = None,
) -> list[Note]:
    """Generate an accompaniment without changing the melody.

    The beat grid is anchored to ``downbeat_sec``. That value is the absolute
    time of a known bar downbeat; earlier notes (including a pickup) fall in
    the preceding bar. Meter and downbeat are supplied explicitly: arbitrary
    MIDI does not provide enough information for reliable automatic inference.
    ``seed`` remains accepted for caller compatibility; rule-based output is
    deterministic and does not use randomness.
    """
    del seed
    _validate_style(style)
    _validate_timing(tempo_bpm, beats_per_bar, downbeat_sec)
    _validate_melody(melody_notes)
    if not melody_notes:
        return []
    plan = harmonic_plan or plan_harmony(
        melody_notes, tempo_bpm, beats_per_bar, downbeat_sec
    )
    if not isinstance(plan, HarmonicPlan):
        raise ValueError("harmonic_plan must be a HarmonicPlan")
    if not plan.chords:
        return []

    beat = 60.0 / tempo_bpm
    melody_end = max(end for _, end, _ in melody_notes)
    beats = _beat_grid(melody_notes, beat, downbeat_sec)
    notes: list[Note] = []
    previous_bass = None

    for beat_index, time in enumerate(beats):
        chord = _chord_at(plan, time)
        melody_pitch = _melody_pitch_at(time, melody_notes)
        bass = _bass_pitch(chord, previous_bass)
        previous_bass = bass
        phrase_base = _phrase_velocity(53 if style == "baroque" else 48,
                                       beat_index, beats_per_bar)

        if style == "baroque":
            _bounded_note(notes, time, 0.42 * beat, bass, phrase_base, melody_end)
            if beat_index % 2 == 1:
                for pitch in _voicing(chord, melody_pitch, bass)[1:3]:
                    _bounded_note(notes, time, 0.38 * beat, pitch,
                                  phrase_base - 7, melody_end)
        else:
            voicing = _voicing(chord, melody_pitch, bass)
            if beat_index % beats_per_bar == 0:
                _bounded_note(notes, time, 0.52 * beat, bass,
                              phrase_base + 2, melody_end)
            pattern = voicing[1:] or [bass]
            for position in range(3):
                start = time + position * beat / 3
                pitch = pattern[position % len(pattern)]
                _bounded_note(notes, start, 0.52 * beat, pitch,
                              phrase_base - position * 2, melody_end)

    return _remove_same_pitch_overlaps(notes)


def notes_to_midi(
    melody_notes: list[tuple[float, float, int]],
    accompaniment: list[Note],
    style: Style,
    output_path: str,
    tempo_bpm: float = 100.0,
    beats_per_bar: int = 4,
    downbeat_sec: float = 0.0,
) -> None:
    """Write melody and accompaniment tracks to a MIDI file."""
    import mido

    _validate_style(style)
    _validate_timing(tempo_bpm, beats_per_bar, downbeat_sec)
    _validate_melody(melody_notes)
    melody_end = max((end for _, end, _ in melody_notes), default=0.0)
    for note in accompaniment:
        if (isinstance(note.pitch, bool) or not isinstance(note.pitch, int)
                or isinstance(note.velocity, bool) or not isinstance(note.velocity, int)
                or not 0 <= note.pitch <= 127 or not 1 <= note.velocity <= 127
                or isinstance(note.start, bool) or not isinstance(note.start, (int, float))
                or isinstance(note.end, bool) or not isinstance(note.end, (int, float))
                or not math.isfinite(note.start) or not math.isfinite(note.end)
                or note.start < 0 or note.end <= note.start
                or note.end > melody_end + 1e-9):
            raise ValueError(f"invalid accompaniment note for MIDI export: {note}")

    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    tempo_us = int(60_000_000 / tempo_bpm)

    def sec_to_tick(sec: float) -> int:
        return round(sec * tempo_bpm / 60.0 * mid.ticks_per_beat)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    tempo_track.append(mido.MetaMessage(
        "time_signature", numerator=beats_per_bar, denominator=4, time=0
    ))
    tempo_track.append(mido.MetaMessage(
        "track_name", name=f"layer_a_scripted_{style}", time=0
    ))
    mid.tracks.append(tempo_track)

    def make_track(note_list: list, channel: int, name: str) -> mido.MidiTrack:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        events: list[tuple[int, int, int, int]] = []
        for item in note_list:
            if isinstance(item, Note):
                start, end, pitch, velocity = item.start, item.end, item.pitch, item.velocity
            else:
                start, end, pitch = item
                velocity = 80
            if start < 0 or end <= start:
                raise ValueError("MIDI note times must satisfy 0 <= start < end")
            start_tick = sec_to_tick(start)
            end_tick = max(start_tick + 1, sec_to_tick(end))
            if not 0 <= pitch <= 127 or not 1 <= velocity <= 127:
                raise ValueError("MIDI pitch and velocity must be from 0 to 127")
            events.append((start_tick, 1, pitch, velocity))
            events.append((end_tick, 0, pitch, 0))
        events.sort(key=lambda event: (event[0], event[1]))
        previous_tick = 0
        for tick, on, pitch, velocity in events:
            delta = tick - previous_tick
            if delta < 0:
                raise ValueError("MIDI events would produce a negative delta time")
            if on:
                track.append(mido.Message(
                    "note_on", channel=channel, note=pitch, velocity=velocity, time=delta
                ))
            else:
                track.append(mido.Message(
                    "note_off", channel=channel, note=pitch, velocity=0, time=delta
                ))
            previous_tick = tick
        return track

    mid.tracks.append(make_track(melody_notes, 0, "melody"))
    mid.tracks.append(make_track(
        accompaniment, 1, f"layer_a_scripted_{style}"
    ))
    mid.save(output_path)


TWINKLE_MELODY: list[tuple[float, float, int]] = [
    (0.0, 0.6, 60), (0.6, 1.2, 60), (1.2, 1.8, 67), (1.8, 2.4, 67),
    (2.4, 3.0, 69), (3.0, 3.6, 69), (3.6, 4.8, 67), (4.8, 5.4, 65),
    (5.4, 6.0, 65), (6.0, 6.6, 64), (6.6, 7.2, 64), (7.2, 7.8, 62),
    (7.8, 8.4, 62), (8.4, 9.6, 60),
]


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(output_dir, exist_ok=True)
    for selected_style in ("baroque", "romantic"):
        accompaniment = generate_accompaniment(TWINKLE_MELODY, selected_style)
        path = os.path.join(
            output_dir, f"twinkle_layer_a_scripted_{selected_style}.mid"
        )
        notes_to_midi(TWINKLE_MELODY, accompaniment, selected_style, path)
        print(f"Saved {path} ({len(accompaniment)} accompaniment notes)")
