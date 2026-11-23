from types import SimpleNamespace

from model.evaluation.evaluate import split_by_known_melody
from model.data.features import split_skyline


def _note(start: float, pitch: int):
    return SimpleNamespace(start=start, end=start + 0.4, pitch=pitch, velocity=80)


def test_matches_onsets_within_25ms_and_preserves_accompaniment():
    output = [_note(1.02, 60), _note(1.0, 48)]
    reference = [_note(1.0, 60)]

    melody, accompaniment, fraction = split_by_known_melody(output, reference)

    assert melody == [output[0]]
    assert accompaniment == [output[1]]
    assert fraction == 1.0


def test_each_output_note_matches_at_most_one_reference_note():
    output = [_note(1.01, 60)]
    reference = [_note(1.0, 60), _note(1.02, 60)]

    melody, accompaniment, fraction = split_by_known_melody(output, reference)

    assert len(melody) == 1
    assert accompaniment == []
    assert fraction == 0.5


def test_notes_outside_tolerance_remain_unmatched():
    output = [_note(1.026, 60)]
    reference = [_note(1.0, 60)]

    melody, accompaniment, fraction = split_by_known_melody(output, reference)

    assert melody == []
    assert accompaniment == output
    assert fraction == 0.0


def test_empty_reference_has_no_match_fraction():
    melody, accompaniment, fraction = split_by_known_melody(
        [_note(1.0, 60)], []
    )

    assert melody == []
    assert len(accompaniment) == 1
    assert fraction != fraction


def test_skyline_split_removes_only_the_selected_melody_notes():
    low = _note(0.0, 48)
    high = _note(0.0, 72)
    later_low = _note(0.5, 50)

    melody, accompaniment = split_skyline([low, high, later_low])

    assert melody == [high, later_low]
    assert accompaniment == [low]
