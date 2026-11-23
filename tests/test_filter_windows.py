from model.data.filter_maestro import cut_windows


def _row(duration: float):
    return {
        "midi_filename": "piece.mid",
        "duration": duration,
        "style": "baroque",
        "split": "train",
        "canonical_composer": "Bach",
        "canonical_title": "Piece",
    }


def test_windows_are_seeded_and_sample_across_performance():
    row = _row(1200)

    first = cut_windows(row, "/midi", 15, 5, seed=42)
    repeat = cut_windows(row, "/midi", 15, 5, seed=42)
    starts = [window["start_sec"] for window in first]

    assert first == repeat
    assert len(first) == 5
    assert starts == sorted(set(starts))
    assert starts[0] < 15 * 16
    assert starts[-1] >= 15 * 64


def test_windows_change_with_seed_and_never_exceed_duration():
    row = _row(1200)

    first = cut_windows(row, "/midi", 15, 5, seed=42)
    second = cut_windows(row, "/midi", 15, 5, seed=43)

    assert [w["start_sec"] for w in first] != [w["start_sec"] for w in second]
    assert all(w["end_sec"] <= row["duration"] for w in first + second)


def test_short_performance_has_no_full_windows():
    assert cut_windows(_row(14.9), "/midi", 15, 5, seed=42) == []
