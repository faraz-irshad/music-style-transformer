"""CPU tests for token alignment, chunking, generation, and safe experiment paths."""
from __future__ import annotations

import importlib
import math
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.training.sequence import (
    ContextOverflowError,
    SequenceSettings,
    UnconsumedControlsError,
    chunk_sequence,
    example_counts,
    interleave_anticipated_controls,
    pad_event_groups,
    token_weighted_loss_weights,
)


@pytest.fixture
def amt_settings():
    time_resolution = 100
    max_time = 10000
    max_duration = 1000
    max_note = 16512
    time_offset = 0
    duration_offset = max_time
    note_offset = duration_offset + max_duration
    rest = note_offset + max_note
    control_offset = rest + 1
    atime_offset = control_offset
    aduration_offset = atime_offset + max_time
    return SequenceSettings(
        control_offset=control_offset,
        rest_token=rest,
        time_offset=time_offset,
        duration_offset=duration_offset,
        anticipate_time_offset=atime_offset,
        anticipate_duration_offset=aduration_offset,
        time_resolution=time_resolution,
        anticipation_delta=5 * time_resolution,
        padding_density=time_resolution,
    )


def _upstream_anticipate(events, controls, delta, settings):
    remaining = [tuple(controls[i:i + 3]) for i in range(0, len(controls), 3)]
    output = []
    event_time = 0
    for group in (events[i:i + 3] for i in range(0, len(events), 3)):
        while remaining:
            control_time = remaining[0][0] - settings.anticipate_time_offset
            if event_time < control_time - delta:
                break
            output.extend(remaining.pop(0))
        event_time = group[0] - settings.time_offset
        output.extend(group)
    return output, [token for group in remaining for token in group]


def test_rest_padding_matches_upstream_density_and_includes_sparse_end(amt_settings):
    events = [0, 10020, 11060, 350, 10030, 11064]
    padded = pad_event_groups(events, 600, amt_settings)
    assert len(padded) % 3 == 0
    groups = [padded[i:i + 3] for i in range(0, len(padded), 3)]
    assert groups[:3] == [
        [0, 10020, 11060],
        [100, 10000, amt_settings.rest_token],
        [200, 10000, amt_settings.rest_token],
    ]
    assert groups[-1] == [550, 10000, amt_settings.rest_token]


def test_interleave_matches_upstream_order_at_anticipation_boundaries(amt_settings):
    events = [
        0, 10020, 11060,
        600, 10020, 11064,
        700, 10020, 11067,
    ]
    event_times = (0, 500, 501, 1000, 2000)
    controls = [
        token
        for onset in event_times
        for token in (
            amt_settings.anticipate_time_offset + onset,
            amt_settings.anticipate_duration_offset + 100,
            amt_settings.control_offset + 11000 + 60,
        )
    ]
    actual, remaining = interleave_anticipated_controls(
        events, controls, amt_settings
    )
    expected, expected_remaining = _upstream_anticipate(
        events, controls, amt_settings.anticipation_delta, amt_settings
    )
    assert actual == expected
    assert remaining == expected_remaining
    assert [g[0] for g in (
        tuple(actual[i:i + 3]) for i in range(0, len(actual), 3)
    ) if g[2] >= amt_settings.control_offset][:2] == [
        amt_settings.anticipate_time_offset,
        amt_settings.anticipate_time_offset + 500,
    ]
    assert len(remaining) == 3


def test_rest_padding_consumes_every_control_or_reports_leftovers(amt_settings):
    control = [
        amt_settings.anticipate_time_offset,
        amt_settings.anticipate_duration_offset + 100,
        amt_settings.control_offset + 11060,
    ]
    padded = pad_event_groups([], 1000, amt_settings)
    stream, remaining = interleave_anticipated_controls(
        padded, control, amt_settings
    )
    assert not remaining
    assert control[2] in stream
    delayed = [
        amt_settings.anticipate_time_offset + 1000,
        amt_settings.anticipate_duration_offset + 100,
        amt_settings.control_offset + 11060,
    ]
    _stream, remaining = interleave_anticipated_controls(
        [0, 10020, 11060], delayed, amt_settings
    )
    assert remaining == delayed
    with pytest.raises(UnconsumedControlsError, match="1 melody control"):
        raise UnconsumedControlsError(len(remaining) // 3)


def test_chunking_masks_context_and_controls_and_supervises_every_target_once(amt_settings):
    groups = []
    for index in range(27):
        onset = index * 100
        if index % 4 == 0:
            groups.append((
                amt_settings.anticipate_time_offset + onset,
                amt_settings.anticipate_duration_offset + 100,
                amt_settings.control_offset + 11000 + 70,
            ))
        pitch = amt_settings.rest_token if index % 3 == 0 else 11000 + 60 + index % 12
        groups.append((onset, 10000 + 50, pitch))
    stream = [token for group in groups for token in group]
    examples = chunk_sequence(
        stream, mode_token=40000, sequence_length=25,
        settings=amt_settings, context_groups=4,
    )
    assert len(examples) > 1
    supervised_tokens = 0
    for ids, labels, attention in examples:
        assert len(ids) == len(labels) == len(attention) == 25
        assert ids[0] == 40000 and labels[0] == -100 and attention[0] == 1
        active_length = sum(attention)
        assert (active_length - 1) % 3 == 0
        assert all(mask == 0 and label == -100
                   for mask, label in zip(attention[active_length:], labels[active_length:]))
        for idx in range(1, active_length, 3):
            is_control = ids[idx + 2] >= amt_settings.control_offset
            if is_control:
                assert labels[idx + 2] == -100
            elif labels[idx + 2] != -100:
                assert labels[idx + 2] == ids[idx + 2]
        supervised_tokens += sum(label != -100 for label in labels)
    expected_target_groups = sum(
        group[2] < amt_settings.control_offset for group in groups
    )
    assert supervised_tokens == expected_target_groups * 3
    assert example_counts(examples, amt_settings)["all_masked_examples"] == 0


def test_later_chunk_retains_controls_inside_anticipation_context(amt_settings):
    groups = []
    for index in range(35):
        onset = index * 100
        if index in (0, 5, 8, 15, 23, 30):
            groups.append((
                amt_settings.anticipate_time_offset + onset,
                amt_settings.anticipate_duration_offset + 200,
                amt_settings.control_offset + 11100 + index,
            ))
        groups.append((onset, 10010, 11060 + index % 10))
    stream = [token for group in groups for token in group]
    examples = chunk_sequence(stream, 40000, 25, amt_settings, context_groups=3)
    assert len(examples) > 2
    second_ids, second_labels, _ = examples[2]
    relevant_control = amt_settings.control_offset + 11100 + 5
    assert relevant_control in second_ids
    assert all(
        label == -100
        for token, label in zip(second_ids, second_labels)
        if token >= amt_settings.control_offset
    )


def test_chunk_overflow_is_explicit(amt_settings):
    groups = [
        (amt_settings.control_offset + i, 0, amt_settings.control_offset + 200 + i)
        for i in range(10)
    ]
    groups.append((0, 10000, 11060))
    stream = [token for group in groups for token in group]
    with pytest.raises(ContextOverflowError):
        chunk_sequence(stream, 40000, 10, amt_settings, context_groups=4)


def test_empty_target_stream_has_no_fake_supervision(amt_settings):
    assert chunk_sequence([], 40000, 1024, amt_settings) == []


def test_gradient_accumulation_weights_mean_losses_by_supervised_tokens():
    weights = token_weighted_loss_weights([2, 6])
    assert weights == [0.25, 0.75]
    assert sum(weights) == 1.0
    assert sum(loss * weight for loss, weight in zip((1.0, 3.0), weights)) == 2.5


@pytest.mark.parametrize("counts", [[], [0, 0], [1, -1]])
def test_gradient_accumulation_rejects_invalid_token_counts(counts):
    with pytest.raises(ValueError, match="supervised tokens"):
        token_weighted_loss_weights(counts)


def _install_fake_anticipation(monkeypatch, settings):
    anticipation = types.ModuleType("anticipation")
    anticipation.__path__ = []
    ops = types.ModuleType("anticipation.ops")
    sample = types.ModuleType("anticipation.sample")
    vocab = types.ModuleType("anticipation.vocab")
    config = types.ModuleType("anticipation.config")
    for name, value in {
        "TIME_OFFSET": settings.time_offset,
        "DUR_OFFSET": settings.duration_offset,
        "NOTE_OFFSET": 11000,
        "CONTROL_OFFSET": settings.control_offset,
        "ATIME_OFFSET": settings.anticipate_time_offset,
        "ADUR_OFFSET": settings.anticipate_duration_offset,
        "ANOTE_OFFSET": settings.anticipate_duration_offset + 1000,
        "ANTICIPATE": 40000,
        "AUTOREGRESS": 40001,
    }.items():
        setattr(vocab, name, value)
    config.DELTA = 5
    config.TIME_RESOLUTION = 100

    def groups(tokens):
        return [tuple(tokens[i:i + 3]) for i in range(0, len(tokens), 3)]

    def max_time(tokens, seconds=True):
        values = []
        for group in groups(tokens):
            offset = (
                settings.anticipate_time_offset
                if group[2] >= settings.control_offset else settings.time_offset
            )
            values.append(group[0] - offset)
        result = max(values, default=0)
        return result / 100 if seconds else result

    def clip(tokens, start, end, clip_duration=True, seconds=True):
        if seconds:
            start, end = int(start * 100), int(end * 100)
        output = []
        for time, duration, note in groups(tokens):
            offset = settings.anticipate_time_offset if note >= settings.control_offset else 0
            onset = time - offset
            if start <= onset <= end:
                output.extend((time, duration, note))
        return output

    def sort(tokens):
        return [t for group in sorted(groups(tokens), key=lambda g: g[0]) for t in group]

    def pad(tokens, end_time):
        return list(tokens)

    def anticipate(events, controls, delta=500):
        remaining = groups(controls)
        output = []
        event_time = 0
        for event in groups(events):
            while remaining and event_time >= remaining[0][0] - settings.anticipate_time_offset - delta:
                output.extend(remaining.pop(0))
            event_time = event[0] - settings.time_offset
            output.extend(event)
        return output, [token for group in remaining for token in group]

    def split(tokens):
        event_tokens, control_tokens = [], []
        for group in groups(tokens):
            target = control_tokens if group[2] >= settings.control_offset else event_tokens
            target.extend(group)
        return event_tokens, control_tokens

    ops.clip = clip
    ops.max_time = max_time
    ops.sort = sort
    ops.pad = pad
    ops.anticipate = anticipate
    ops.split = split
    ops.unpad = lambda tokens: list(tokens)
    seen = []

    def add_token(model, mode, tokens, top_p, current_time, debug=False):
        seen.append((mode, list(tokens), current_time))
        return [settings.time_offset + 101, settings.duration_offset, 11060]

    sample.add_token = add_token
    anticipation.ops = ops
    anticipation.sample = sample
    anticipation.vocab = vocab
    anticipation.config = config
    for name, module in (
        ("anticipation", anticipation), ("anticipation.ops", ops),
        ("anticipation.sample", sample), ("anticipation.vocab", vocab),
        ("anticipation.config", config),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return seen, settings


def test_generation_passes_time_zero_controls_to_actual_sampler(monkeypatch, amt_settings):
    seen, settings = _install_fake_anticipation(monkeypatch, amt_settings)
    module = importlib.import_module("model.generation.anticipation_compat")
    control = [
        settings.anticipate_time_offset,
        settings.anticipate_duration_offset + 100,
        settings.control_offset + settings.anticipate_duration_offset + 1060,
    ]
    module.generate_conditioned(
        model=object(), start_time=0, end_time=1,
        controls=control, top_p=0.9,
    )
    assert seen
    assert control[0] in seen[0][1]
    assert seen[0][0] == [40000]


def test_rest_is_not_removed_by_piano_sampling_mask(monkeypatch):
    from model.generation.sampling import constrained_sampling

    anticipation = types.ModuleType("anticipation")
    sample = types.ModuleType("anticipation.sample")
    vocab = types.ModuleType("anticipation.vocab")
    vocab.NOTE_OFFSET = 100
    vocab.MAX_NOTE = 160
    vocab.MAX_PITCH = 128
    vocab.REST = vocab.NOTE_OFFSET + vocab.MAX_NOTE
    vocab.TIME_OFFSET = 400
    vocab.MAX_TIME = 1000
    vocab.TIME_RESOLUTION = 10
    anticipation.sample = sample
    anticipation.vocab = vocab
    sample.safe_logits = lambda logits, idx: logits
    sample.future_logits = lambda logits, curtime: logits
    monkeypatch.setitem(sys.modules, "anticipation", anticipation)
    monkeypatch.setitem(sys.modules, "anticipation.sample", sample)
    monkeypatch.setitem(sys.modules, "anticipation.vocab", vocab)
    with constrained_sampling():
        logits = [0.0] * 400
        sample.safe_logits(logits, 2)
        assert logits[vocab.NOTE_OFFSET + 127] == 0.0
        assert logits[vocab.NOTE_OFFSET + 128] == float("-inf")
        assert logits[vocab.REST] == 0.0


def test_explicit_shorter_generation_interval_preserves_full_melody(monkeypatch):
    torch_stub = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    generation = importlib.import_module("model.generation.generate")
    anticipation = types.ModuleType("anticipation")
    anticipation.__path__ = []
    ops = types.ModuleType("anticipation.ops")
    vocab = types.ModuleType("anticipation.vocab")
    vocab.CONTROL_OFFSET = 10000
    ops.clip = lambda tokens, start, end, seconds=True: [
        token for group in (
            tokens[i:i + 3] for i in range(0, len(tokens), 3)
        )
        if start * 100 <= group[0] <= end * 100
        for token in group
    ]
    def combine(events, controls):
        control_groups = [
            tuple(token - vocab.CONTROL_OFFSET for token in controls[i:i + 3])
            for i in range(0, len(controls), 3)
        ]
        event_groups = [
            tuple(events[i:i + 3]) for i in range(0, len(events), 3)
        ]
        return [
            token
            for group in sorted(event_groups + control_groups, key=lambda g: g[0])
            for token in group
        ]

    ops.combine = combine
    anticipation.ops = ops
    anticipation.vocab = vocab
    monkeypatch.setitem(sys.modules, "anticipation", anticipation)
    monkeypatch.setitem(sys.modules, "anticipation.ops", ops)
    monkeypatch.setitem(sys.modules, "anticipation.vocab", vocab)
    full_melody = [
        0, 100, 60,
        900, 200, 64,
    ]
    controls = [token + vocab.CONTROL_OFFSET for token in full_melody]
    result = generation._combine_and_slice(
        generated_events=[100, 50, 67, 700, 50, 69],
        controls=controls,
        duration=5.0,
    )
    assert 900 in result
    assert 700 not in result


def test_saved_melody_round_trip_preserves_notes_within_midi_tick(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    generation = importlib.import_module("model.generation.generate")
    pretty_midi = pytest.importorskip("pretty_midi")
    melody = [(0.1234, 0.5678, 60), (2.0, 2.4, 72)]
    path = tmp_path / "source.mid"

    generation.save_melody_midi(melody, str(path))

    exported = pretty_midi.PrettyMIDI(str(path))
    notes = sorted(
        (note.start, note.end, note.pitch)
        for instrument in exported.instruments
        for note in instrument.notes
    )
    assert len(notes) == len(melody)
    tick_seconds = 60 / (120 * exported.resolution)
    for (start, end, pitch), (source_start, source_end, source_pitch) in zip(
        notes, melody
    ):
        assert pitch == source_pitch
        assert abs(start - source_start) <= tick_seconds
        assert abs(end - source_end) <= tick_seconds


def test_default_generation_endpoint_uses_last_melody_note():
    torch_stub = types.ModuleType("torch")
    sys.modules.setdefault("torch", torch_stub)
    generation = importlib.import_module("model.generation.generate")
    assert generation.melody_endpoint([(0.0, 0.6, 60), (3.0, 9.6, 64)]) == 9.6
    assert generation.melody_endpoint([]) == 0.0


def test_nll_aggregation_is_weighted_by_supervised_token_count(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    module = importlib.import_module("model.evaluation.nll_eval")
    result = module.mean_scores([
        {
            "total": 2.0, "valid_tokens": 1,
            "time": 2.0, "time_tokens": 1,
            "duration": 2.0, "duration_tokens": 1,
            "note": 2.0, "note_tokens": 1,
        },
        {
            "total": 10.0, "valid_tokens": 9,
            "time": 10.0, "time_tokens": 9,
            "duration": 10.0, "duration_tokens": 9,
            "note": 10.0, "note_tokens": 9,
        },
    ])
    assert result["total"] == pytest.approx(9.2)
    assert result["valid_tokens"] == 10


def test_cache_signature_changes_with_objective_or_settings(monkeypatch, amt_settings):
    import model.training.train_lora as training

    anticipation = types.ModuleType("anticipation")
    anticipation.__path__ = []
    config = types.ModuleType("anticipation.config")
    vocab = types.ModuleType("anticipation.vocab")
    for name in (
        "control_offset", "rest_token", "time_offset", "duration_offset",
        "anticipate_time_offset", "anticipate_duration_offset",
    ):
        setattr(vocab, {
            "control_offset": "CONTROL_OFFSET",
            "rest_token": "REST",
            "time_offset": "TIME_OFFSET",
            "duration_offset": "DUR_OFFSET",
            "anticipate_time_offset": "ATIME_OFFSET",
            "anticipate_duration_offset": "ADUR_OFFSET",
        }[name], getattr(amt_settings, name))
    config.TIME_RESOLUTION = 100
    config.DELTA = 5
    anticipation.config = config
    anticipation.vocab = vocab
    monkeypatch.setitem(sys.modules, "anticipation", anticipation)
    monkeypatch.setitem(sys.modules, "anticipation.config", config)
    monkeypatch.setitem(sys.modules, "anticipation.vocab", vocab)
    with tempfile.NamedTemporaryFile() as midi:
        windows = [{
            "midi_path": midi.name, "start_sec": 0.0,
            "end_sec": 15.0, "window_id": "w0",
        }]
        unconditional = training._cache_path(
            Path("cache"), "baroque", "val", 42, windows,
            amt_settings.control_offset, 1024, mode="unconditional",
        )
        conditioned = training._cache_path(
            Path("cache"), "baroque", "val", 42, windows,
            amt_settings.control_offset, 1024, mode="conditioned",
        )
        different_length = training._cache_path(
            Path("cache"), "baroque", "val", 42, windows,
            amt_settings.control_offset, 512, mode="unconditional",
        )
        different_context = training._cache_path(
            Path("cache"), "baroque", "val", 42, windows,
            amt_settings.control_offset, 1024, mode="unconditional",
            context_groups=80,
        )
    assert len({
        unconditional, conditioned, different_length, different_context,
    }) == 4


def test_new_training_refuses_to_overwrite_and_legacy_adapter_is_resolvable():
    from model.generation.adapter_paths import resolve_adapter_path
    from model.training.experiment import new_experiment_root, prepare_new_adapter_dir

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        experiment = new_experiment_root(
            {"experiment_name": "new_run"}, str(root / "drive")
        )
        adapter = prepare_new_adapter_dir(experiment, "baroque")
        assert adapter.is_dir()
        marker = adapter / "resume_step.txt"
        marker.write_text("30")
        with pytest.raises(FileExistsError, match="Incomplete optimizer state"):
            prepare_new_adapter_dir(experiment, "baroque")
        assert marker.read_text() == "30"

        legacy = root / "drive" / "adapter_romantic" / "final"
        legacy.mkdir(parents=True)
        assert resolve_adapter_path(
            root / "drive", experiment, "romantic"
        ) == legacy
        revised = experiment / "adapter_romantic" / "final"
        revised.mkdir(parents=True)
        assert resolve_adapter_path(
            root / "drive", experiment, "romantic"
        ) == revised
