"""Integration checks against the pinned Anticipation package, never test doubles."""
from __future__ import annotations

import math
import json
from importlib import metadata
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from model.training.sequence import (
    SequenceSettings,
    interleave_anticipated_controls,
    pad_event_groups,
)


@pytest.fixture(scope="module")
def real_stack():
    anticipation = pytest.importorskip(
        "anticipation", reason="real integration requires pinned Anticipation"
    )
    torch = pytest.importorskip(
        "torch", reason="real sampling integration requires CPU PyTorch"
    )
    try:
        distribution = metadata.distribution("anticipation")
    except metadata.PackageNotFoundError:
        pytest.skip("cannot verify pinned Anticipation distribution metadata")
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        pytest.skip("Anticipation install does not expose its pinned VCS revision")
    vcs_info = json.loads(direct_url).get("vcs_info", {})
    if vcs_info.get("commit_id") != "af37397922665a0fb8d474d7988b0f3755a38d45":
        pytest.skip(
            "real integration requires Anticipation commit "
            "af37397922665a0fb8d474d7988b0f3755a38d45"
        )
    from anticipation import ops, sample, vocab
    from anticipation import config

    return SimpleNamespace(
        anticipation=anticipation,
        torch=torch,
        ops=ops,
        sample=sample,
        vocab=vocab,
        config=config,
    )


def _settings(real_stack) -> SequenceSettings:
    vocab, config = real_stack.vocab, real_stack.config
    return SequenceSettings(
        control_offset=vocab.CONTROL_OFFSET,
        rest_token=vocab.REST,
        time_offset=vocab.TIME_OFFSET,
        duration_offset=vocab.DUR_OFFSET,
        anticipate_time_offset=vocab.ATIME_OFFSET,
        anticipate_duration_offset=vocab.ADUR_OFFSET,
        time_resolution=config.TIME_RESOLUTION,
        anticipation_delta=config.DELTA * config.TIME_RESOLUTION,
        padding_density=config.TIME_RESOLUTION,
    )


@pytest.mark.parametrize(
    "events,end_time",
    [
        ([], 0),
        ([], 7),
        ([0, 10020, 11060, 350, 10030, 11064], 600),
        ([0, 10020, 11060, 800, 10030, 11064], 1200),
    ],
)
def test_real_pad_matches_pinned_ops(real_stack, events, end_time):
    settings = _settings(real_stack)
    expected = real_stack.ops.pad(
        list(events), end_time, density=settings.padding_density
    )
    actual = pad_event_groups(events, end_time, settings)
    assert actual == expected


def test_real_anticipate_matches_boundaries_sparse_and_leftovers(real_stack):
    settings = _settings(real_stack)
    events = [
        0, settings.duration_offset + 20, real_stack.vocab.NOTE_OFFSET + 60,
        settings.anticipation_delta,
        settings.duration_offset + 20,
        real_stack.vocab.NOTE_OFFSET + 62,
        settings.anticipation_delta + 1,
        settings.duration_offset + 20,
        real_stack.vocab.NOTE_OFFSET + 64,
    ]
    onsets = (
        0,
        settings.anticipation_delta,
        settings.anticipation_delta + 1,
        2 * settings.anticipation_delta,
        10 * settings.anticipation_delta,
    )
    controls = [
        token
        for onset in onsets
        for token in (
            settings.anticipate_time_offset + onset,
            settings.anticipate_duration_offset + 20,
            settings.control_offset + real_stack.vocab.NOTE_OFFSET + 70,
        )
    ]
    expected, expected_remaining = real_stack.ops.anticipate(
        list(events), list(controls), delta=settings.anticipation_delta
    )
    actual, remaining = interleave_anticipated_controls(
        events, controls, settings
    )
    assert (actual, remaining) == (expected, expected_remaining)
    assert len(remaining) % 3 == 0

    empty_actual, empty_remaining = interleave_anticipated_controls(
        [], controls, settings
    )
    empty_expected, empty_expected_remaining = real_stack.ops.anticipate(
        [], list(controls), delta=settings.anticipation_delta
    )
    assert (empty_actual, empty_remaining) == (
        empty_expected,
        empty_expected_remaining,
    )


def test_real_synthetic_midi_token_round_trip_with_tick_tolerance(real_stack):
    import pretty_midi
    from anticipation.convert import midi_to_events
    from model.amt_utils import events_to_pretty_midi, write_notes_to_midi
    from model.data.skyline import Note

    source = [
        Note(0.137, 0.493, 60),
        Note(1.111, 1.777, 67),
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / "source.mid"
        write_notes_to_midi(source, str(source_path), tempo=100.0)
        tokens = list(midi_to_events(str(source_path)))
        recovered = events_to_pretty_midi(tokens)

    recovered_notes = sorted(
        (note.start, note.end, note.pitch)
        for instrument in recovered.instruments
        for note in instrument.notes
    )
    assert len(recovered_notes) == len(source)
    token_quantum = 1 / real_stack.config.TIME_RESOLUTION
    midi_tick = 60 / (100 * 480)
    tolerance = token_quantum + midi_tick
    for (start, end, pitch), original in zip(recovered_notes, source):
        assert pitch == original.pitch
        assert abs(start - original.start) <= tolerance
        assert abs(end - original.end) <= tolerance
    assert math.isfinite(recovered.get_end_time())


def test_time_zero_control_reaches_real_model_forward(real_stack, monkeypatch):
    from model.amt_utils import melody_notes_to_controls
    from model.generation.anticipation_compat import generate_conditioned
    from model.generation.sampling import constrained_sampling

    torch, vocab, config = real_stack.torch, real_stack.vocab, real_stack.config
    melody = [(0.0, 0.2, 60)]
    controls = melody_notes_to_controls(melody, vocab.CONTROL_OFFSET)
    endpoint = 0.2
    endpoint_tick = int(config.TIME_RESOLUTION * endpoint)
    seen_inputs = []
    vocab_size = max(
        vocab.SPECIAL_OFFSET,
        vocab.ANOTE_OFFSET + vocab.MAX_NOTE,
        vocab.AUTOREGRESS,
        vocab.ANTICIPATE,
    ) + 1

    class RecordingModel:
        device = "cpu"

        def __call__(self, input_ids):
            seen_inputs.append(input_ids.detach().clone())
            logits = torch.full(
                (input_ids.shape[0], input_ids.shape[1], vocab_size),
                -100.0,
            )
            step = input_ids.shape[1] - 1
            if step % 3 == 0:
                token = vocab.TIME_OFFSET + endpoint_tick
            elif step % 3 == 1:
                token = vocab.DUR_OFFSET + 20
            else:
                token = vocab.NOTE_OFFSET + 60
            logits[:, -1, token] = 100.0
            return SimpleNamespace(logits=logits)

    model = RecordingModel()
    with constrained_sampling():
        generate_conditioned(
            model,
            start_time=0,
            end_time=endpoint,
            controls=controls,
            max_events=3,
        )

    assert seen_inputs
    observed = {
        int(token)
        for input_ids in seen_inputs
        for token in input_ids.reshape(-1)
    }
    assert controls[0] in observed
    assert controls[1] in observed
    assert controls[2] in observed


def test_real_sampling_masks_keep_rest_and_remove_non_piano_notes(real_stack):
    from model.generation.sampling import constrained_sampling

    vocab, torch = real_stack.vocab, real_stack.torch
    logits = torch.zeros(vocab.SPECIAL_OFFSET + 1)
    with constrained_sampling():
        real_stack.sample.safe_logits(logits, 2)

    assert torch.isfinite(logits[vocab.NOTE_OFFSET + 60])
    assert torch.isneginf(logits[vocab.NOTE_OFFSET + 128])
    assert torch.isfinite(logits[vocab.REST])
