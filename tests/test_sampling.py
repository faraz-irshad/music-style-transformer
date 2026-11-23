"""CPU-only tests for temporary anticipation sampling constraints."""
import sys
import types


def test_constrained_sampling_patches_and_restores(monkeypatch):
    from model.generation.sampling import constrained_sampling

    anticipation = types.ModuleType("anticipation")
    sample = types.ModuleType("anticipation.sample")
    vocab = types.ModuleType("anticipation.vocab")
    vocab.NOTE_OFFSET = 100
    vocab.MAX_NOTE = 160
    vocab.TIME_OFFSET = 400
    vocab.MAX_TIME = 1000
    vocab.TIME_RESOLUTION = 10
    anticipation.sample = sample
    anticipation.vocab = vocab

    def safe_logits(logits, idx):
        return logits

    def future_logits(logits, curtime):
        return logits

    sample.safe_logits = safe_logits
    sample.future_logits = future_logits
    monkeypatch.setitem(sys.modules, "anticipation", anticipation)
    monkeypatch.setitem(sys.modules, "anticipation.sample", sample)
    monkeypatch.setitem(sys.modules, "anticipation.vocab", vocab)

    with constrained_sampling():
        note_logits = [0.0] * 300
        sample.safe_logits(note_logits, 2)
        assert note_logits[227] == 0.0
        assert note_logits[228:260] == [float("-inf")] * 32
        assert note_logits[260] == 0.0

        other_logits = [0.0] * 300
        sample.safe_logits(other_logits, 1)
        assert other_logits == [0.0] * 300

        time_logits = [0.0] * 1401
        sample.future_logits(time_logits, 5)
        assert time_logits[415] == 0.0
        assert time_logits[416:1400] == [float("-inf")] * (1400 - 416)
        assert time_logits[1400] == 0.0

    assert sample.safe_logits is safe_logits
    assert sample.future_logits is future_logits

    try:
        with constrained_sampling(max_gap_ticks=3):
            raise RuntimeError("exercise exceptional exit")
    except RuntimeError as error:
        assert str(error) == "exercise exceptional exit"
    else:
        raise AssertionError("Expected the context body to raise")
    assert sample.safe_logits is safe_logits
    assert sample.future_logits is future_logits
