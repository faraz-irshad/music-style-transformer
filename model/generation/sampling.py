"""Temporary constraints for Anticipation Music Transformer sampling."""
from __future__ import annotations

from contextlib import contextmanager
import inspect
from typing import Iterator


def _argument(
    function, args: tuple, kwargs: dict, name: str, position: int
):
    bound = inspect.signature(function).bind_partial(*args, **kwargs)
    if name in bound.arguments:
        return bound.arguments[name]
    if position < len(args):
        return args[position]
    raise TypeError(
        f"Cannot locate {name!r} in {function.__module__}.{function.__name__}"
    )


def _mask_range(logits, start: int, end: int) -> None:
    if end <= start:
        return
    if isinstance(logits, list):
        logits[start:end] = [float("-inf")] * (end - start)
    else:
        logits[..., start:end] = float("-inf")


@contextmanager
def constrained_sampling(max_gap_ticks: int | None = None) -> Iterator[None]:
    """Constrain sampled notes to piano and bound gaps between event times."""
    from anticipation import vocab
    sample = __import__("anticipation.sample", fromlist=["sample"])

    max_gap = vocab.TIME_RESOLUTION if max_gap_ticks is None else max_gap_ticks
    if max_gap < 0:
        raise ValueError("max_gap_ticks must be non-negative")

    original_safe_logits = sample.safe_logits
    original_future_logits = sample.future_logits

    def safe_logits(*args, **kwargs):
        logits = _argument(original_safe_logits, args, kwargs, "logits", 0)
        idx = _argument(original_safe_logits, args, kwargs, "idx", 1)
        result = original_safe_logits(*args, **kwargs)
        if int(idx) % 3 == 2:
            _mask_range(
                logits,
                vocab.NOTE_OFFSET + 128,
                vocab.NOTE_OFFSET + vocab.MAX_NOTE,
            )
        return result

    def future_logits(*args, **kwargs):
        logits = _argument(original_future_logits, args, kwargs, "logits", 0)
        curtime = _argument(original_future_logits, args, kwargs, "curtime", 1)
        result = original_future_logits(*args, **kwargs)
        start = (
            vocab.TIME_OFFSET + max(int(curtime), 0) + max_gap + 1
        )
        _mask_range(logits, start, vocab.TIME_OFFSET + vocab.MAX_TIME)
        return result

    sample.safe_logits = safe_logits
    sample.future_logits = future_logits
    try:
        yield
    finally:
        sample.safe_logits = original_safe_logits
        sample.future_logits = original_future_logits
