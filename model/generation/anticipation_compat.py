"""Narrow compatibility implementation of Anticipation's generation loop."""
from __future__ import annotations

import math


def generate_conditioned(
    model,
    start_time: float,
    end_time: float,
    inputs: list[int] | None = None,
    controls: list[int] | None = None,
    top_p: float = 1.0,
    debug: bool = False,
    delta: int | None = None,
) -> list[int]:
    """Generate while retaining time-zero and other nonnegative controls.

    The installed upstream ``generate`` clips controls with ``DELTA`` in
    seconds even though controls contain ticks. That excludes the time-zero
    note. This local copy keeps upstream's token order and anticipation loop,
    but does not apply that erroneous clip.
    """
    from anticipation import ops
    from anticipation.config import DELTA, TIME_RESOLUTION
    from anticipation.vocab import (
        ANTICIPATE,
        ATIME_OFFSET,
        AUTOREGRESS,
        CONTROL_OFFSET,
        DUR_OFFSET,
        NOTE_OFFSET,
        TIME_OFFSET,
    )
    from anticipation.sample import add_token

    inputs = [] if inputs is None else list(inputs)
    controls = [] if controls is None else list(controls)
    if len(inputs) % 3 or len(controls) % 3:
        raise ValueError("inputs and controls must contain complete three-token groups")
    if delta is None:
        delta = DELTA * TIME_RESOLUTION
    if delta < 0:
        raise ValueError("anticipation delta must be non-negative")

    start_tick = int(TIME_RESOLUTION * start_time)
    end_tick = int(TIME_RESOLUTION * end_time)
    if start_tick < 0 or end_tick < start_tick:
        raise ValueError("generation interval must satisfy 0 <= start <= end")

    prompt = ops.pad(
        ops.clip(inputs, 0, start_tick, clip_duration=False, seconds=False),
        start_tick,
    )
    future = ops.clip(
        inputs,
        start_tick + 1,
        ops.max_time(inputs, seconds=False),
        clip_duration=False,
        seconds=False,
    )
    control_groups = [
        tuple(controls[index:index + 3])
        for index in range(0, len(controls), 3)
    ]
    if any(group[2] < CONTROL_OFFSET for group in control_groups):
        raise ValueError("controls must use Anticipation's control-token vocabulary")
    sorted_controls = [
        token
        for group in sorted(control_groups, key=lambda group: group[0])
        for token in group
    ]
    mode = (
        [ANTICIPATE]
        if sorted_controls or future
        else [AUTOREGRESS]
    )
    tokens, remaining = ops.anticipate(
        prompt,
        ops.sort(sorted_controls + [CONTROL_OFFSET + token for token in future]),
        delta=delta,
    )
    current_time = ops.max_time(prompt, seconds=False)

    if remaining:
        atime, adur, anote = remaining[:3]
        anticipated_tokens = remaining[3:]
        anticipated_time = atime - ATIME_OFFSET
    else:
        anticipated_time = math.inf
        atime = adur = anote = 0
        anticipated_tokens = []

    while current_time < end_tick:
        while current_time >= anticipated_time - delta:
            tokens.extend((atime, adur, anote))
            if anticipated_tokens:
                atime, adur, anote = anticipated_tokens[:3]
                anticipated_tokens = anticipated_tokens[3:]
                anticipated_time = atime - ATIME_OFFSET
            else:
                anticipated_time = math.inf

        generated = add_token(
            model, mode, tokens, top_p, max(start_tick, current_time), debug
        )
        if len(generated) != 3:
            raise ValueError("Anticipation sampler must return one three-token event")
        generated_time = generated[0] - TIME_OFFSET
        if generated_time >= end_tick:
            break
        if generated_time < current_time:
            raise ValueError(
                f"sampler moved backwards in time: {generated_time} < {current_time}"
            )
        tokens.extend(generated)
        current_time = generated_time

    events, _controls = ops.split(tokens)
    return ops.sort(ops.unpad(events) + future)
