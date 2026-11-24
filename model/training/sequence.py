"""CPU-only AMT sequence construction with explicit event/control objectives."""
from __future__ import annotations

from dataclasses import dataclass


CACHE_SCHEMA = "anticipatory-sequence-v9"
SEQUENCE_POLICY = "anticipation-v1-context-bounded-generation-history"
INVALID_WINDOW_POLICY = "fail-on-any-window-error-empty-or-all-masked-v1"


@dataclass(frozen=True)
class SequenceSettings:
    control_offset: int
    rest_token: int
    time_offset: int
    duration_offset: int
    anticipate_time_offset: int
    anticipate_duration_offset: int
    time_resolution: int = 100
    anticipation_delta: int = 500
    padding_density: int = 100


class UnconsumedControlsError(ValueError):
    def __init__(self, count: int) -> None:
        self.count = count
        super().__init__(f"{count} melody control group(s) remained after REST padding")


class ContextOverflowError(ValueError):
    pass


def _groups(tokens: list[int]) -> list[tuple[int, int, int]]:
    if len(tokens) % 3:
        raise ValueError(f"expected complete three-token groups, got {len(tokens)} tokens")
    return [tuple(tokens[index:index + 3]) for index in range(0, len(tokens), 3)]


def pad_event_groups(
    events: list[int], end_time: int, settings: SequenceSettings
) -> list[int]:
    """Match upstream ops.pad: add REST groups at the configured density."""
    groups = sorted(_groups(events), key=lambda group: group[0])
    output: list[int] = []
    previous_time = settings.time_offset
    absolute_end = settings.time_offset + end_time
    for time, duration, note in groups:
        if note >= settings.control_offset:
            raise ValueError("ops.pad-compatible padding accepts events, not controls")
        while time > previous_time + settings.padding_density:
            previous_time += settings.padding_density
            output.extend((
                previous_time,
                settings.duration_offset,
                settings.rest_token,
            ))
        output.extend((time, duration, note))
        previous_time = time
    while absolute_end > previous_time + settings.padding_density:
        previous_time += settings.padding_density
        output.extend((
            previous_time,
            settings.duration_offset,
            settings.rest_token,
        ))
    return output


def interleave_anticipated_controls(
    events: list[int],
    controls: list[int],
    settings: SequenceSettings,
) -> tuple[list[int], list[int]]:
    """Interleave controls using upstream anticipation-v1 ordering."""
    event_groups = _groups(events)
    control_groups = sorted(_groups(controls), key=lambda group: group[0])
    output: list[int] = []
    remaining = list(control_groups)
    event_time = 0
    for group in event_groups:
        while remaining:
            control_time = remaining[0][0] - settings.anticipate_time_offset
            if event_time < control_time - settings.anticipation_delta:
                break
            output.extend(remaining.pop(0))
        if group[2] >= settings.control_offset:
            raise ValueError("anticipate event stream contains a control group")
        event_time = group[0] - settings.time_offset
        output.extend(group)
    return output, [token for group in remaining for token in group]


def _decode_group_time(
    group: tuple[int, int, int], settings: SequenceSettings
) -> int:
    return (
        group[0] - settings.anticipate_time_offset
        if group[2] >= settings.control_offset
        else group[0] - settings.time_offset
    )


def _translate_groups(
    groups: list[tuple[int, int, int]], settings: SequenceSettings
) -> list[tuple[int, int, int]]:
    if not groups:
        return []
    minimum = min(_decode_group_time(group, settings) for group in groups)
    translated = []
    for time, duration, note in groups:
        translated.append((time - minimum, duration, note))
    return translated


def chunk_sequence(
    tokens: list[int],
    mode_token: int,
    sequence_length: int,
    settings: SequenceSettings,
    context_groups: int = 170,
) -> list[tuple[list[int], list[int], list[int]]]:
    """Build aligned, overlapping-context examples and supervise each target once.

    Every chunk has up to 170 newly supervised event groups and up to 170
    preceding context groups (less if the sequence limit requires it). Controls
    in the anticipation window of each chunk's first target are retained even
    when that requires earlier context. If that context cannot fit, fail
    explicitly instead of silently training on an unconditioned target.
    """
    groups = _groups(tokens)
    if sequence_length < 4:
        raise ValueError("sequence_length must fit a mode token and one event group")
    max_groups = (sequence_length - 1) // 3
    history_capacity = min(context_groups, max_groups // 2)
    target_capacity = max(1, history_capacity)
    target_indices = [
        index for index, group in enumerate(groups)
        if group[2] < settings.control_offset
    ]
    if not target_indices:
        return []

    examples = []
    first_target_position = 0
    core_start = 0
    while first_target_position < len(target_indices):
        chosen_end = min(
            first_target_position + target_capacity, len(target_indices)
        )
        while True:
            last_target_position = chosen_end - 1
            first_target_index = target_indices[first_target_position]
            last_target_index = target_indices[last_target_position]
            core_end = (
                len(groups)
                if chosen_end == len(target_indices)
                else target_indices[chosen_end]
            )
            target_time = _decode_group_time(groups[first_target_index], settings)

            required_control_indices = [
                index
                for index in range(core_start)
                if groups[index][2] >= settings.control_offset
                and target_time - settings.anticipation_delta
                <= _decode_group_time(groups[index], settings)
                <= target_time + settings.anticipation_delta
            ]
            context_start = max(0, core_start - history_capacity)
            if required_control_indices:
                context_start = min(context_start, min(required_control_indices))
            context_length = core_start - context_start
            core_length = core_end - core_start
            if context_length + core_length <= max_groups:
                break
            if chosen_end - first_target_position <= 1:
                raise ContextOverflowError(
                    "one target group plus its required anticipation/history "
                    f"context needs {context_length + core_length} groups; "
                    f"sequence limit permits {max_groups}"
                )
            chosen_end -= 1

        selected = _translate_groups(groups[context_start:core_end], settings)
        ids = [mode_token] + [token for group in selected for token in group]
        labels = [-100] * len(ids)
        core_offset = 1 + context_length * 3
        core_token_end = core_offset + core_length * 3
        for token_index in range(core_offset, core_token_end):
            token = ids[token_index]
            if token < settings.control_offset:
                labels[token_index] = token
        attention = [1] * len(ids)
        padding = sequence_length - len(ids)
        ids.extend([0] * padding)
        labels.extend([-100] * padding)
        attention.extend([0] * padding)
        examples.append((ids, labels, attention))

        first_target_position = chosen_end
        core_start = core_end

    return examples


def example_counts(
    examples: list[tuple[list[int], list[int], list[int]]],
    settings: SequenceSettings,
) -> dict[str, int]:
    supervised_tokens = 0
    rest_groups = 0
    rest_tokens = 0
    all_masked = 0
    for ids, labels, attention in examples:
        count = sum(label != -100 and mask for label, mask in zip(labels, attention))
        if count == 0:
            all_masked += 1
        supervised_tokens += count
        for index in range(1, len(ids) - 2, 3):
            if (attention[index] and attention[index + 1] and attention[index + 2]
                    and ids[index + 2] == settings.rest_token
                    and labels[index + 2] != -100):
                rest_groups += 1
                rest_tokens += 3
    return {
        "produced_examples": len(examples),
        "supervised_tokens": supervised_tokens,
        "supervised_rest_groups": rest_groups,
        "supervised_rest_tokens": rest_tokens,
        "supervised_accompaniment_tokens": supervised_tokens - rest_tokens,
        "all_masked_examples": all_masked,
    }


def token_weighted_loss_weights(token_counts: list[int]) -> list[float]:
    """Weights for combining micro-batch mean losses into one token mean."""
    total = sum(token_counts)
    if not token_counts or total <= 0 or any(count < 0 for count in token_counts):
        raise ValueError("micro-batches must contain supervised tokens")
    return [count / total for count in token_counts]
