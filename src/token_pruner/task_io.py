"""Shared timing, decoding, and media preparation for VLM backends."""

from dataclasses import dataclass
from pathlib import Path
import os
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .hevc_cache import HevcSelectionResult
from .measurements import SpanAccumulator
from .selection import resolve_pruning_budget
from .signals import signal_for_batch
from .tokens import PruningScope, SignalSource
from .video_io import sample_video_uniform

STOP_STRINGS_ENV = "TOKEN_PRUNER_STOP_STRINGS"
UNTRIMMED_SCORING_ENV = "TOKEN_PRUNER_UNTRIMMED_SCORING"
REASONING_SEGMENT_ENV = "TOKEN_PRUNER_REASONING_SEGMENT"
MAX_NEW_TOKENS_ENV = "TOKEN_PRUNER_MAX_NEW_TOKENS"
GREEDY_DECODING_ENV = "TOKEN_PRUNER_GREEDY_DECODING"
ANSWER_FORMAT_SUFFIX_ENV = "TOKEN_PRUNER_ANSWER_FORMAT_SUFFIX"
QWEN3_VL_DECODING_POLICY_ENV = "QWEN3_VL_DECODING_POLICY"
PROMPT_STYLE_ENV = "VIDEOLLAVA_OFFICIAL_PROMPT"
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


def inference_dtype(device):
    """BF16 on CUDA, FP32 otherwise."""

    return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32


def inference_dtype_name(device):
    """Serializable name for :func:`inference_dtype`."""

    return str(inference_dtype(device)).removeprefix("torch.")


def stop_strings_enabled():
    """Whether ``until`` halts generation."""

    return os.environ.get(STOP_STRINGS_ENV, "").strip().lower() not in _FALSE_VALUES


def untrimmed_scoring_enabled():
    """Whether the harness should score the generation before ``until`` trimming."""

    value = os.environ.get(UNTRIMMED_SCORING_ENV, "").strip().lower()
    return value not in _FALSE_VALUES


def reasoning_segment_mode():
    """Which segment of a reasoning generation the harness should score."""

    return os.environ.get(REASONING_SEGMENT_ENV, "").strip().lower() or "raw"


def split_reasoning(text, close_tag):
    """Split at the final reasoning close tag."""

    if not close_tag:
        return "", text
    index = text.rfind(close_tag)
    if index < 0:
        return text, ""
    return text[:index], text[index + len(close_tag) :].strip()


def max_new_tokens_override():
    """Explicit generation budget, or ``None`` to keep the task's own value."""

    raw = os.environ.get(MAX_NEW_TOKENS_ENV, "").strip()
    return int(raw) if raw else None


def greedy_decoding_enabled():
    """Whether decoding overrides the checkpoint's configuration with greedy."""

    value = os.environ.get(GREEDY_DECODING_ENV, "").strip().lower()
    return value not in _FALSE_VALUES


def qwen3_vl_decoding_policy():
    """Return the Qwen-specific decoding policy."""

    if greedy_decoding_enabled():
        return "greedy"
    return os.environ.get(QWEN3_VL_DECODING_POLICY_ENV, "").strip().lower() or "task"


def as_bool(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        raise TypeError(f"Cannot interpret {value!r} as a boolean.")
    return bool(value)


def exception_summary(exc, max_chars=180):
    text = str(exc).strip()
    return (text.splitlines()[0] if text else type(exc).__name__)[:max_chars]


def resolve_run_modes(run_mode):
    modes = {"pruned": (True, False), "full": (False, True), "both": (True, True)}
    try:
        return modes[str(run_mode)]
    except KeyError as exc:
        raise RuntimeError("run_mode must be pruned, full, or both") from exc


def resolve_pretrained_source(pretrained, local_files_only=False):
    source = str(pretrained)
    expanded = Path(source).expanduser()
    if expanded.exists():
        return str(expanded.resolve()), True
    return source, as_bool(local_files_only)


def public_model_source(pretrained):
    """A hub id as given; a local checkpoint by its directory name only."""

    source = str(pretrained)
    expanded = Path(source).expanduser()
    return expanded.name if expanded.exists() else source


@dataclass
class PreparedVideoSample:
    video_path: Path
    frame_indices: list[int]
    sampled_frames: object | None = None
    prompt: str = ""
    fps: float | None = None


class StageTimer:
    """Adapt :class:`SpanAccumulator` to Torch forward hooks."""

    def __init__(self, device):
        self.spans = SpanAccumulator(device)

    def clear(self):
        self.spans.clear()

    def pre_hook(self, _module, _args):
        self.spans.begin()

    def post_hook(self, _module, _args, _output):
        self.spans.end()

    def elapsed(self):
        return self.spans.elapsed()


class GenerationStepObserver(StoppingCriteria):
    """Timestamp completed generation steps without changing stop decisions."""

    def __init__(self, device):
        self.use_cuda = device.type == "cuda" and torch.cuda.is_available()
        self.start = None
        self.steps = []

    def begin(self):
        self.steps.clear()
        if self.use_cuda:
            self.start = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time.perf_counter()

    def __call__(self, input_ids, _scores, **_kwargs):
        if self.use_cuda:
            step = torch.cuda.Event(enable_timing=True)
            step.record()
        else:
            step = time.perf_counter()
        self.steps.append(step)
        return torch.zeros(
            input_ids.shape[0],
            dtype=torch.bool,
            device=input_ids.device,
        )

    def elapsed_steps(self):
        if self.start is None:
            return []
        if self.use_cuda:
            return [self.start.elapsed_time(step) for step in self.steps]
        return [(step - self.start) * 1000.0 for step in self.steps]


def normalize_until(value):
    if isinstance(value, str):
        return [value] if value else []
    return [item for item in (value or []) if item]


def trim_at_stop_sequences(text, stop_sequences):
    end = len(text)
    for stop in stop_sequences:
        position = text.find(stop)
        if position >= 0:
            end = min(end, position)
    return text[:end].strip()


STAGE_TIMER_NAMES = ("vision_encoder", "projector", "language_model")


def pad_row(row, width, value, side):
    padding = torch.full(
        (width - row.numel(),), value, dtype=row.dtype, device=row.device
    )
    return torch.cat((padding, row) if side == "left" else (row, padding))


def placeholder_runs(tokens, token_id):
    """Return half-open ranges of contiguous placeholder tokens."""

    matches = tokens == int(token_id)
    if not matches.any():
        return []
    boundaries = torch.where(matches[1:] != matches[:-1])[0] + 1
    edges = [0, *boundaries.tolist(), tokens.numel()]
    return [
        (start, end) for start, end in zip(edges, edges[1:]) if bool(matches[start])
    ]


def rewrite_placeholder_runs(
    inputs,
    token_id,
    run_lengths,
    *,
    pad_token_id,
    padding_side,
    allow_expand=True,
):
    """Resize placeholder runs and re-pad the batch."""

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    per_row = isinstance(run_lengths[0], (list, tuple))
    lengths_by_row = (
        run_lengths if per_row else [run_lengths for _ in range(input_ids.shape[0])]
    )

    rows = []
    for index, row in enumerate(input_ids):
        valid = (
            attention_mask[index].bool()
            if attention_mask is not None
            else torch.ones_like(row, dtype=torch.bool)
        )
        tokens = row[valid]
        runs = placeholder_runs(tokens, token_id)
        lengths = [int(length) for length in lengths_by_row[index]]
        assert len(runs) == len(lengths), "placeholder topology changed"

        parts = []
        previous = 0
        for (start, end), length in zip(runs, lengths):
            if not allow_expand:
                assert length <= end - start, "placeholder run is shorter than its cell budget"
            parts.append(tokens[previous:start])
            parts.append(tokens.new_full((length,), int(token_id)))
            previous = end
        parts.append(tokens[previous:])
        rows.append(torch.cat(parts))

    width = max(row.numel() for row in rows)
    inputs["input_ids"] = torch.stack(
        [pad_row(row, width, pad_token_id, padding_side) for row in rows]
    )
    if attention_mask is not None:
        masks = [
            torch.ones(row.numel(), dtype=attention_mask.dtype, device=row.device)
            for row in rows
        ]
        inputs["attention_mask"] = torch.stack(
            [pad_row(mask, width, 0, padding_side) for mask in masks]
        )
    return inputs


def collect_batch_timing(self, output, inputs):
    """Return batch-scoped spans and exact per-sequence token timings."""

    vision_times = self._stage_timers["vision_encoder"].elapsed()
    projector_times = self._stage_timers["projector"].elapsed()
    lm_times = self._stage_timers["language_model"].elapsed()
    step_times = self._step_observer.elapsed_steps()
    sequences = output.sequences if hasattr(output, "sequences") else output
    input_length = self._resolve_input_length(inputs)
    eos_token_ids = self._generation_config().eos_token_id
    eos_token_ids = (
        [eos_token_ids]
        if isinstance(eos_token_ids, int)
        else list(eos_token_ids or [])
    )
    request_timing = []
    for sequence in sequences:
        generated = self._slice_generated(sequence, input_length)
        output_tokens = int(generated.numel())
        stopped_on_eos = False
        if eos_token_ids and output_tokens:
            eos = torch.zeros_like(generated, dtype=torch.bool)
            for token_id in eos_token_ids:
                eos |= generated == int(token_id)
            positions = torch.where(eos)[0]
            if positions.numel():
                output_tokens = int(positions[0]) + 1
                stopped_on_eos = True
        observed_steps = min(output_tokens, len(step_times))
        request_timing.append(
            {
                "output_tokens": output_tokens,
                "hit_token_cap": not stopped_on_eos,
                "model_ttft_ms": step_times[0] if observed_steps else None,
                "tpot_ms": (
                    (step_times[observed_steps - 1] - step_times[0])
                    / (observed_steps - 1)
                    if observed_steps > 1
                    else None
                ),
            }
        )
    batch_timing = {
        "vision_encoder_ms": sum(vision_times),
        "projector_ms": sum(projector_times),
        "prefill_ms": lm_times[0] if lm_times else 0.0,
        "decode_ms": sum(lm_times[1:]) if len(lm_times) > 1 else 0.0,
        "decode_steps": max(0, len(lm_times) - 1),
    }
    return batch_timing, request_timing


def current_keep_budgets(self, batch_size):
    partitions = int(self.context.num_groups)
    if self.context.plan is None:
        total = self.total_patches * partitions
        self.realized_keep_per_partition_min.extend(
            [self.total_patches] * int(batch_size)
        )
        self.realized_keep_per_partition_max.extend(
            [self.total_patches] * int(batch_size)
        )
        return [(total, 0, total)] * int(batch_size)

    config = self.context.plan.config
    signal = self.context.plan.signal
    anchors = getattr(signal, "anchor_partitions", None) if signal is not None else None
    anchors = torch.as_tensor(anchors, dtype=torch.bool) if anchors is not None else None
    budgets = []
    for index in range(int(batch_size)):
        if config.signal_source in {SignalSource.RANDOM, SignalSource.UNIFORM}:
            anchor_indices = []
        elif config.scope == PruningScope.SHARED:
            anchor_indices = []
        elif anchors is None:
            anchor_indices = [0]
        else:
            row = anchors[0 if anchors.shape[0] == 1 else index]
            anchor_indices = row.nonzero().flatten().tolist()
        budget = resolve_pruning_budget(
            config,
            partition_sizes={i: self.total_patches for i in range(partitions)},
            anchor_partitions=anchor_indices,
        )
        allocations = {**budget.anchor_allocations, **budget.other_allocations}
        values = list(allocations.values())
        self.realized_keep_per_partition_min.append(min(values) if values else 0)
        self.realized_keep_per_partition_max.append(max(values) if values else 0)
        budgets.append((budget.total_tokens, len(budget.anchor_partitions), budget.other_tokens))
    return budgets


def decode_tokens(self, output, inputs, stop_sequences):
    output = output.sequences if hasattr(output, "sequences") else output
    input_length = self._resolve_input_length(inputs)
    if isinstance(stop_sequences, (str, bytes)):
        stop_sequences = [stop_sequences] * output.shape[0]
    untrimmed = [
        self._decode_ids(self._slice_generated(sequence, input_length)).strip()
        for sequence in output
    ]
    self._untrimmed_texts = untrimmed
    if self.reasoning_close_tag and reasoning_segment_mode() == "answer":
        untrimmed = [
            split_reasoning(text, self.reasoning_close_tag)[1]
            for text in untrimmed
        ]
    if untrimmed_scoring_enabled():
        return list(untrimmed)
    return [
        trim_at_stop_sequences(text, stops)
        for text, stops in zip(untrimmed, stop_sequences)
    ]


def prepare_batch(self, video_paths, frame_indices, sampled_frames):
    options = self._begin_signal()
    if self.signal_provider is None:
        return HevcSelectionResult()
    result = signal_for_batch(
        self.signal_provider,
        video_paths,
        frame_indices,
        sampled_frames,
        options=options,
        device=self.device,
        needs_hevc=self.needs_hevc,
    )
    self.context.plan = self.context.plan.bind(result.selector_outputs)
    effective = [
        value for value in result.hevc_effective_gop_sizes if value is not None
    ]
    self.hevc_effective_gop_sizes.extend(effective)
    unique = set(self.hevc_effective_gop_sizes)
    self.hevc_effective_gop_size = next(iter(unique)) if len(unique) == 1 else None
    self.hevc_artifact_keys.extend(str(key) for key in result.hevc_artifact_keys)
    return result


def request_video(media):
    """The single video of one request."""

    videos = list(media.get("videos") or [])
    if len(videos) != 1:
        raise RuntimeError(f"Expected one video per request, got {len(videos)}.")
    return Path(videos[0]).expanduser()


def prepare_sample(self, messages, media, num_frames):
    """Sample the frames of a request's video."""

    video_path = request_video(media)
    frames, frame_indices, fps = sample_video_uniform(
        video_path,
        num_frames,
        with_fps=True,
    )
    return PreparedVideoSample(
        video_path=video_path,
        frame_indices=frame_indices,
        sampled_frames=frames,
        prompt=self._prompt_from_messages(messages),
        fps=fps if self.needs_video_fps else None,
    )


def _begin_generation(self, generation_kwargs):
    """Reset instrumentation and attach the step observer."""
    for timer in self._stage_timers.values():
        timer.clear()
    self._step_observer.begin()
    generation_kwargs = dict(generation_kwargs)
    stopping = StoppingCriteriaList(
        generation_kwargs.pop("stopping_criteria", None) or []
    )
    stopping.append(self._step_observer)
    return generation_kwargs, stopping
