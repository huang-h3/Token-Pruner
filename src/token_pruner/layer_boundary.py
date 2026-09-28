"""Layer-boundary pruning hooks for VLM vision encoders."""

from dataclasses import dataclass, field

import torch

from .engine import prune_tokens
from .measurements import SpanAccumulator
from .selection import LayerPruningPlan
from .tokens import PruneTokens


@dataclass
class EncoderPruningContext:
    """Mutable plan consumed by an installed vision encoder forward."""

    enabled: bool = True
    num_groups: int = 8
    plan: LayerPruningPlan | None = None
    #: Cell budget the placeholders were resized to, and what the forward produced.
    expected_group_cells: list[int] | None = None
    kept_cells: list | None = None
    token_reduce_spans: SpanAccumulator = field(default_factory=SpanAccumulator)

    def reset_timing(self):
        self.token_reduce_spans.clear()

    def record_token_reduce(self, fn, hidden_states):
        tensor = (
            hidden_states
            if isinstance(hidden_states, torch.Tensor)
            else hidden_states[0]
        )
        return self.token_reduce_spans.record(fn, device=tensor.device)

    def token_reduce_ms(self):
        return self.token_reduce_spans.total_ms()


def prune_frame_sequence(hidden_states, num_groups, plan):
    """Prune a ``[B*G, 1+N, D]`` sequence; ragged groups return a list."""

    grouped = hidden_states.reshape(
        -1, int(num_groups), hidden_states.shape[-2], hidden_states.shape[-1]
    )
    cls_tokens, patch_tokens = grouped[:, :, :1], grouped[:, :, 1:]
    pruned = prune_tokens(
        PruneTokens.from_partitioned(patch_tokens, cls_tokens=cls_tokens),
        plan.config,
        plan.signal,
    )
    group_states = [
        torch.cat((cls_tokens[batch_idx, partition_idx], part), dim=0).unsqueeze(0)
        for batch_idx, partitions in enumerate(pruned.partitions)
        for partition_idx, part in enumerate(partitions)
    ]
    if len({state.shape[1] for state in group_states}) == 1:
        return torch.cat(group_states, dim=0)
    return group_states


def pack_group_states(hidden_states, num_groups):
    """Concatenate per-group states into ``[B, sum, D]``; rectangles pass through."""

    if isinstance(hidden_states, torch.Tensor):
        return hidden_states
    hidden_states = list(hidden_states)
    num_groups = int(num_groups)
    samples = [
        torch.cat(hidden_states[start : start + num_groups], dim=1)
        for start in range(0, len(hidden_states), num_groups)
    ]
    return torch.cat(samples, dim=0)


class _Carrier:
    __slots__ = ("groups", "active")

    def __init__(self):
        self.groups = None
        self.active = False

    def clear(self):
        self.groups = None
        self.active = False


def sequence_arg(args, kwargs):
    """Return the tower sequence argument."""

    if args:
        return args[0]
    name = "inputs_embeds" if "inputs_embeds" in kwargs else "hidden_states"
    return kwargs[name]


def replace_sequence(args, kwargs, hidden_states):
    """Replace the tower sequence argument."""

    if args:
        return (hidden_states, *args[1:]), kwargs
    updated = dict(kwargs)
    name = "inputs_embeds" if "inputs_embeds" in updated else "hidden_states"
    updated[name] = hidden_states
    return args, updated


def _run_groups(carrier, module, args, kwargs, first_output):
    carrier.active = True
    try:
        outputs = [first_output]
        for group in carrier.groups[1:]:
            group_args, group_kwargs = replace_sequence(args, kwargs, group)
            outputs.append(module.forward(*group_args, **group_kwargs))
    except Exception:
        carrier.clear()
        raise
    finally:
        carrier.active = False

    carrier.groups = [
        output[0] if isinstance(output, tuple) else output for output in outputs
    ]
    if not isinstance(first_output, tuple):
        return ()
    return tuple(
        [output[position] if isinstance(output, tuple) else None for output in outputs]
        for position in range(1, len(first_output))
    )


def install_group_carrier(
    blocks,
    encoder,
    context,
    num_groups,
    reduce,
    *,
    reset=None,
    output_module=None,
):
    """Install layer-boundary pruning and ragged-group forwarding hooks."""

    blocks = tuple(blocks)
    carrier = _Carrier()
    initial_hidden_state = None
    group_counts = []
    reset = reset or (lambda: None)

    def prune(hidden_states):
        plan = context.plan
        if not context.enabled or plan is None:
            return None
        reduced = reduce(hidden_states, plan)
        if isinstance(reduced, torch.Tensor):
            carrier.clear()
            return reduced

        groups = list(reduced)
        carrier.groups = groups
        return groups[0]

    def packed(hidden_states):
        return (
            hidden_states
            if carrier.groups is None
            else pack_group_states(carrier.groups, num_groups)
        )

    def block_pre(module, args, kwargs):
        if carrier.active or carrier.groups is None:
            return None
        return replace_sequence(args, kwargs, carrier.groups[0])

    def block_post(index):
        def hook(module, args, kwargs, output):
            if carrier.active:
                return None
            group_counts.append(len(carrier.groups) if carrier.groups is not None else 1)
            if carrier.groups is not None:
                extras = _run_groups(carrier, module, args, kwargs, output)
                result = packed(carrier.groups[0])
                return (result, *extras) if isinstance(output, tuple) else result

            plan = context.plan
            if plan is None or int(plan.layer) != index + 1:
                return None
            hidden_states = output[0] if isinstance(output, tuple) else output
            reduced = prune(hidden_states)
            if reduced is None:
                return None
            result = packed(reduced)
            return (result, *output[1:]) if isinstance(output, tuple) else result

        return hook

    def encoder_pre(module, args, kwargs):
        nonlocal initial_hidden_state
        initial_hidden_state = None
        group_counts.clear()
        carrier.clear()
        reset()
        plan = context.plan
        if not context.enabled or plan is None:
            return None
        layer = int(plan.layer)
        if layer:
            return None
        reduced = prune(sequence_arg(args, kwargs))
        initial_hidden_state = packed(reduced)
        return replace_sequence(args, kwargs, initial_hidden_state)

    def encoder_done(module, args, kwargs, output):
        carrier.clear()
        if output is None:
            reset()

    def capture_outputs(module, args, kwargs, output):
        nonlocal initial_hidden_state
        if output is None:
            initial_hidden_state = None
            group_counts.clear()
            return
        as_tuple = isinstance(output, tuple)
        if as_tuple:
            values = list(output)
            hidden_enabled = kwargs.get("output_hidden_states", module.config.output_hidden_states)
            attention_enabled = kwargs.get("output_attentions", module.config.output_attentions)
            hidden = values[2] if hidden_enabled else None
            attentions = values[2 + int(hidden_enabled)] if attention_enabled else None
        else:
            hidden, attentions = output.hidden_states, output.attentions
        if hidden is not None and initial_hidden_state is not None:
            hidden = (initial_hidden_state, *hidden[1:])
        if attentions is not None and any(count > 1 for count in group_counts):
            grouped, offset = [], 0
            for count in group_counts:
                grouped.append(attentions[offset] if count == 1 else tuple(attentions[offset:offset + count]))
                offset += count
            attentions = tuple(grouped)
        initial_hidden_state = None
        group_counts.clear()
        if as_tuple:
            if hidden_enabled:
                values[2] = hidden
            if attention_enabled:
                values[2 + int(hidden_enabled)] = attentions
            return tuple(values)
        if hidden is not None:
            output.hidden_states = hidden
        if attentions is not None:
            output.attentions = attentions

    handles = [
        encoder.register_forward_pre_hook(encoder_pre, with_kwargs=True),
        encoder.register_forward_hook(encoder_done, with_kwargs=True, always_call=True),
    ]
    if output_module is not None:
        handles.append(output_module.register_forward_hook(
            capture_outputs, with_kwargs=True, always_call=True
        ))
    for index, block in enumerate(blocks):
        handles.extend(
            (
                block.register_forward_pre_hook(block_pre, with_kwargs=True),
                block.register_forward_hook(block_post(index), with_kwargs=True, prepend=True),
            )
        )

    def cleanup():
        nonlocal initial_hidden_state
        initial_hidden_state = None
        group_counts.clear()
        for handle in handles:
            handle.remove()
        carrier.clear()
        reset()

    return cleanup
