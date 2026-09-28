"""Direct encoder forward that prunes once and runs ragged groups one at a time."""

import torch

from token_pruner.layer_boundary import pack_group_states, prune_frame_sequence


def clip_block(layer, state):
    return layer(state, None)


def videobind_block(layer, state):
    return layer(state, None, None)[0]


def reference_block_states(layers, hidden_states, num_groups, plan, block):
    """The packed input of every block, then the packed output of the last one."""

    states = []
    for index, layer in enumerate(layers):
        if index == plan.layer:
            hidden_states = prune_frame_sequence(hidden_states, num_groups, plan)
        states.append(pack_group_states(hidden_states, num_groups))
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = block(layer, hidden_states)
        else:
            hidden_states = [block(layer, state) for state in hidden_states]
    if plan.layer == len(layers):
        hidden_states = prune_frame_sequence(hidden_states, num_groups, plan)
    states.append(pack_group_states(hidden_states, num_groups))
    return states
