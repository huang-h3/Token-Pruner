"""Group-carrier pruning on CLIP-style vision towers."""

import pytest
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

from reference_forward import clip_block, reference_block_states, videobind_block
from token_pruner import run_videollava_hf as videollava_hf
from token_pruner import run_videollava_official as videollava_official
from token_pruner.layer_boundary import EncoderPruningContext
from token_pruner.selection import LayerPruningPlan
from token_pruner.tokens import (
    PruningConfig, PruningScope, PruningSignal, SignalSource, TokenReducer,
)

GROUPS, BOUNDARY = 4, 2          # four frames, prune before block 2
SIDE, PATCH = 32, 8
PATCHES = (SIDE // PATCH) ** 2   # 16 patches per frame


def _tower():
    torch.manual_seed(0)
    return CLIPVisionModel(CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=2, image_size=SIDE, patch_size=PATCH,
    )).eval()


def _encoder_model():
    """A deeper toy tower driven directly through its encoder."""

    torch.manual_seed(1)
    return CLIPVisionModel(CLIPVisionConfig(
        hidden_size=32, num_hidden_layers=6, num_attention_heads=4,
        intermediate_size=64, image_size=32, patch_size=8,
    )).eval()


def _pixels():
    torch.manual_seed(1)
    return torch.randn(GROUPS, 3, SIDE, SIDE)


def _embedded(tower, pixels):
    return tower.pre_layrnorm(tower.embeddings(pixels))


def _plan(keep=8):
    """Every frame keeps the same first ``keep`` patches: uniform by construction."""

    config = PruningConfig(
        scope=PruningScope.LOCAL, reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE, keep_per_partition=keep,
        keep_total=keep * GROUPS, signal_source=SignalSource.UNIFORM,
        uniform_partitions=True,
    )
    visible = torch.cat([
        torch.arange(keep) + group * PATCHES for group in range(GROUPS)
    ])
    signal = PruningSignal(
        visible_indices=[visible],
        anchor_partitions=torch.zeros(1, GROUPS, dtype=torch.bool),
    )
    return LayerPruningPlan(BOUNDARY, config, signal)


def _ragged_plan(layer=0, keep=5, groups=GROUPS):
    """Frame 0 is a preserved anchor; the others keep their first ``keep`` patches."""

    config = PruningConfig(
        scope=PruningScope.LOCAL, reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE, keep_per_partition=keep,
        keep_total=PATCHES + keep * (groups - 1),
        signal_source=SignalSource.HEVC,
    )
    visible = torch.cat([
        torch.arange(keep) + group * PATCHES
        for group in range(1, groups)
    ])
    signal = PruningSignal(
        visible_indices=[visible],
        anchor_partitions=torch.tensor([[group == 0 for group in range(groups)]]),
    )
    return LayerPruningPlan(layer, config, signal)


def _context(plan, groups=GROUPS):
    return EncoderPruningContext(enabled=True, num_groups=groups, plan=plan)


def _adapted(plan):
    tower = _tower()
    cleanup = videollava_hf.install_pruning(tower, _context(plan))
    try:
        with torch.no_grad():
            out = tower(_pixels(), output_hidden_states=True)
    finally:
        cleanup()
    return out.last_hidden_state, tower


@pytest.mark.parametrize("keep", [4, 8, 12])
def test_carrier_matches_the_reference_clip_forward(keep):
    reference = _tower()
    with torch.no_grad():
        expected = reference_block_states(
            reference.encoder.layers,
            _embedded(reference, _pixels()), GROUPS, _plan(keep), clip_block,
        )[-1]
    produced, _ = _adapted(_plan(keep))
    assert produced.shape == expected.shape, (
        f"{tuple(produced.shape)} != {tuple(expected.shape)}")
    assert torch.equal(produced, expected)


def test_removing_the_hook_restores_the_stock_tower():
    baseline_tower = _tower()
    with torch.no_grad():
        baseline = baseline_tower(_pixels()).last_hidden_state
    _, tower = _adapted(_plan(8))
    with torch.no_grad():
        after = tower(_pixels()).last_hidden_state
    assert torch.equal(after, baseline)


def test_a_global_split_that_empties_frames_is_carried():
    """Groups of different widths pack to ``groups + keep_total``."""

    config = PruningConfig(
        scope=PruningScope.GLOBAL, reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE, keep_per_partition=8,
        keep_total=20, signal_source=SignalSource.UNIFORM,
    )
    # 14 patches from frame 0, 6 from frame 1, none from the rest.
    visible = torch.cat([torch.arange(14), PATCHES + torch.arange(6)])
    plan = LayerPruningPlan(BOUNDARY, config, PruningSignal(
        visible_indices=[visible],
        anchor_partitions=torch.zeros(1, GROUPS, dtype=torch.bool),
    ))
    produced, _ = _adapted(plan)
    assert produced.shape[0] == 1, "ragged groups pack to one row per sample"
    assert produced.shape[1] == GROUPS + 20, (
        f"expected {GROUPS} class token(s) plus 20 patches, "
        f"got width {produced.shape[1]}")


def _videobind_tower():
    """The LanguageBind tower this project reimplements, at toy size."""

    from token_pruner.run_videollava_official import (
        VideoBindVisionConfig, VideoBindVisionTransformer,
    )

    torch.manual_seed(0)
    config = VideoBindVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=2, image_size=SIDE, patch_size=PATCH,
        num_frames=GROUPS,
    )
    return VideoBindVisionTransformer(config).eval()


def test_each_model_names_its_own_block_list():
    clip_blocks = videollava_hf.tower_blocks(_tower())
    assert len(clip_blocks) == 6, f"found {len(clip_blocks)} blocks"
    assert type(clip_blocks[0]).__name__ == "CLIPEncoderLayer"

    videobind_blocks = videollava_official.tower_blocks(_videobind_tower())
    assert len(videobind_blocks) == 6, f"found {len(videobind_blocks)} blocks"
    assert type(videobind_blocks[0]).__name__ == "VideoBindEncoderLayer"


def test_carrier_matches_the_reference_videobind_forward():
    """Divided space-time attention follows the same CLIP contract."""

    torch.manual_seed(2)
    pixels = torch.randn(1, 3, GROUPS, SIDE, SIDE)

    reference = _videobind_tower()
    with torch.no_grad():
        frames = pixels.permute(0, 2, 1, 3, 4).reshape(GROUPS, 3, SIDE, SIDE)
        embedded = reference.pre_layrnorm(
            reference.patch_dropout(reference.embeddings(frames), 1, GROUPS)
        )
        expected = reference_block_states(
            reference.encoder.layers, embedded, GROUPS, _plan(8), videobind_block,
        )[-1]

    tower = _videobind_tower()
    cleanup = videollava_official.install_pruning(tower, _context(_plan(8)))
    try:
        with torch.no_grad():
            produced = tower(pixels).last_hidden_state
    finally:
        cleanup()

    assert produced.shape == expected.shape
    assert torch.equal(produced, expected)


@pytest.mark.parametrize("layer", [0, 2])
@pytest.mark.parametrize("keep", [5, 7])
def test_ragged_hidden_states_match_the_reference_forward(layer, keep):
    """Ragged groups run one per block call; every recorded state matches."""

    bound = _ragged_plan(layer=layer, keep=keep)

    reference = _encoder_model()
    with torch.no_grad():
        expected = reference_block_states(
            reference.encoder.layers, _embedded(reference, _pixels()), GROUPS,
            bound, clip_block,
        )

    hooked = _encoder_model()
    cleanup = videollava_hf.install_pruning(hooked, _context(bound))
    try:
        with torch.no_grad():
            produced = hooked(_pixels(), output_hidden_states=True).hidden_states
    finally:
        cleanup()

    assert len(produced) == len(expected)
    for index, (want, got) in enumerate(zip(expected, produced)):
        assert want.shape == got.shape and torch.equal(want, got), (
            f"layer={layer} keep={keep}: hidden_states[{index}] diverged")


def test_an_anchor_that_takes_the_whole_budget_empties_the_other_frames():
    """An anchor may consume the whole budget; the partition count still travels."""

    groups = 8
    anchors = torch.tensor([[index == 0 for index in range(groups)]])
    torch.manual_seed(0)
    pixels = torch.randn(groups, 3, SIDE, SIDE)

    def plan(reducer, source, signal):
        return LayerPruningPlan(
            0,
            PruningConfig(
                scope=PruningScope.LOCAL, reducer=reducer,
                anchor_reducer=TokenReducer.PRESERVE, keep_per_partition=0,
                keep_total=PATCHES, signal_source=source),
            signal)

    def run(bound):
        model = _encoder_model()
        cleanup = videollava_hf.install_pruning(model, _context(bound, groups))
        try:
            with torch.no_grad():
                return model(pixels, output_hidden_states=True).hidden_states[-2]
        finally:
            cleanup()

    gather = plan(TokenReducer.GATHER, SignalSource.HEVC, PruningSignal(
        visible_indices=[torch.zeros(0, dtype=torch.long)],
        anchor_partitions=anchors))
    by_hook = run(gather)

    # One class token per frame plus the anchor's patches.
    assert by_hook.shape[1] == groups + PATCHES

    reference = _encoder_model()
    with torch.no_grad():
        expected = reference_block_states(
            reference.encoder.layers, _embedded(reference, pixels), groups, gather,
            clip_block,
        )[-2]
    assert torch.equal(by_hook, expected)

    folded = run(plan(
        TokenReducer.FOLDER, SignalSource.NONE,
        PruningSignal(visible_indices=None, anchor_partitions=anchors)))
    assert torch.equal(by_hook, folded), (
        "with nothing left to reduce, the p_modes must coincide")


def test_ragged_groups_stay_ragged_through_every_block():
    """``hidden_states`` records the packed sequence; blocks still see one frame."""

    model = _encoder_model()
    cleanup = videollava_hf.install_pruning(model, _context(_ragged_plan(layer=0)))
    widths = []
    for block in model.encoder.layers:
        def record(hidden_states, *args, forward=block.forward, **kwargs):
            widths.append(tuple(hidden_states.shape))
            return forward(hidden_states, *args, **kwargs)

        block.forward = record
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            model.encoder(torch.randn(GROUPS, PATCHES + 1, 32))
    finally:
        cleanup()

    # Every block sees one frame at a time.
    seen = {shape[1] for shape in widths}
    assert seen == {PATCHES + 1, 6}, (
        f"a block was handed a packed sequence: token counts {sorted(seen)}")


def test_boundary_zero_works_through_the_real_keyword_call():
    """CLIP supplies inputs_embeds to its encoder as a keyword argument."""

    bound = _ragged_plan(layer=0)
    hooked = _tower()
    hooked.config._attn_implementation = "eager"
    cleanup = videollava_hf.install_pruning(hooked, _context(bound))
    try:
        with torch.no_grad():
            output = hooked(
                _pixels(), output_hidden_states=True, output_attentions=True)
    finally:
        cleanup()

    reference = _tower()
    reference.config._attn_implementation = "eager"
    with torch.no_grad():
        expected = reference_block_states(
            reference.encoder.layers,
            _embedded(reference, _pixels()), GROUPS, bound, clip_block,
        )

    assert len(output.hidden_states) == len(expected)
    for want, got in zip(expected, output.hidden_states):
        assert want.shape == got.shape and torch.equal(want, got)
    assert len(output.attentions) == len(reference.encoder.layers)
    for layer_attentions in output.attentions:
        assert len(layer_attentions) == GROUPS


def test_failed_ragged_forward_does_not_leak_into_the_next_call():
    """Forward-scoped carrier state must be discarded on every exception."""

    model = _tower()
    context = _context(_ragged_plan(layer=0))
    cleanup = videollava_hf.install_pruning(model, context)
    calls = 0

    layer = model.encoder.layers[0]
    forward = layer.forward

    def fail_on_second_group(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected group failure")
        return forward(*args, **kwargs)

    layer.forward = fail_on_second_group
    try:
        try:
            with pytest.raises(RuntimeError, match="injected group failure"):
                with torch.no_grad():
                    model(_pixels(), output_hidden_states=True)
        finally:
            del layer.forward

        context.enabled = False
        baseline = _tower()
        with torch.no_grad():
            actual = model(_pixels()).last_hidden_state
            expected = baseline(_pixels()).last_hidden_state
    finally:
        cleanup()

    assert calls == 2
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("return_dict", [True, False])
def test_recording_after_an_unpruned_forward_and_cleanup(return_dict):
    model = _tower()
    pixels = _pixels()
    with torch.no_grad():
        baseline = model(pixels, output_hidden_states=True)
    bound = _ragged_plan(layer=0)
    cleanup = videollava_hf.install_pruning(model, _context(bound))
    try:
        with torch.no_grad():
            result = model(pixels, output_hidden_states=True, return_dict=return_dict)
            expected = reference_block_states(
                _tower().encoder.layers, _embedded(model, pixels), GROUPS,
                bound, clip_block,
            )
        states = result.hidden_states if return_dict else result[2]
        assert len(states) == len(expected)
        for actual, wanted in zip(states, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    finally:
        cleanup()
    with torch.no_grad():
        restored = model(pixels, output_hidden_states=True)
    for actual, wanted in zip(restored.hidden_states, baseline.hidden_states):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
