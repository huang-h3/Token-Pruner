"""Whole-model pruning: tower, prompt, and mrope must be cut together."""

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)

from token_pruner.run_qwen3_vl import (
    install_lm_shims, shrink_visual_placeholders,
)
from token_pruner.selection import LayerPruningPlan
from token_pruner.tokens import (
    PruningConfig, PruningScope, PruningSignal, SignalSource, TokenReducer,
)

T, H, W, BOUNDARY = 2, 4, 4, 2
CELLS = (H * W) // 4
VIDEO_TOKEN = 150


def _model():
    vision = Qwen3VLVisionConfig(
        hidden_size=32, intermediate_size=64, num_heads=2, depth=6,
        patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
        out_hidden_size=32, deepstack_visual_indexes=[3, 4],
        num_position_embeddings=64,
    )
    text = Qwen3VLTextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=200,
        max_position_embeddings=512,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1],
                      "mrope_interleaved": False},
    )
    config = Qwen3VLConfig(
        vision_config=vision.to_dict(), text_config=text.to_dict(),
        video_token_id=VIDEO_TOKEN, image_token_id=151, vision_start_token_id=152,
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval()


def _group_cells():
    """Group 0 keeps every cell; group 1 keeps its first two."""

    return [[torch.arange(CELLS), torch.arange(2)]]


#: Qwen3-VL writes one placeholder run per temporal position.
TIMESTAMP_TOKEN = 99


def _plan():
    """Group 0 preserved whole; group 1 keeps two of its four cells."""

    config = PruningConfig(
        scope=PruningScope.LOCAL, reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE, keep_per_partition=2,
        keep_total=CELLS + 2, signal_source=SignalSource.HEVC,
    )
    signal = PruningSignal(
        visible_indices=[torch.tensor([CELLS, CELLS + 1])],
        anchor_partitions=torch.tensor([[True, False]]),
    )
    return LayerPruningPlan(BOUNDARY, config, signal)


def _inputs():
    """One prompt whose placeholder runs are sized for the unpruned grid."""

    row = [10, 11]
    for group in range(T):
        row += [TIMESTAMP_TOKEN] + [VIDEO_TOKEN] * CELLS
    row += [12]
    ids = torch.tensor([row])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "mm_token_type_ids": (ids == VIDEO_TOKEN).long() * 2,
    }


def test_placeholders_shrink_to_the_surviving_cells():
    model = _model()
    inputs = _inputs()
    before = int((inputs["input_ids"] == VIDEO_TOKEN).sum())

    pruned, rope_state = shrink_visual_placeholders(
        model, inputs, _group_cells(), pad_token_id=0)
    after = int((pruned["input_ids"] == VIDEO_TOKEN).sum())

    assert before == T * CELLS
    assert after == CELLS + 2, "placeholders were not cut to the kept cells"
    assert torch.equal(
        pruned["mm_token_type_ids"], (pruned["input_ids"] == VIDEO_TOKEN).long() * 2
    )
    full_ids, full_mask, full_types, keep_masks = rope_state
    assert torch.equal(full_types, inputs["mm_token_type_ids"])
    assert torch.equal(full_ids, inputs["input_ids"]), (
        "the rope state must hold the pre-shrink ids: positions are derived on "
        "the full grid and only then restricted"
    )
    assert int(keep_masks[0].sum()) == pruned["input_ids"].shape[1]


def test_rope_shim_restricts_positions_to_survivors():
    model = _model()
    inputs = _inputs()
    state = {}
    restore = install_lm_shims(model, state)
    try:
        inner = model.model
        # Inert until armed: identical to the stock path.
        stock, stock_delta = inner.get_rope_index(
            inputs["input_ids"], inputs["mm_token_type_ids"],
            video_grid_thw=torch.tensor([[T, H, W]]),
            attention_mask=inputs["attention_mask"])
        pruned, rope_state = shrink_visual_placeholders(
            model, inputs, _group_cells(), pad_token_id=0)
        state["rope"] = rope_state
        positions, deltas = inner.get_rope_index(
            pruned["input_ids"], pruned["mm_token_type_ids"],
            video_grid_thw=torch.tensor([[T, H, W]]),
            attention_mask=pruned["attention_mask"])
    finally:
        restore()

    assert positions.shape[-1] == pruned["input_ids"].shape[1]
    assert stock.shape[-1] == inputs["input_ids"].shape[1]
    assert positions.shape[-1] < stock.shape[-1]
    # Every surviving position is one the full grid actually assigned.
    full = set(stock.reshape(3, -1)[0].tolist())
    assert set(positions.reshape(3, -1)[0].tolist()).issubset(full)
    assert deltas.shape == (1, 1)


def test_shims_restore_the_stock_entry_points():
    """Restoring must leave the object as it was, shadow included."""

    model = _model()
    inner = model.model
    original = inner.get_rope_index.__func__
    assert "get_rope_index" not in inner.__dict__, "fixture assumes a class method"

    restore = install_lm_shims(model, {})
    assert "get_rope_index" in inner.__dict__, "the shim did not take effect"
    restore()

    # Not merely callable again: no instance attribute left shadowing the class.
    assert "get_rope_index" not in inner.__dict__
    assert "get_video_features" not in inner.__dict__
    assert inner.get_rope_index.__func__ is original


def _generation_inputs():
    """Processor-shaped inputs: prompt, pixels, and the grid they describe."""

    vision = _model().config.vision_config
    width = vision.in_channels * vision.temporal_patch_size * vision.patch_size ** 2
    torch.manual_seed(2)
    inputs = _inputs()
    inputs["pixel_values_videos"] = torch.randn(T * H * W, width)
    inputs["video_grid_thw"] = torch.tensor([[T, H, W]])
    return inputs


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_pruned_generation_uses_cache_and_full_grid_positions(padding_side):
    from token_pruner.layer_boundary import EncoderPruningContext
    from token_pruner.run_qwen3_vl import install_qwen3_vl_encoder_pruning

    model = _model()
    inputs = _generation_inputs()
    pad = torch.zeros(1, 2, dtype=torch.long)
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        parts = (pad, inputs[key]) if padding_side == "left" else (inputs[key], pad)
        inputs[key] = torch.cat(parts, dim=1)
    pruned, rope = shrink_visual_placeholders(
        model, inputs, _group_cells(), pad_token_id=0, padding_side=padding_side
    )
    context = EncoderPruningContext(num_groups=T, plan=_plan())
    context.expected_group_cells = _group_cells()
    install_qwen3_vl_encoder_pruning(model.model.visual, context)
    restore = install_lm_shims(model, {"rope": rope})
    try:
        with torch.no_grad():
            output = model(**pruned, use_cache=True)
            generated = model.generate(
                **pruned, max_new_tokens=3, min_new_tokens=3,
                do_sample=False, pad_token_id=0, eos_token_id=None,
            )
        assert torch.isfinite(output.logits).all()
        assert output.past_key_values.get_seq_length() == pruned["input_ids"].shape[1]
        assert generated.shape[1] == pruned["input_ids"].shape[1] + 3
    finally:
        restore()


def test_full_budget_reproduces_visual_features_and_logits():
    from token_pruner.layer_boundary import EncoderPruningContext
    from token_pruner.run_qwen3_vl import install_qwen3_vl_encoder_pruning

    model = _model()
    inputs = _generation_inputs()
    with torch.no_grad():
        expected = model(**inputs).logits
        visual = model.model.visual(
            inputs["pixel_values_videos"], grid_thw=inputs["video_grid_thw"]
        )
    config = PruningConfig(
        scope=PruningScope.LOCAL, reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE, keep_per_partition=CELLS,
        keep_total=CELLS * T, signal_source=SignalSource.HEVC,
    )
    signal = PruningSignal(
        visible_indices=[torch.arange(CELLS * T)],
        anchor_partitions=torch.zeros(1, T, dtype=torch.bool),
    )
    context = EncoderPruningContext(
        num_groups=T, plan=LayerPruningPlan(BOUNDARY, config, signal)
    )
    install_qwen3_vl_encoder_pruning(model.model.visual, context)
    restore = install_lm_shims(model, {})
    try:
        with torch.no_grad():
            actual = model(**inputs).logits
            features = model.model.visual(
                inputs["pixel_values_videos"], grid_thw=inputs["video_grid_thw"]
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(features.pooler_output, visual.pooler_output, rtol=0, atol=0)
        for actual, expected in zip(features.deepstack_features, visual.deepstack_features):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        restore()
