from token_pruner.run_videollava_hf import VIDEO_LLAVA_HF_LEGAL_MODES
from token_pruner.run_videollava_official import VIDEO_LLAVA_OFFICIAL_LEGAL_MODES
import torch

from token_pruner.tokens import PruningScope, SignalSource, PruneTokens
from token_pruner.selection import build_model_pruning_config as _build_model_pruning_config
from token_pruner.engine import prune_tokens
from token_pruner.signals import BaselineTokenSelector
from token_pruner.run_vivit import VIVIT_LEGAL_MODES


_LEGAL_MODES = {
    "video_llava_hf": VIDEO_LLAVA_HF_LEGAL_MODES,
    "video_llava_official": VIDEO_LLAVA_OFFICIAL_LEGAL_MODES,
    "vivit": VIVIT_LEGAL_MODES,
}


def build_model_pruning_config(**kwargs):
    kwargs.setdefault("legal_modes", _LEGAL_MODES[kwargs["model_family"]])
    return _build_model_pruning_config(**kwargs)


def test_uniform_baseline_uses_interval_midpoints_and_exact_budget():
    selector = BaselineTokenSelector(
        "uniform",
        total_tokens=12,
        keep_total=5,
        num_partitions=3,
    )

    signal = selector.select(["sample-a", "sample-b"])

    expected = torch.tensor([1, 3, 6, 8, 10])
    assert signal.visible_indices.shape == (2, 5)
    assert torch.equal(signal.visible_indices[0], expected)
    assert torch.equal(signal.visible_indices[1], expected)
    assert not signal.anchor_partitions.any()


def test_random_baseline_is_stable_per_sample_across_batch_order():
    selector = BaselineTokenSelector(
        "random",
        total_tokens=64,
        keep_total=16,
        num_partitions=8,
        seed=17,
    )

    first = selector.select(["video-a", "video-b"]).visible_indices
    reordered = selector.select(["video-b", "video-a"]).visible_indices
    repeated = selector.select("video-a").visible_indices[0]

    assert torch.equal(first[0], reordered[1])
    assert torch.equal(first[1], reordered[0])
    assert torch.equal(first[0], repeated)
    assert not torch.equal(first[0], first[1])
    assert torch.equal(first, first.sort(dim=1).values)
    assert torch.unique(first[0]).numel() == 16


def test_global_baseline_floor_is_stable_and_preserves_divisible_uniform_budget():
    no_floor = BaselineTokenSelector(
        "uniform",
        total_tokens=12,
        keep_total=6,
        num_partitions=3,
    ).select(["clip"]).visible_indices
    with_floor = BaselineTokenSelector(
        "uniform",
        total_tokens=12,
        keep_total=6,
        num_partitions=3,
        min_per_partition=1,
    ).select(["clip"]).visible_indices
    assert torch.equal(with_floor, no_floor)

    first = BaselineTokenSelector(
        "random",
        total_tokens=12,
        keep_total=5,
        num_partitions=3,
        seed=17,
        min_per_partition=1,
    ).select(["clip"]).visible_indices[0]
    second = BaselineTokenSelector(
        "random",
        total_tokens=12,
        keep_total=5,
        num_partitions=3,
        seed=17,
        min_per_partition=1,
    ).select(["clip"]).visible_indices[0]
    assert torch.equal(first, second)
    assert torch.bincount(first // 4, minlength=3).min().item() >= 1


def test_random_baseline_changes_with_seed():
    first = BaselineTokenSelector(
        "random", 64, 16, 8, seed=1
    ).select("video").visible_indices
    second = BaselineTokenSelector(
        "random", 64, 16, 8, seed=2
    ).select("video").visible_indices

    assert not torch.equal(first, second)


def test_videollava_input_baselines_are_global_and_have_no_anchors():
    config = build_model_pruning_config(
        model_family="video_llava_hf",
        stage="input",
        prune_mode="global",
        i_mode="random",
        p_mode="random",
        keep_per_partition=2,
        keep_total=5,
    )

    assert config.scope == PruningScope.GLOBAL
    assert config.signal_source == SignalSource.RANDOM


def test_global_baseline_gather_keeps_exact_flattened_indices_per_sample():
    tokens = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4, 1)
    batch = PruneTokens.from_partitioned(tokens, cls_tokens=torch.zeros(2, 1, 1))
    config = build_model_pruning_config(
        model_family="vivit",
        stage="last",
        prune_mode="global",
        i_mode="uniform",
        p_mode="uniform",
        keep_per_partition=2,
        keep_total=5,
    )
    signal = BaselineTokenSelector(
        "uniform", 12, 5, 3
    ).select(["a", "b"])

    output = prune_tokens(batch, config, signal)

    for sample_index in range(2):
        selected = torch.cat(output.partitions[sample_index]).squeeze(-1)
        expected = tokens[sample_index].flatten()[signal.visible_indices[sample_index]]
        assert selected.numel() == 5
        assert torch.equal(selected, expected)


def test_local_baseline_selector_emits_flat_partition_indices():
    selector = BaselineTokenSelector(
        "uniform",
        total_tokens=12,
        keep_total=6,
        num_partitions=3,
        keep_per_partition=2,
    )

    signal = selector.select(["sample-a", "sample-b"])

    assert signal.visible_indices.shape == (2, 6)
    expected = torch.tensor([1, 3, 5, 7, 9, 11])
    assert torch.equal(signal.visible_indices, expected.expand(2, -1))
    assert not signal.anchor_partitions.any()


def test_local_random_baseline_varies_across_frames_but_stays_frame_local():
    selector = BaselineTokenSelector(
        "random",
        total_tokens=32,
        keep_total=8,
        num_partitions=4,
        keep_per_partition=2,
        seed=11,
    )

    signal = selector.select(["clip"])
    indices = signal.visible_indices

    assert indices.shape == (1, 8)
    local_rows = [
        indices[0, partition * 2 : (partition + 1) * 2] - partition * 8
        for partition in range(4)
    ]
    assert all(bool((row >= 0).all()) and bool((row < 8).all()) for row in local_rows)
    # Partitions use distinct random streams rather than one shared mask.
    rows = {tuple(row.tolist()) for row in local_rows}
    assert len(rows) > 1


def test_local_baseline_pruning_leaves_every_frame_the_same_width():
    """The native tower rebuilds one block per frame; ragged frames break it."""

    tokens = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4, 1)
    batch = PruneTokens.from_partitioned(tokens, cls_tokens=torch.zeros(2, 1, 1))
    config = build_model_pruning_config(
        model_family="video_llava_official",
        stage="input",
        prune_mode="local",
        i_mode="uniform",
        p_mode="uniform",
        keep_per_partition=2,
        keep_total=6,
    )
    signal = BaselineTokenSelector(
        "uniform", 12, 6, 3, keep_per_partition=2
    ).select(["a", "b"])

    output = prune_tokens(batch, config, signal)

    for sample_index in range(2):
        widths = {int(group.shape[0]) for group in output.partitions[sample_index]}
        assert widths == {2}, f"ragged frames: {widths}"
        for frame_index, group in enumerate(output.partitions[sample_index]):
            expected = tokens[sample_index, frame_index, [1, 3]]
            assert torch.equal(group, expected)


def test_global_baseline_pruning_leaves_frames_ragged():
    """Contrast case: this is why the native backend cannot use global scope."""

    tokens = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4, 1)
    batch = PruneTokens.from_partitioned(tokens, cls_tokens=torch.zeros(1, 1, 1))
    config = build_model_pruning_config(
        model_family="video_llava_hf",
        stage="last",
        prune_mode="global",
        i_mode="uniform",
        p_mode="uniform",
        keep_per_partition=2,
        keep_total=5,
    )
    # Midpoints [1, 3, 6, 8, 10] fall 2/1/2 across the three frames.
    signal = BaselineTokenSelector("uniform", 12, 5, 3).select(["a"])

    output = prune_tokens(batch, config, signal)

    widths = [int(group.shape[0]) for group in output.partitions[0]]
    assert widths == [2, 1, 2]
    assert len(set(widths)) > 1, "expected a ragged clip-wide draw"


def test_local_baseline_hits_an_arbitrary_exact_clip_budget():
    tokens = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4, 1)
    request = PruneTokens.from_partitioned(
        tokens, cls_tokens=torch.zeros(1, 1, 1)
    )
    config = build_model_pruning_config(
        model_family="video_llava_hf",
        stage="last",
        prune_mode="local",
        i_mode="uniform",
        p_mode="uniform",
        keep_per_partition=1,
        keep_total=5,
    )
    signal = BaselineTokenSelector(
        "uniform",
        total_tokens=12,
        keep_total=5,
        num_partitions=3,
        local=True,
    ).select(["clip"])

    output = prune_tokens(request, config, signal)

    assert output.partition_token_counts == [[2, 2, 1]]
    assert output.samples[0].shape[0] == 5
