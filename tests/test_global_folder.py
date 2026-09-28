"""Global FOLDER merges all temporal groups once per sample."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from benchmark.vlm_cases import MODEL_RUNS
from token_pruner.engine import prune_tokens
from token_pruner.layer_boundary import prune_frame_sequence
from token_pruner.selection import LayerPruningPlan, resolve_pruning_budget
from token_pruner.task_io import current_keep_budgets
from token_pruner.tokens import PruneTokens, PruningConfig, PruningSignal


def config(**kwargs):
    return PruningConfig(
        scope="global",
        reducer="global_folder",
        keep_per_partition=2,
        keep_total=5,
        **kwargs,
    )


@pytest.fixture
def merges(monkeypatch):
    calls = []

    class Folder:
        def reduce(self, cls, tokens, keep):
            calls.append((cls.clone(), tokens.clone(), keep))
            return tokens.mean(dim=1, keepdim=True).expand(-1, keep, -1).clone()

    monkeypatch.setattr("token_pruner.engine.folder_wrapper", Folder)
    return calls


def test_one_merge_per_sample_includes_all_frames_and_ignores_anchors(merges):
    tokens = torch.arange(48, dtype=torch.float32).reshape(2, 3, 4, 2)
    cls = torch.arange(12, dtype=torch.float32).reshape(2, 3, 1, 2)
    signal = PruningSignal(anchor_partitions=torch.ones(2, 3, dtype=torch.bool))
    result = prune_tokens(
        PruneTokens.from_partitioned(tokens, cls_tokens=cls), config(), signal
    )
    assert len(merges) == 2
    for sample, (mean_cls, merged_input, keep) in enumerate(merges):
        assert torch.equal(merged_input, tokens[sample].reshape(1, 12, 2))
        assert torch.equal(mean_cls, cls[sample].mean(0, keepdim=True))
        assert keep == 5
        assert torch.equal(
            result.samples[sample], tokens[sample].mean((0, 1)).expand(5, 2)
        )
        assert [len(part) for part in result.partitions[sample]] == [2, 2, 1]
        assert (result.source_indices[sample] == -1).all()


def test_uniform_output_slots_still_use_one_global_merge(merges):
    plan = LayerPruningPlan(0, config(uniform_partitions=True))
    hidden = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2)
    result = prune_frame_sequence(hidden, 3, plan)
    assert len(merges) == 1
    assert torch.equal(merges[0][1], hidden[:, 1:].reshape(1, 12, 2))
    assert merges[0][2] == 6
    assert result.shape == (3, 3, 2)
    assert torch.equal(result[:, :1], hidden[:, :1])


def test_budget_and_report_have_zero_anchor_cost():
    cfg = config()
    budget = resolve_pruning_budget(
        cfg, partition_sizes={0: 4, 1: 4, 2: 4}, anchor_partitions=[0, 1, 2]
    )
    assert budget.anchor_partitions == ()
    assert budget.other_tokens == 5
    session = SimpleNamespace(
        context=SimpleNamespace(num_groups=3, plan=LayerPruningPlan(0, cfg)),
        total_patches=4,
        realized_keep_per_partition_min=[],
        realized_keep_per_partition_max=[],
    )
    assert current_keep_budgets(session, 2) == [(5, 0, 5)] * 2


@pytest.mark.parametrize(
    "family,module",
    [
        ("video_llava_hf", "run_videollava_hf"),
        ("video_llava_official", "run_videollava_official"),
        ("llava_next_video", "run_llava_next_video"),
        ("internvl", "run_internvl"),
        ("timesformer", "run_timesformer"),
        ("vivit", "run_vivit"),
    ],
)
def test_public_modes_match_the_supported_models(family, module):
    m = import_module("token_pruner." + module)
    legal = next(getattr(m, n) for n in vars(m) if n.endswith("_LEGAL_MODES"))
    for stage in ("input", "last"):
        assert (stage, "global", "global_folder", "global_folder") in legal
    if family in MODEL_RUNS:
        assert any(mode.name == "global_folder" for mode in MODEL_RUNS[family].modes)


@pytest.mark.parametrize("module", ["run_qwen3_vl", "run_llava_onevision2"])
def test_position_sensitive_models_do_not_advertise_merging(module):
    m = import_module("token_pruner." + module)
    legal = next(getattr(m, n) for n in vars(m) if n.endswith("_LEGAL_MODES"))
    assert all(mode[-1] != "global_folder" for mode in legal)


def test_classification_layouts_merge_all_temporal_groups_once(merges):
    from token_pruner.run_timesformer import prune_timesformer_hidden_states
    from token_pruner.run_vivit import prune_vivit_hidden_states

    frames = torch.arange(24, dtype=torch.float32).reshape(1, 3, 4, 2)
    cls = torch.zeros(1, 1, 2)
    plan = LayerPruningPlan(0, config(uniform_partitions=True))
    ts_input = torch.cat([cls, frames.permute(0, 2, 1, 3).reshape(1, 12, 2)], 1)
    ts_output, groups, patches = prune_timesformer_hidden_states(ts_input, 3, 4, plan)
    assert ts_output.shape == (1, 7, 2) and (groups, patches) == (3, 2)
    vi_input = torch.cat([cls, frames.reshape(1, 12, 2)], 1)
    model = SimpleNamespace(
        config=SimpleNamespace(num_frames=6, tubelet_size=[2, 1, 1])
    )
    vi_output = prune_vivit_hidden_states(
        model, vi_input, LayerPruningPlan(0, config())
    )
    assert vi_output.shape == (1, 6, 2)
    assert len(merges) == 2
    for _, merged_input, _ in merges:
        assert torch.equal(merged_input, frames.reshape(1, 12, 2))
