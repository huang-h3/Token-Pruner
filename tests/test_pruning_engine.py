import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch

from benchmark.reporting.summary import (
    build_row,
    ordered_columns,
    task_suite_name,
)
from token_pruner import run_videollava_hf as videollava_hf
from token_pruner.hevc import (
    build_sampled_frames_ffmpeg_cmd,
)
from token_pruner.run_videollava_hf import VIDEO_LLAVA_HF_LEGAL_MODES
from token_pruner.run_videollava_official import VIDEO_LLAVA_OFFICIAL_LEGAL_MODES
from token_pruner.selection import LayerPruningPlan, resolve_pruning_layer
from token_pruner.tokens import (
    PruningConfig,
    PruningScope,
    PruningSignal,
    SharedFoldingOptions,
    PruneTokens,
)
from token_pruner.selection import (
    build_model_pruning_config as _build_model_pruning_config,
    build_shared_folding_config as _build_shared_folding_config,
    normalize_pruning_scope,
    resolve_pruning_budget,
    validate_pruning_modes,
)
from token_pruner.engine import prune_tokens
from token_pruner.selectors import (
    TubeletSelector,
    spatial_patch_scores,
    video_patch_selector,
)
from token_pruner.layer_boundary import (
    EncoderPruningContext,
)
from token_pruner.run_vivit import (
    forward_vivit_encoder,
)
from token_pruner.run_timesformer import (
    forward_timesformer_encoder,
)
from token_pruner.run_timesformer import (
    TIMESFORMER_LEGAL_MODES,
)
from token_pruner.run_vivit import (
    VIVIT_LEGAL_MODES,
)


_LEGAL_MODES = {
    "timesformer": TIMESFORMER_LEGAL_MODES,
    "vivit": VIVIT_LEGAL_MODES,
    "video_llava_hf": VIDEO_LLAVA_HF_LEGAL_MODES,
    "video_llava_official": VIDEO_LLAVA_OFFICIAL_LEGAL_MODES,
}


def validate_modes(model_family, stage, prune_mode, i_mode, p_mode):
    return validate_pruning_modes(
        _LEGAL_MODES[model_family],
        model_family,
        stage,
        prune_mode,
        i_mode,
        p_mode,
    )


def build_model_pruning_config(**kwargs):
    kwargs.setdefault("legal_modes", _LEGAL_MODES[kwargs["model_family"]])
    if kwargs["model_family"] == "timesformer":
        kwargs.setdefault("uniform_partitions", True)
    return _build_model_pruning_config(**kwargs)


def build_shared_folding_config(*, model_family="video_llava_hf", stage="input", **kwargs):
    kwargs.setdefault("legal_modes", _LEGAL_MODES[model_family])
    return _build_shared_folding_config(
        model_family=model_family,
        stage=stage,
        **kwargs,
    )


class FirstKFolder:
    def reduce(self, cls_token, tokens, keep):
        self.cls_shape = tuple(cls_token.shape)
        return tokens[:, : int(keep)]


class PruningConstraintTests(unittest.TestCase):
    def test_shared_scope_is_preserved_at_last_stage(self):
        self.assertEqual(
            normalize_pruning_scope("video_llava_hf", "input", "global"),
            PruningScope.GLOBAL,
        )
        self.assertEqual(
            normalize_pruning_scope("video_llava_hf", "last", "shared"),
            PruningScope.SHARED,
        )
        self.assertEqual(
            normalize_pruning_scope("video_llava_hf", "last", "global"),
            PruningScope.GLOBAL,
        )

    def test_stage_names_resolve_to_layer_boundaries(self):
        self.assertEqual(resolve_pruning_layer(None, stage="input", num_layers=12), 0)
        self.assertEqual(resolve_pruning_layer(None, stage="last", num_layers=12), 11)
        self.assertEqual(resolve_pruning_layer(5, stage="input", num_layers=12), 5)
        self.assertEqual(resolve_pruning_layer(-1, stage="input", num_layers=12), 11)
        self.assertEqual(resolve_pruning_layer(-2, stage="input", num_layers=12), 10)

    def test_model_constraints_reject_invalid_combinations(self):
        validate_modes("timesformer", "input", "local", "folder", "hevc")
        validate_modes("timesformer", "last", "local", "folder", "hevc")
        validate_modes("vivit", "last", "global", "preserve", "folder")
        validate_modes("video_llava_hf", "input", "shared", "shared_hevc", "shared_hevc")
        validate_modes("video_llava_hf", "input", "local", "preserve", "hevc")
        validate_modes(
            "video_llava_hf", "last", "shared", "shared_folding", "shared_folding"
        )
        invalid = (
            ("timesformer", "input", "global", "folder", "hevc"),
            ("video_llava_hf", "last", "global", "preserve", "shared_folding"),
            ("vivit", "input", "global", "folder", "folder"),
            ("video_llava_hf", "input", "shared", "folder", "folder"),
            ("video_llava_hf", "last", "global", "folder", "folder"),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                validate_modes(*values)

    def test_hevc_anchor_mode_matrix_matches_model_layout_constraints(self):
        # Ragged-capable models expose all four public combinations.
        for family in ("vivit", "video_llava_hf"):
            for scope in ("global", "local"):
                for i_mode in ("preserve", "folder"):
                    with self.subTest(family=family, scope=scope, i_mode=i_mode):
                        validate_modes(
                            family, "last", scope, i_mode, "hevc"
                        )

        # This tower needs equal-width frame partitions.
        validate_modes(
            "video_llava_official", "last", "local", "folder", "hevc"
        )
        for scope, i_mode in (("global", "folder"), ("local", "preserve")):
            with self.subTest(scope=scope, i_mode=i_mode), self.assertRaises(
                RuntimeError
            ):
                validate_modes(
                    "video_llava_official", "last", scope, i_mode, "hevc"
                )

    def test_config_rejects_incoherent_reducer_and_signal(self):
        with self.assertRaisesRegex(RuntimeError, "selection signal"):
            PruningConfig(
                scope="local",
                reducer="gather",
                keep_per_partition=2,
            )
        with self.assertRaisesRegex(RuntimeError, "does not consume"):
            PruningConfig(
                scope="local",
                reducer="folder",
                keep_per_partition=2,
                signal_source="hevc",
            )

    def test_dense_anchor_budget_overflow_is_rejected(self):
        config = PruningConfig(
            scope="local",
            reducer="gather",
            anchor_reducer="preserve",
            keep_per_partition=2,
            keep_total=3,
            signal_source="hevc",
        )
        with self.assertRaisesRegex(RuntimeError, "Dense GOP anchors exceed"):
            resolve_pruning_budget(
                config, partition_sizes={0: 4, 1: 4}, anchor_partitions=(0,)
            )

        config = PruningConfig(
            scope="local",
            reducer="gather",
            anchor_reducer="folder",
            keep_per_partition=2,
            keep_total=3,
            signal_source="hevc",
            min_per_partition=1,
        )
        budget = resolve_pruning_budget(
            config, partition_sizes={0: 4, 1: 4}, anchor_partitions=(0,)
        )
        self.assertEqual(budget.total_tokens, 3)

        config = PruningConfig(
            scope="local",
            reducer="gather",
            anchor_reducer="folder",
            keep_per_partition=2,
            keep_total=2,
            signal_source="hevc",
            min_per_partition=1,
        )
        with self.assertRaisesRegex(RuntimeError, "minimum_other_cost"):
            resolve_pruning_budget(
                config, partition_sizes={0: 4, 1: 4}, anchor_partitions=(0,)
            )

    def test_two_dense_gop_anchors_use_exact_budget_and_reject_overflow(self):
        config = PruningConfig(
            scope="local",
            reducer="gather",
            anchor_reducer="preserve",
            keep_per_partition=1,
            keep_total=10,
            signal_source="hevc",
            min_per_partition=1,
        )
        budget = resolve_pruning_budget(
            config,
            partition_sizes={0: 4, 1: 1, 2: 4, 3: 1},
            anchor_partitions=(0, 2),
        )
        self.assertEqual(budget.anchor_allocations, {0: 4, 2: 4})
        self.assertEqual(budget.other_allocations, {1: 1, 3: 1})
        self.assertEqual(budget.total_tokens, 10)

        config = PruningConfig(
            scope="local",
            reducer="gather",
            anchor_reducer="preserve",
            keep_per_partition=1,
            keep_total=7,
            signal_source="hevc",
            min_per_partition=1,
        )
        with self.assertRaisesRegex(RuntimeError, "Dense GOP anchors exceed"):
            resolve_pruning_budget(
                config,
                partition_sizes={0: 4, 1: 1, 2: 4, 3: 1},
                anchor_partitions=(0, 2),
            )


class PruningEngineTests(unittest.TestCase):
    def setUp(self):
        self.tokens = torch.arange(1 * 3 * 4 * 2).reshape(1, 3, 4, 2).float()
        self.global_cls = torch.zeros(1, 1, 2)
        self.group_cls = torch.zeros(1, 3, 1, 2)

    def test_local_hevc_gather_and_anchor_folder(self):
        config = build_model_pruning_config(
            model_family="timesformer",
            stage="input",
            prune_mode="local",
            i_mode="folder",
            p_mode="hevc",
            keep_per_partition=2,
            keep_total=6,
        )
        signal = PruningSignal(
            anchor_partitions=torch.tensor([[True, False, False]]),
            visible_indices=torch.tensor([[5, 7, 8, 10]]),
        )
        with mock.patch(
            "token_pruner.engine.folder_wrapper",
            return_value=FirstKFolder(),
        ):
            result = prune_tokens(
                PruneTokens.from_partitioned(self.tokens, cls_tokens=self.global_cls),
                config,
                signal,
            )
        self.assertEqual(result.partition_token_counts, [[2, 2, 2]])
        self.assertTrue(torch.equal(result.partitions[0][1], self.tokens[0, 1, [1, 3]]))

    def test_shared_gather_uses_one_spatial_map_for_every_group(self):
        config = build_model_pruning_config(
            model_family="video_llava_hf",
            stage="input",
            prune_mode="shared",
            i_mode="shared_hevc",
            p_mode="shared_hevc",
            keep_per_partition=2,
            keep_total=6,
        )
        result = prune_tokens(
            PruneTokens.from_partitioned(self.tokens, cls_tokens=self.group_cls),
            config,
            PruningSignal(visible_indices=torch.tensor([[1, 3]])),
        )
        self.assertEqual(result.partition_token_counts, [[2, 2, 2]])
        for group_idx in range(3):
            self.assertTrue(
                torch.equal(
                    result.partitions[0][group_idx],
                    self.tokens[0, group_idx, [1, 3]],
                )
            )

    def test_global_gather_preserves_anchor_and_enforces_total_budget(self):
        config = build_model_pruning_config(
            model_family="vivit",
            stage="last",
            prune_mode="global",
            i_mode="preserve",
            p_mode="hevc",
            keep_per_partition=2,
            keep_total=6,
        )
        signal = PruningSignal(
            anchor_partitions=torch.tensor([[True, False, False]]),
            visible_indices=torch.tensor([[5, 10]]),
        )
        result = prune_tokens(
            PruneTokens.from_partitioned(self.tokens, cls_tokens=self.global_cls),
            config,
            signal,
        )
        self.assertEqual(result.partition_token_counts, [[4, 1, 1]])
        self.assertEqual(tuple(result.pack_with_global_cls().shape), (1, 7, 2))

    def test_folder_reducer_is_reused_through_engine(self):
        config = build_model_pruning_config(
            model_family="vivit",
            stage="input",
            prune_mode="local",
            i_mode="folder",
            p_mode="folder",
            keep_per_partition=2,
            keep_total=6,
        )
        reducer = FirstKFolder()
        with mock.patch(
            "token_pruner.engine.folder_wrapper",
            return_value=reducer,
        ):
            result = prune_tokens(
                PruneTokens.from_partitioned(self.tokens, cls_tokens=self.group_cls),
                config,
            )
        self.assertEqual(result.partition_token_counts, [[2, 2, 2]])
        self.assertEqual(reducer.cls_shape, (1, 1, 2))

    def test_shared_folding_returns_the_configured_group_budget(self):
        options = SharedFoldingOptions(
            mode="uniform",
            patch_width=2,
            block_wise=False,
            global_k=2,
        )
        config = build_shared_folding_config(options=options, keep_total=6)
        result = prune_tokens(
            PruneTokens.from_partitioned(self.tokens, cls_tokens=self.global_cls),
            config,
        )
        self.assertEqual(result.partition_token_counts, [[2, 2, 2]])

    def test_flat_request_needs_no_temporal_partition(self):
        tokens = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
        request = PruneTokens(tokens)
        config = PruningConfig(
            scope="global",
            reducer="gather",
            anchor_reducer="preserve",
            keep_per_partition=3,
            keep_total=3,
            signal_source="uniform",
        )
        signal = PruningSignal(visible_indices=torch.tensor([[1, 4, 7]]))

        result = prune_tokens(request, config, signal)

        self.assertEqual(result.partition_token_counts, [[3]])
        self.assertEqual(result.samples[0].squeeze(-1).tolist(), [1.0, 4.0, 7.0])

    def test_ragged_flat_samples_use_only_optional_partition_metadata(self):
        samples = [
            torch.arange(7, dtype=torch.float32).unsqueeze(1),
            torch.arange(10, 15, dtype=torch.float32).unsqueeze(1),
        ]
        request = PruneTokens(
            samples,
            partition_ids=[
                torch.tensor([0, 0, 0, 1, 1, 1, 1]),
                torch.tensor([0, 0, 1, 1, 1]),
            ],
        )
        config = PruningConfig(
            scope="global",
            reducer="gather",
            anchor_reducer="preserve",
            keep_per_partition=2,
            keep_total=4,
            signal_source="hevc",
        )
        signal = PruningSignal(
            visible_indices=[torch.tensor([4]), torch.tensor([2, 4])],
            anchor_partitions=torch.tensor([[True, False], [True, False]]),
        )

        result = prune_tokens(request, config, signal)

        self.assertEqual(result.partition_token_counts, [[3, 1], [2, 2]])
        self.assertEqual(result.samples[0].squeeze(-1).tolist(), [0.0, 1.0, 2.0, 4.0])
        self.assertEqual(result.samples[1].squeeze(-1).tolist(), [10.0, 11.0, 12.0, 14.0])

    def test_hevc_anchor_modes_share_one_flat_interface_in_global_and_local(self):
        cases = {
            "preserve": (torch.tensor([[4, 5, 8]]), [4, 2, 1]),
            "folder": (torch.tensor([[4, 5, 6, 8, 9]]), [2, 3, 2]),
        }
        for scope in ("global", "local"):
            for i_mode, (visible, expected_counts) in cases.items():
                with self.subTest(scope=scope, i_mode=i_mode):
                    config = build_model_pruning_config(
                        model_family="vivit",
                        stage="last",
                        prune_mode=scope,
                        i_mode=i_mode,
                        p_mode="hevc",
                        keep_per_partition=2,
                        keep_total=7,
                    )
                    signal = PruningSignal(
                        visible_indices=visible,
                        anchor_partitions=torch.tensor([[True, False, False]]),
                    )
                    with mock.patch(
                        "token_pruner.engine.folder_wrapper",
                        return_value=FirstKFolder(),
                    ):
                        result = prune_tokens(
                            PruneTokens.from_partitioned(
                                self.tokens, cls_tokens=self.global_cls
                            ),
                            config,
                            signal,
                        )
                    self.assertEqual(result.partition_token_counts, [expected_counts])
                    self.assertEqual(result.samples[0].shape[0], 7)

    def test_global_tubelet_floor_covers_every_temporal_group(self):
        selector = TubeletSelector(
            patch_size=2,
            tubelet_size=2,
            tubelet_budget=1,
            image_shape=(1, 3, 6, 4, 4),
            prune_mode="global",
            min_per_group=1,
        )
        saliency = np.zeros((6, 4, 4), dtype=np.float32)
        saliency[:2] = 100.0
        selected = selector.tubelet_score_cal(
            spatial_patch_scores(saliency, 2),
            keep_tubelets=1,
            tubelet_size=2,
            prune_mode="global",
            anchor_frames=[0],
            total_keep_tubelets=3,
        )
        assert selected.numel() == 2
        assert {int(index) // 4 for index in selected} == {1, 2}

    def test_global_tubelet_hevc_excludes_anchor_from_p_budget(self):
        selector = TubeletSelector(
            patch_size=2,
            tubelet_size=2,
            tubelet_budget=1,
            image_shape=(1, 3, 6, 4, 4),
            prune_mode="global",
        )
        saliency = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
        cases = (
            # Folded I: one output slot, leaving four of target five for P.
            (False, 5, 4),
            # Preserved I: four source cells, leaving two of target six for P.
            (True, 6, 2),
        )
        for preserve, target, expected_p in cases:
            with self.subTest(preserve=preserve):
                selected = selector.tubelet_score_cal(
                    spatial_patch_scores(saliency, 2),
                    keep_tubelets=1,
                    tubelet_size=2,
                    prune_mode="global",
                    anchor_frames=[0],
                    preserve_anchor_tubelets=preserve,
                    total_keep_tubelets=target,
                )
                self.assertEqual(int(selected.numel()), expected_p)
                self.assertTrue(bool((selected >= 4).all()))


class EncoderBoundaryTests(unittest.TestCase):
    class TimesformerLayer:
        attention_type = "joint_space_time"

        def __init__(self):
            self.input_shapes = []

        def __call__(self, hidden_states, output_attentions=False):
            self.input_shapes.append(tuple(hidden_states.shape))
            return (hidden_states,)

    class VivitLayer:
        def __init__(self):
            self.input_shapes = []

        def __call__(self, hidden_states, head_mask=None):
            self.input_shapes.append(tuple(hidden_states.shape))
            return hidden_states

    def test_timesformer_prunes_only_at_requested_layer(self):
        layers = [self.TimesformerLayer(), self.TimesformerLayer()]
        model = SimpleNamespace(
            config=SimpleNamespace(image_size=4, patch_size=1),
            timesformer=SimpleNamespace(
                encoder=SimpleNamespace(layer=layers),
                layernorm=lambda value: value,
            ),
            classifier=lambda value: value,
        )
        config = build_model_pruning_config(
            model_family="timesformer",
            stage="last",
            prune_mode="local",
            i_mode="folder",
            p_mode="folder",
            keep_per_partition=2,
            keep_total=4,
        )
        hidden_states = torch.arange(18).reshape(1, 9, 2).float()
        with mock.patch(
            "token_pruner.engine.folder_wrapper",
            return_value=FirstKFolder(),
        ):
            forward_timesformer_encoder(
                model,
                hidden_states,
                num_frames=2,
                num_patches=4,
                pruning_plan=LayerPruningPlan(1, config),
            )
        self.assertEqual(layers[0].input_shapes, [(1, 9, 2)])
        self.assertEqual(layers[1].input_shapes, [(1, 5, 2)])

    def test_vivit_prunes_only_at_requested_layer(self):
        layers = [self.VivitLayer(), self.VivitLayer()]
        model = SimpleNamespace(
            config=SimpleNamespace(num_frames=4, tubelet_size=[2, 1, 1]),
            vivit=SimpleNamespace(
                layers=layers,
                layernorm=lambda value: value,
            ),
            classifier=lambda value: value,
        )
        config = build_model_pruning_config(
            model_family="vivit",
            stage="last",
            prune_mode="local",
            i_mode="folder",
            p_mode="folder",
            keep_per_partition=2,
            keep_total=4,
        )
        hidden_states = torch.arange(18).reshape(1, 9, 2).float()
        with mock.patch(
            "token_pruner.engine.folder_wrapper",
            return_value=FirstKFolder(),
        ):
            forward_vivit_encoder(
                model,
                hidden_states,
                pruning_plan=LayerPruningPlan(1, config),
            )
        self.assertEqual(layers[0].input_shapes, [(1, 9, 2)])
        self.assertEqual(layers[1].input_shapes, [(1, 5, 2)])


class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shapes = []

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=False,
    ):
        self.input_shapes.append(tuple(hidden_states.shape))
        outputs = (hidden_states,)
        return outputs + ((torch.zeros(1),) if output_attentions else ())


class DummyEncoder(torch.nn.Module):
    """Identity blocks behind the CLIP encoder loop the hooks attach to."""

    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList(DummyLayer() for _ in range(num_layers))

    def forward(self, inputs_embeds, output_hidden_states=False):
        states = ()
        hidden_states = inputs_embeds
        for layer in self.layers:
            if output_hidden_states:
                states += (hidden_states,)
            hidden_states = layer(hidden_states, None, None)[0]
        if output_hidden_states:
            states += (hidden_states,)
        return SimpleNamespace(last_hidden_state=hidden_states, hidden_states=states)


class DummyClipTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DummyEncoder()


class VideoLlavaHookTests(unittest.TestCase):
    def _run(self, plan, hidden, **kwargs):
        tower = DummyClipTower()
        cleanup = videollava_hf.install_pruning(
            tower, EncoderPruningContext(num_groups=3, plan=plan),
        )
        try:
            return tower, tower.encoder(hidden, **kwargs)
        finally:
            cleanup()

    def test_input_hook_uses_shared_spatial_map(self):
        hidden = torch.arange(3 * 5 * 2).reshape(3, 5, 2).float()
        config = build_model_pruning_config(
            model_family="video_llava_hf",
            stage="input",
            prune_mode="shared",
            i_mode="shared_hevc",
            p_mode="shared_hevc",
            keep_per_partition=2,
            keep_total=6,
        )
        plan = LayerPruningPlan(
            0, config, PruningSignal(visible_indices=torch.tensor([[1, 3]])),
        )
        tower, output = self._run(plan, hidden)
        self.assertEqual(tuple(output.last_hidden_state.shape), (3, 3, 2))
        self.assertTrue(torch.equal(output.last_hidden_state, hidden[:, [0, 2, 4]]))
        self.assertEqual(tower.encoder.layers[0].input_shapes, [(3, 3, 2)])

    def test_input_hook_runs_shared_folding(self):
        hidden = torch.arange(3 * 5 * 2).reshape(3, 5, 2).float()
        for pooling in ("weighted", "coverage-hard"):
            with self.subTest(pooling=pooling):
                options = SharedFoldingOptions(
                    mode="uniform",
                    patch_width=2,
                    block_size=2,
                    slots_per_block=2,
                    pooling=pooling,
                )
                config = build_shared_folding_config(
                    options=options,
                    keep_total=6,
                    model_family="video_llava_hf",
                    stage="input",
                )
                _, output = self._run(LayerPruningPlan(0, config), hidden)
                self.assertEqual(tuple(output.last_hidden_state.shape), (3, 3, 2))

    def test_last_hook_preserves_anchor_and_packs_per_frame_cls(self):
        hidden = torch.arange(3 * 5 * 2).reshape(3, 5, 2).float()
        config = build_model_pruning_config(
            model_family="video_llava_hf",
            stage="last",
            prune_mode="global",
            i_mode="preserve",
            p_mode="hevc",
            keep_per_partition=2,
            keep_total=6,
        )
        plan = LayerPruningPlan(
            1,
            config,
            PruningSignal(
                visible_indices=torch.tensor([[5, 10]]),
                anchor_partitions=torch.tensor([[True, False, False]]),
            ),
        )
        tower, output = self._run(plan, hidden, output_hidden_states=True)
        self.assertEqual(tuple(output.hidden_states[-2].shape), (1, 9, 2))
        self.assertEqual(tower.encoder.layers[0].input_shapes, [(3, 5, 2)])
        self.assertEqual(
            tower.encoder.layers[1].input_shapes,
            [(1, 5, 2), (1, 2, 2), (1, 2, 2)],
        )

    def test_last_hook_runs_shared_folding_without_position_output(self):
        hidden = torch.arange(3 * 5 * 2).reshape(3, 5, 2).float()
        options = SharedFoldingOptions(
            mode="uniform",
            patch_width=2,
            block_size=2,
            slots_per_block=2,
            pooling="weighted",
        )
        config = build_shared_folding_config(
            options=options,
            keep_total=6,
            model_family="video_llava_hf",
            stage="last",
        )
        _, output = self._run(
            LayerPruningPlan(1, config), hidden, output_hidden_states=True,
        )
        expected_content = hidden[:, 1:].reshape(3, 2, 2, 2).mean(dim=2)
        expected = torch.cat((hidden[:, :1], expected_content), dim=1)
        self.assertTrue(torch.equal(output.hidden_states[-2], expected))


class HevcEncodingTests(unittest.TestCase):
    def test_sampled_clip_is_encoded_as_one_gop(self):
        frames = np.zeros((8, 12, 16, 3), dtype=np.uint8)
        command = build_sampled_frames_ffmpeg_cmd(
            frames,
            "sampled.mp4",
            "libx265",
        )
        self.assertEqual(command[command.index("-g") + 1], "8")
        self.assertEqual(command[command.index("-s:v") + 1], "16x12")
        self.assertEqual(command[command.index("-i") + 1], "pipe:0")
        x265_params = command[command.index("-x265-params") + 1]
        self.assertIn("keyint=8:min-keyint=8:scenecut=0", x265_params)


class ResultSummaryTests(unittest.TestCase):
    def test_task_suite_name_keeps_subtask_names_verbatim(self):
        self.assertEqual(task_suite_name("nextqa_mc_test"), "nextqa_mc_test")
        self.assertEqual(
            task_suite_name("nextqa_mc_test+nextqa_oe_test+nextqa_oe_val"),
            "nextqa_mc_test+nextqa_oe_test+nextqa_oe_val",
        )
        self.assertEqual(
            task_suite_name(["nextqa_mc_test", "nextqa_oe_test"]),
            "nextqa_mc_test,nextqa_oe_test",
        )
        self.assertEqual(
            task_suite_name("mvbench_action_sequence"),
            "mvbench_action_sequence",
        )

    def test_lmms_row_keeps_metrics_metadata_timing_and_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stem = "nextqa_mvbench_global_preserve_hevc_k-0p25"
            result_path = root / f"{stem}.pt"
            responses_path = root / f"{stem}.responses.jsonl"
            batch = {"variant": "pruned", "measurement_scope": "batch", "sample_count": 1}
            torch.save(
                {
                    "timing": {},
                    "memory": {},
                    "measurements": {
                        "batches": [
                            {**batch, "signal_ms": 0.0, "model_ms": 100.0,
                             "peak_mem_mb": 1024.0},
                            {**batch, "signal_ms": 12.0, "model_ms": 200.0,
                             "peak_mem_mb": 1536.0},
                        ],
                        "requests": [],
                    },
                    "params": {
                        "model_type": "video_llava_hf",
                        "model_name": "Video-LLaVA-7B-hf",
                        "model_path": "LanguageBind/Video-LLaVA-7B-hf",
                        "task": (
                            "mvbench_action_sequence+nextqa_mc_test"
                            "+nextqa_oe_test+nextqa_oe_val"
                        ),
                        "run_mode": "pruned",
                        "prune_stage": "last",
                        "prune_mode": "global",
                        "i_mode": "preserve",
                        "p_mode": "hevc",
                        "k_keep_rate": 0.25,
                        "target_keep_total": 512,
                        "effective_keep_total": 512,
                        "num_frames": 8,
                        "num_samples": 1,
                        "answers_path": responses_path.name,
                    },
                },
                result_path,
            )
            artifact_dir = root / "lmms_artifacts" / stem / "run"
            artifact_dir.mkdir(parents=True)
            (artifact_dir.parent / ".complete").touch()
            (artifact_dir / "run_results.json").write_text(
                json.dumps(
                    {
                        "results": {
                            "nextqa_mc_test": {
                                "exact_match,none": 0.625,
                                "exact_match_stderr,none": 0.08,
                            },
                            "mvbench_action_sequence": {
                                "mvbench_accuracy,none": 75.0,
                            },
                            "vinoground": {
                                "vinoground_score,none": [62.5, 58.0, 41.25],
                            },
                        },
                        "config": {
                            "bootstrap_iters": 100000,
                            "resolved_cli_args": {"baseline": False},
                        },
                    }
                ),
                encoding="utf-8",
            )

            row = build_row(result_path, root)
            columns = ordered_columns([row])

        self.assertEqual(row["metric__nextqa_mc_test__exact_match"], 0.625)
        # stderr/stddev companions are disabled; they do not affect scores.
        self.assertNotIn(
            "metric__nextqa_mc_test__exact_match_stderr",
            row,
        )
        self.assertEqual(
            row["metric__mvbench_action_sequence__mvbench_accuracy"],
            75.0,
        )
        self.assertEqual(
            row["metric__vinoground__vinoground_score_text"],
            62.5,
        )
        self.assertEqual(
            row["metric__vinoground__vinoground_score_video"],
            58.0,
        )
        self.assertEqual(
            row["metric__vinoground__vinoground_score_group"],
            41.25,
        )
        self.assertEqual(
            row["task"],
            ("mvbench_action_sequence+nextqa_mc_test" "+nextqa_oe_test+nextqa_oe_val"),
        )
        self.assertEqual(row["model_path"], "LanguageBind/Video-LLaVA-7B-hf")
        self.assertEqual(row["timing__signal_ms__mean"], 6.0)
        self.assertEqual(row["timing__model_ms__mean"], 150.0)
        self.assertEqual(row["memory__peak_mem_mb__mean"], 1280.0)
        self.assertEqual(row["memory__peak_mem_mb__max"], 1536.0)
        self.assertEqual(row["source_path"], result_path.name)
        self.assertEqual(row["num_samples"], 1)
        self.assertNotIn("responses_path", row)
        self.assertNotIn("lmms_native_result_path", row)
        self.assertFalse(any(column.endswith("__count") for column in columns))
        self.assertFalse(any(column.endswith("__n") for column in columns))
        self.assertFalse(any(column.endswith("__p50") for column in columns))
        self.assertFalse(any(column.endswith("__p95") for column in columns))
        self.assertFalse(any(column.endswith("__total") for column in columns))
        self.assertFalse(any(column.startswith("param__") for column in columns))
        self.assertFalse(any(column.startswith("lmms_config__") for column in columns))
        self.assertIn("timing__signal_ms__mean", columns)
        self.assertIn("memory__peak_mem_mb__max", columns)
        self.assertNotIn("responses_sha256", columns)

        # Task suites are equally weighted; Vinoground uses its group score.
        self.assertAlmostEqual(
            row["avg_score"],
            (0.625 + 0.75 + 0.4125) / 3,
        )
        self.assertEqual(
            columns[:6],
            [
                "prune_mode", "i_mode", "p_mode", "prune_stage",
                "k_keep_rate", "run_mode",
            ],
        )

        # Configuration leads; results follow; provenance remains at the end.
        self.assertLess(columns.index("run_mode"), columns.index("avg_score"))
        self.assertLess(
            columns.index("timing__signal_ms__mean"),
            columns.index("metric__nextqa_mc_test__exact_match"),
        )
        self.assertLess(
            columns.index("memory__peak_mem_mb__max"),
            columns.index("metric__nextqa_mc_test__exact_match"),
        )
        self.assertEqual(
            columns[-4:],
            ["experiment", "result_type", "source_path", "model_type"],
        )


class BatchedHevcSelectorTests(unittest.TestCase):
    def test_ragged_visible_indices_remain_per_sample(self):
        selector = video_patch_selector(
            patch_size=16,
            keep_patches=2,
            total_keep_patches=6,
            num_frames=2,
            image_size=32,
            prune_mode="local",
            i_mode="preserve",
        )
        saliency = np.arange(2 * 32 * 32, dtype=np.float32).reshape(2, 32, 32)
        with mock.patch(
            "token_pruner.selectors._extract_hevc_saliency",
            side_effect=[
                (saliency, [0]),
                (saliency, [0, 1]),
            ],
        ):
            signal = selector.select(
                ["first.mp4", "second.mp4"],
                [[0, 1], [0, 1]],
            )

        self.assertEqual(
            signal.anchor_partitions.tolist(), [[True, False], [True, True]]
        )
        self.assertIsInstance(signal.visible_indices, list)
        self.assertEqual(
            [indices.numel() for indices in signal.visible_indices], [2, 0]
        )
