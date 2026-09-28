"""Model entry points and sweep configurations share the runtime contract."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from benchmark.vlm_cases import MODEL_RUNS
from token_pruner.manifests import MODEL_CLASSES, get_manifests


@pytest.mark.parametrize("model_id", ["internvl", "llava_onevision2"])
def test_registration_defaults_and_every_sweep_mode(model_id, tmp_path):
    module_path, class_name = MODEL_CLASSES[model_id].rsplit(".", 1)
    module = import_module(module_path)
    registered = getattr(module, class_name)
    profile = MODEL_RUNS[model_id]
    manifest = next(m for m in get_manifests() if m.model_id == model_id)
    assert manifest.chat_class_path == MODEL_CLASSES[model_id]
    assert registered.default_pretrained == profile.default_pretrained
    assert registered.default_num_frames == profile.default_num_frames
    backend_type = getattr(module, class_name.replace("RegisteredModel", "Backend"))
    pruning_type = getattr(module, class_name.replace("RegisteredModel", "Pruning"))
    backend = SimpleNamespace(
        device=torch.device("cpu"),
        vision_tower=lambda: None,
        encoder_layers=lambda: [None] * 4,
        feature_boundaries=lambda n: ([n], -1),
        token_geometry=lambda: (64, 16),
        install_pruning=lambda context: None,
    )
    if model_id == "llava_onevision2":
        backend.stage_layer = backend_type.stage_layer.__get__(backend)
    for stage in ("input", "last"):
        for mode in profile.modes:
            owner = registered(
                device="cpu",
                prune_stage=stage,
                prune_mode=mode.prune_mode,
                i_mode=mode.i_mode,
                p_mode=mode.p_mode,
            )
            assert owner.backend is None
            session = pruning_type.create(
                backend,
                num_frames=2,
                prune_stage=stage,
                prune_mode=mode.prune_mode,
                i_mode=mode.i_mode,
                p_mode=mode.p_mode,
                k_keep_rate=0.5,
                run_mode="pruned",
                hevc_n_parallel=1,
                hevc_dir=tmp_path,
            )
            assert session.pruning_active
            assert session.total_patches == 16
            assert session.prune_layer == (
                0 if stage == "input" else 4 if model_id == "llava_onevision2" else 3
            )


def test_onevision_rejects_folding_before_loading_weights():
    from token_pruner.run_llava_onevision2 import LlavaOnevision2RegisteredModel

    with pytest.raises(RuntimeError, match="Unsupported"):
        LlavaOnevision2RegisteredModel(
            prune_mode="local", i_mode="folder", p_mode="folder"
        )
