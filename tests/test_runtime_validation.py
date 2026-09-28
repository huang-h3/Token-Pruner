"""Runtime precision and pruning-boundary validation."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from token_pruner.task_io import inference_dtype


def test_default_cuda_dtype_remains_bfloat16_without_hardware_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (7, 5))
    assert inference_dtype('cuda:0') == torch.bfloat16
    assert inference_dtype('cpu') == torch.float32


@pytest.mark.parametrize('model_name', ['videollava_hf', 'videollava_official', 'llava_next_video', 'qwen3_vl'])
def test_pruning_after_consumed_features_is_rejected(model_name, tmp_path):
    import importlib
    module = importlib.import_module(f'token_pruner.run_{model_name}')
    names = {
        'videollava_hf': 'VideoLlavaHfPruning',
        'videollava_official': 'VideoLlavaOfficialPruning',
        'llava_next_video': 'LlavaNextVideoPruning',
        'qwen3_vl': 'Qwen3VlPruning',
    }
    tower = NS(config=NS(image_size=8, patch_size=2),
               vision_model=NS(encoder=NS(layers=[None] * 4)),
               encoder=NS(layers=[None] * 4),
               blocks=[None] * 4, deepstack_visual_indexes=[1, 2])
    backend = NS(device='cpu', video_tower=tower,
                 model=NS(config=NS(vision_feature_layer=-2),
                          model=NS(video_tower=tower, vision_tower=tower, visual=tower)),
                 token_geometry=lambda: (8, 2), partition_count=lambda frames: frames // 2,
                 install_pruning=Mock())
    if model_name == 'qwen3_vl':
        backend.stage_layer = lambda layer, **kwargs: module.Qwen3VlHfBackend.stage_layer(None, layer, **kwargs)
    with pytest.raises(ValueError, match='after the earliest consumed'):
        getattr(module, names[model_name]).create(
            backend, num_frames=4, prune_stage='last', prune_layer=4,
            prune_mode='local', i_mode='uniform', p_mode='uniform',
            k_keep_rate=0.5, run_mode='pruned', hevc_n_parallel=1, hevc_dir=tmp_path,
        )
    backend.install_pruning.assert_not_called()
