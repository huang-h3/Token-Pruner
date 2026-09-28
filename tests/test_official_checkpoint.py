import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from token_pruner import run_videollava_official as official


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs CUDA"
    )),
])
def test_checkpoint_weights_and_rotary_buffers_load_on_target_device(tmp_path, monkeypatch, device):
    config = LlamaConfig(
        vocab_size=32, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=1, num_key_value_heads=1,
    )
    language_model = LlamaForCausalLM(config)
    tower = torch.nn.Linear(4, 4)
    projector = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 8)
    )
    weights = dict(language_model.state_dict())
    weights.update({f"model.video_tower.video_tower.{key}": value
                    for key, value in tower.state_dict().items()})
    weights.update({f"model.mm_projector.{key}": value
                    for key, value in projector.state_dict().items()})
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    monkeypatch.setattr(official, "VideoBindVisionTransformer", lambda config: torch.nn.Linear(4, 4))
    monkeypatch.setattr(official, "nn", SimpleNamespace(
        Linear=lambda inputs, outputs: torch.nn.Linear(4 if inputs == 1024 else 8, 8),
        GELU=torch.nn.GELU,
        Sequential=torch.nn.Sequential,
    ))
    loaded_tower, loaded_projector, loaded = official.load_official_checkpoint(
        tmp_path, device=torch.device(device), dtype=torch.bfloat16,
        load_4bit=False, load_8bit=False,
    )
    for expected, actual in [(tower, loaded_tower), (projector, loaded_projector),
                             (language_model, loaded)]:
        for name, value in expected.state_dict().items():
            tensor = actual.state_dict()[name]
            assert tensor.device.type == device
            torch.testing.assert_close(tensor.cpu(), value.to(torch.bfloat16), rtol=0, atol=0)
    inverse = loaded.model.rotary_emb.inv_freq
    frequencies = torch.arange(0, 128, 2, dtype=torch.float32)
    expected_inverse = (1.0 / (10000.0 ** (frequencies / 128))).to(device)
    torch.testing.assert_close(inverse, expected_inverse, rtol=0, atol=0)
    assert inverse.dtype == torch.float32
    with torch.no_grad():
        result = loaded(torch.tensor([[1, 2, 3]], device=device))
    assert torch.isfinite(result.logits).all()
