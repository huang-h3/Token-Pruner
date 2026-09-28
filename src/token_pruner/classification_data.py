"""Vision-encoder configuration for classification runs."""

from dataclasses import dataclass

from .tokens import SharedFoldingOptions


@dataclass(frozen=True)
class VisionEncoderConfig:
    model_name: str | None = None
    run_mode: str = "pruned"
    num_frames: int | None = None
    k_keep_rate: float = 0.5
    k_keep: int | None = None
    patch_size: int | None = None
    tubelet_size: int | None = None
    prune_mode: str | None = None
    prune_stage: str = "input"
    prune_layer: int | None = None
    i_mode: str | None = None
    p_mode: str | None = None
    shared_folding: SharedFoldingOptions | None = None
    seed: int = 42
    hevc_gop_size: int = 0
    hevc_encode_scope: str = "sampled-clip"
    hevc_anchor_policy: str = "first"
