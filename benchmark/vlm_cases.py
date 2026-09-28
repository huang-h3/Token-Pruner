"""VLM defaults used by the shell sweep."""

from dataclasses import dataclass

from token_pruner.selection import BASELINE_MODES


@dataclass(frozen=True)
class Mode:
    prune_mode: str
    i_mode: str
    p_mode: str

    @property
    def name(self):
        if self.p_mode in BASELINE_MODES + ("shared_hevc", "shared_folding", "global_folder"):
            mode = self.p_mode.removeprefix(f"{self.prune_mode}_")
            return f"{self.prune_mode}_{mode.replace('_', '-')}"
        return f"{self.prune_mode}_i-{self.i_mode}_p-{self.p_mode.replace('_', '-')}"


@dataclass(frozen=True)
class SweepModel:
    model_id: str
    default_pretrained: str
    default_num_frames: int
    default_prune_mode: str
    default_i_mode: str
    default_p_mode: str
    modes: tuple[Mode, ...]


def _baseline_modes(scopes):
    return tuple(Mode(scope, method, method) for scope in scopes for method in BASELINE_MODES)


_GROUPED = (
    Mode("global", "preserve", "hevc"),
    Mode("global", "preserve", "folder"),
    Mode("global", "folder", "hevc"),
    Mode("local", "preserve", "hevc"),
    Mode("local", "preserve", "folder"),
    Mode("local", "folder", "hevc"),
    Mode("local", "folder", "folder"),
)
_HF = (
    Mode("global", "global_folder", "global_folder"),
    Mode("shared", "shared_hevc", "shared_hevc"),
    Mode("shared", "shared_folding", "shared_folding"),
    *_GROUPED,
    *_baseline_modes(("global", "local")),
)
_OFFICIAL = (
    Mode("global", "global_folder", "global_folder"),
    Mode("shared", "shared_hevc", "shared_hevc"),
    Mode("shared", "shared_folding", "shared_folding"),
    Mode("local", "folder", "hevc"),
    Mode("local", "folder", "folder"),
    *_baseline_modes(("local",)),
)
_QWEN = (
    Mode("local", "preserve", "hevc"),
    Mode("global", "preserve", "hevc"),
    Mode("shared", "shared_hevc", "shared_hevc"),
    *_baseline_modes(("local", "global")),
)


MODEL_RUNS = {
    "internvl": SweepModel("internvl", "OpenGVLab/InternVL3_5-8B-Instruct", 8, "local", "preserve", "hevc", _HF),
    "llava_onevision2": SweepModel("llava_onevision2", "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct", 32, "local", "preserve", "hevc", _QWEN),
    "video_llava_hf": SweepModel(
        "video_llava_hf", "LanguageBind/Video-LLaVA-7B-hf", 8,
        "local", "folder", "hevc", _HF,
    ),
    "video_llava_official": SweepModel(
        "video_llava_official", "LanguageBind/Video-LLaVA-7B", 8,
        "local", "folder", "hevc", _OFFICIAL,
    ),
    "llava_next_video": SweepModel(
        "llava_next_video", "llava-hf/LLaVA-NeXT-Video-7B-hf", 8,
        "local", "preserve", "hevc", _HF,
    ),
    "qwen3_vl": SweepModel(
        "qwen3_vl", "Qwen/Qwen3-VL-8B-Instruct", 64,
        "local", "preserve", "hevc", _QWEN,
    ),
    "qwen3_vl_thinking": SweepModel(
        "qwen3_vl_thinking", "Qwen/Qwen3-VL-8B-Thinking", 64,
        "local", "preserve", "hevc", _QWEN,
    ),
}


__all__ = ("MODEL_RUNS", "Mode", "SweepModel", "legal_modes")


#: Legal (stage, scope, i_mode, p_mode) triples, read from the model modules.
LEGAL_MODE_SOURCES = {
    "internvl": ("run_internvl", "INTERNVL_LEGAL_MODES"),
    "ov2": ("run_llava_onevision2", "LLAVA_ONEVISION2_LEGAL_MODES"),
    "hf": ("run_videollava_hf", "VIDEO_LLAVA_HF_LEGAL_MODES"),
    "official": ("run_videollava_official", "VIDEO_LLAVA_OFFICIAL_LEGAL_MODES"),
    "lnv": ("run_llava_next_video", "LLAVA_NEXT_VIDEO_LEGAL_MODES"),
    "qwen3": ("run_qwen3_vl", "QWEN3_VL_LEGAL_MODES"),
}


def legal_modes(model: str) -> frozenset:
    import importlib

    module_name, attribute = LEGAL_MODE_SOURCES[model]
    module = importlib.import_module(f"token_pruner.{module_name}")
    return getattr(module, attribute)
