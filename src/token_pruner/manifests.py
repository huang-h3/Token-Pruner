"""lmms-eval entry point: model ids without importing the generation loop."""

MODEL_CLASSES = {
    "internvl": "token_pruner.run_internvl.InternVlRegisteredModel",
    "llava_onevision2": "token_pruner.run_llava_onevision2.LlavaOnevision2RegisteredModel",
    "video_llava_hf": "token_pruner.run_videollava_hf.VideoLlavaHfRegisteredModel",
    "video_llava_official": (
        "token_pruner.run_videollava_official.VideoLlavaOfficialRegisteredModel"
    ),
    "llava_next_video": (
        "token_pruner.run_llava_next_video.LlavaNextVideoRegisteredModel"
    ),
    "qwen3_vl": "token_pruner.run_qwen3_vl.Qwen3VlRegisteredModel",
    "qwen3_vl_thinking": (
        "token_pruner.run_qwen3_vl.Qwen3VlThinkingRegisteredModel"
    ),
}


def get_manifests():
    from lmms_eval.models.registry_v2 import ModelManifest

    return tuple(
        ModelManifest(model_id=model_id, chat_class_path=class_path)
        for model_id, class_path in MODEL_CLASSES.items()
    )
