"""ViViT classification backend with tubelet pruning."""

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from transformers import (
    VivitConfig,
    VivitForVideoClassification,
    VivitImageProcessor,
)

from .selectors import BaselineTokenSelector, TubeletSelector
from .selection import (
    BASELINE_MODES,
    GROUPED_MODE_COMBINATIONS,
    LayerPruningPlan,
    build_model_pruning_config,
    resolve_pruning_layer,
    validate_pruning_modes,
)
from .tokens import SignalSource, PruneTokens
from .engine import prune_tokens
from .task_io import resolve_run_modes
from .hevc import HEVC_CODEC_LAYOUT, HEVC_SIGNAL_POLICY, resolve_hevc_encode_scope


VIVIT_LEGAL_MODES = frozenset({
    *((stage, "global", "global_folder", "global_folder") for stage in ("input", "last")),
    *((stage, *combination) for stage in ("input", "last")
      for combination in GROUPED_MODE_COMBINATIONS),
    *((stage, "global", p, p) for stage in ("input", "last")
      for p in BASELINE_MODES),
})

VIVIT_MODEL_NAME = "google/vivit-b-16x2-kinetics400"


def resolve_vivit_keep_budget(config, total_tubelets, num_frames, tubelet_t):
    k_keep_rate = config.k_keep_rate
    k_keep_num = max(1, int(total_tubelets * k_keep_rate))
    total_num_per_frame = total_tubelets // (num_frames // tubelet_t)
    k_keep_num_per_frame = max(1, int(total_num_per_frame * k_keep_rate))
    return k_keep_num, k_keep_num_per_frame, total_tubelets, total_num_per_frame


def load_vivit(config, device):
    """Load ViViT and return the typed classification session."""

    device = torch.device(device)
    model_name = config.model_name or VIVIT_MODEL_NAME
    prune_mode = config.prune_mode or "global"
    i_mode = config.i_mode or "preserve"
    p_mode = config.p_mode or "hevc"
    if p_mode in BASELINE_MODES:
        prune_mode = "global"
        i_mode = p_mode
    run_pruned, run_full = resolve_run_modes(config.run_mode)
    if run_pruned:
        validate_pruning_modes(
            VIVIT_LEGAL_MODES,
            "vivit",
            config.prune_stage,
            prune_mode,
            i_mode,
            p_mode,
        )

    model_config = VivitConfig.from_pretrained(model_name)
    model_num_frames = int(model_config.num_frames)
    num_frames = (
        int(config.num_frames)
        if config.num_frames is not None
        else model_num_frames
    )
    image_size = int(model_config.image_size)
    tubelet_size = list(model_config.tubelet_size)
    patch_size = int(tubelet_size[1])
    tubelet_t = (
        int(config.tubelet_size)
        if config.tubelet_size is not None
        else int(tubelet_size[0])
    )
    requested_gop_size = int(config.hevc_gop_size)
    encode_scope = resolve_hevc_encode_scope(config.hevc_encode_scope)
    hevc_active = bool(run_pruned and p_mode == "hevc")
    # The effective GOP resolves only once a full source video is encoded.
    effective_gop_size = None
    anchor_policy = str(config.hevc_anchor_policy) if hevc_active else "none"
    tubelet_size[0] = tubelet_t
    model_config.tubelet_size = tubelet_size
    model_config.num_frames = num_frames

    image_processor = VivitImageProcessor.from_pretrained(model_name)
    model = VivitForVideoClassification.from_pretrained(
        model_name,
        config=model_config,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()

    total_tubelets = (
        num_frames // tubelet_t
    ) * (image_size // patch_size) ** 2
    (
        keep_total,
        keep_per_partition,
        total_num,
        total_per_group,
    ) = resolve_vivit_keep_budget(
        config,
        total_tubelets,
        num_frames,
        tubelet_t,
    )
    pruning_config = (
        build_model_pruning_config(
            model_family="vivit",
            stage=config.prune_stage,
            prune_mode=prune_mode,
            i_mode=i_mode,
            p_mode=p_mode,
            legal_modes=VIVIT_LEGAL_MODES,
            keep_per_partition=keep_per_partition,
            keep_total=keep_total,
        )
        if run_pruned
        else None
    )
    prune_layer = resolve_pruning_layer(
        config.prune_layer,
        stage=config.prune_stage,
        num_layers=len(model.vivit.layers),
    )
    pruning_plan = (
        LayerPruningPlan(prune_layer, pruning_config)
        if pruning_config is not None
        else None
    )
    selector = None
    if (
        pruning_config is not None
        and pruning_config.signal_source == SignalSource.HEVC
    ):
        selector = TubeletSelector(
            patch_size=patch_size,
            tubelet_size=tubelet_t,
            tubelet_budget=keep_per_partition,
            image_shape=(1, 3, num_frames, image_size, image_size),
            prune_mode=prune_mode,
            preserve_anchor_tubelets=i_mode == "preserve",
            total_tubelet_budget=keep_total,
            hevc_gop_size=requested_gop_size,
            hevc_anchor_policy=anchor_policy,
        )
    elif pruning_config is not None and p_mode in BASELINE_MODES:
        selector = BaselineTokenSelector(
            p_mode,
            total_tokens=total_tubelets,
            keep_total=keep_total,
            num_partitions=num_frames // tubelet_t,
            seed=config.seed,
        )

    if not run_pruned:
        info = f"Model = {model_name}, Full | tubelet size={tubelet_t}"
        short_info = f"ViViT full, tubelet size={tubelet_t}"
    elif prune_mode == "global":
        info = (
            f"Model = {model_name}, target K={keep_total}/{total_num} "
            f"tubelets I:{i_mode}, P:{p_mode} | tubelet size={tubelet_t}"
        )
        short_info = (
            f"ViViT global, target K={keep_total}/{total_num} "
            f"tubelets I:{i_mode}, P:{p_mode}"
        )
    else:
        budget_info = (
            f"target global {keep_total}/{total_num}, "
            f"local {keep_per_partition}/{total_per_group} per frame"
        )
        info = (
            f"Model = {model_name}, {budget_info} tubelets. | "
            f"I:{i_mode}, P:{p_mode} | tubelet size={tubelet_t}"
        )
        short_info = (
            f"ViViT local, {budget_info} I:{i_mode}, P:{p_mode}"
        )

    return VivitRun(
        model_type="vivit",
        model_name=model_name,
        model=model,
        model_config=model_config,
        image_processor=image_processor,
        device=device,
        num_frames=num_frames,
        image_size=image_size,
        patch_size=patch_size,
        tubelet_size=tubelet_t,
        run_pruned=run_pruned,
        run_full=run_full,
        pruning_plan=pruning_plan,
        selector=selector,
        process_frames_fn=process_vivit_frames,
        prepare_pruned_fn=prepare_vivit_state_pruned,
        forward_pruned_fn=forward_vivit_state_prepared,
        info=info,
        short_info=short_info,
        accuracy_key=p_mode,
        total_tokens=total_tubelets,
        total_tokens_per_step=total_per_group,
        keep_per_frame=keep_per_partition,
        hevc_gop_size=requested_gop_size,
        hevc_encode_scope=encode_scope,
        hevc_effective_gop_size=effective_gop_size,
        hevc_anchor_policy=anchor_policy,
        params={
            "k_keep": keep_per_partition if run_pruned else None,
            "k_keep_rate": config.k_keep_rate,
            "total_tubelets": total_tubelets,
            "num_frames": num_frames,
            "tubelet_size": tubelet_t,
            "patch_size": patch_size,
            "i_mode": i_mode if run_pruned else None,
            "p_mode": p_mode if run_pruned else None,
            "prune_mode": prune_mode if run_pruned else None,
            "prune_stage": config.prune_stage if run_pruned else None,
            "prune_layer": prune_layer if run_pruned else None,
            "hevc_gop_size": requested_gop_size if hevc_active else None,
            "hevc_encode_scope": encode_scope if hevc_active else None,
            "hevc_effective_gop_size": effective_gop_size if hevc_active else None,
            "hevc_anchor_policy": anchor_policy if hevc_active else None,
            "hevc_anchor_reducer": i_mode if hevc_active else None,
            "hevc_codec_layout": HEVC_CODEC_LAYOUT if hevc_active else None,
            "hevc_signal_policy": HEVC_SIGNAL_POLICY if hevc_active else None,
            "selection_seed": (
                config.seed if run_pruned and p_mode == "random" else None
            ),
        },
    )


def process_vivit_frames(session, frames_batch, device):
    inputs = session.image_processor(
        frames_batch,
        return_tensors="pt",
        size={"shortest_edge": 256},
        do_normalize=False,
    )
    return inputs["pixel_values"].to(device)


def prepare_vivit_state_pruned(session, pixel_values, selector_outputs):
    plan = (
        session.pruning_plan.bind(selector_outputs)
        if selector_outputs is not None
        else session.pruning_plan
    )
    return session.model.vivit.embeddings(pixel_values), plan


def forward_vivit_state_prepared(session, prepared_hidden_states, token_reduce=None):
    hidden_states, pruning_plan = prepared_hidden_states
    return forward_vivit_encoder(
        session.model,
        hidden_states,
        pruning_plan,
        token_reduce=token_reduce,
    )


def classify_hidden_state_parts(hidden_states_parts, classify_func):
    if isinstance(hidden_states_parts, torch.Tensor):
        return classify_func(hidden_states_parts)
    hidden_states_parts = [
        hidden_states.unsqueeze(0) if hidden_states.ndim == 2 else hidden_states
        for hidden_states in hidden_states_parts
    ]
    lengths = {hidden_states.shape[1] for hidden_states in hidden_states_parts}
    if len(lengths) == 1:
        return classify_func(torch.cat(hidden_states_parts, dim=0))
    return torch.cat(
        [classify_func(hidden_states) for hidden_states in hidden_states_parts],
        dim=0,
    )


def prune_vivit_hidden_states(model, hidden_states, pruning_plan):
    """Adapt the ViViT sequence layout to the model-neutral pruning API."""

    cls_token, tubelet_tokens = hidden_states[:, :1], hidden_states[:, 1:]
    batch_size, num_tubelets = tubelet_tokens.shape[:2]
    tubelet_size = model.config.tubelet_size
    tubelet_t = int(
        tubelet_size[0] if isinstance(tubelet_size, (list, tuple)) else tubelet_size
    )
    temporal_groups = int(model.config.num_frames) // tubelet_t
    grouped_tokens = tubelet_tokens.reshape(
        batch_size,
        temporal_groups,
        num_tubelets // temporal_groups,
        tubelet_tokens.shape[-1],
    )
    return prune_tokens(
        PruneTokens.from_partitioned(grouped_tokens, cls_tokens=cls_token),
        pruning_plan.config,
        pruning_plan.signal,
    ).pack_with_global_cls()


def forward_vivit_encoder(model, hidden_states, pruning_plan=None, token_reduce=None):
    """Run the encoder, pruning once at the plan's layer boundary."""

    def prune_now(state):
        def run():
            return prune_vivit_hidden_states(model, state, pruning_plan)

        return token_reduce(run) if token_reduce else run()

    layers = model.vivit.layers
    prune_layer = pruning_plan.layer if pruning_plan is not None else None
    for layer_index, layer in enumerate(layers):
        if layer_index == prune_layer:
            hidden_states = prune_now(hidden_states)
        # Pruning can leave one state per sample instead of a batched tensor.
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = layer(hidden_states, None)
        else:
            hidden_states = [
                layer(sample.unsqueeze(0), None).squeeze(0)
                for sample in hidden_states
            ]
    if prune_layer == len(layers):
        hidden_states = prune_now(hidden_states)

    return classify_hidden_state_parts(
        hidden_states,
        lambda value: model.classifier(model.vivit.layernorm(value)[:, 0, :]),
    )


@dataclass
class VivitRun:
    """Loaded classification backend consumed by the benchmark runner."""

    model_type: str
    model_name: str
    model: Any
    model_config: Any
    image_processor: Any
    device: torch.device
    num_frames: int
    image_size: int
    patch_size: int
    run_pruned: bool
    run_full: bool
    pruning_plan: LayerPruningPlan | None
    selector: Any
    process_frames_fn: Callable
    prepare_pruned_fn: Callable
    forward_pruned_fn: Callable
    info: str
    short_info: str
    accuracy_key: str
    total_tokens: int
    total_tokens_per_step: int
    keep_per_frame: int | None = None
    tubelet_size: int | None = None
    hevc_gop_size: int = 0
    hevc_effective_gop_size: int | None = None
    hevc_encode_scope: str = "sampled-clip"
    hevc_anchor_policy: str = "first"
    hevc_effective_gop_sizes: list[int] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_hevc(self):
        return bool(
            self.run_pruned
            and self.selector is not None
            and self.pruning_plan.config.signal_source == "hevc"
        )

    @property
    def needs_signal(self):
        return bool(self.run_pruned and self.selector is not None)

    def process_frames(self, frames_batch, device=None):
        return self.process_frames_fn(
            self,
            frames_batch,
            torch.device(device) if device is not None else self.device,
        )

    def select(self, video_paths, frame_indices, *, n_parallel=1, device=None,
               cache_keys=None, score_cache_dir=None):
        if self.selector is None:
            return None
        return self.selector.select(
            video_paths,
            frame_indices,
            n_parallel=n_parallel,
            device=torch.device(device) if device is not None else self.device,
            cache_keys=cache_keys,
            score_cache_dir=score_cache_dir,
        )

    def prepare_pruned(self, pixel_values, selector_outputs=None):
        return self.prepare_pruned_fn(self, pixel_values, selector_outputs)

    def forward_pruned_prepared(self, prepared, token_reduce=None):
        return self.forward_pruned_fn(self, prepared, token_reduce)

    def forward_pruned(self, pixel_values, selector_outputs=None, token_reduce=None):
        prepared = self.prepare_pruned(pixel_values, selector_outputs)
        return self.forward_pruned_prepared(prepared, token_reduce)

    def forward_full(self, pixel_values):
        return self.model(pixel_values).logits
