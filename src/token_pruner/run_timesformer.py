"""TimeSFormer classification backend with divided space-time pruning."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import torch
from transformers import (
    AutoImageProcessor,
    TimesformerConfig,
    TimesformerForVideoClassification,
)

from .selectors import (
    BaselineTokenSelector,
    patch_selector,
    spatial_patch_score_selector,
)
from .selection import (
    BASELINE_MODES,
    LayerPruningPlan,
    build_model_pruning_config,
    build_shared_folding_config,
    resolve_pruning_layer,
    validate_pruning_modes,
)
from .tokens import SignalSource, PruneTokens
from .engine import prune_tokens
from .shared_folding import resolve_shared_folding_options
from .task_io import resolve_run_modes
from .hevc import HEVC_CODEC_LAYOUT, HEVC_SIGNAL_POLICY, resolve_hevc_encode_scope

TIMESFORMER_LEGAL_MODES = frozenset({
    *((stage, "global", "global_folder", "global_folder") for stage in ("input", "last")),
    *((stage, "local", "folder", p) for stage in ("input", "last")
      for p in ("hevc", "folder")),
    # Baselines are per frame, never global: see the note in `load_timesformer`.
    *((stage, "local", p, p) for stage in ("input", "last")
      for p in BASELINE_MODES),
    ("input", "shared", "shared_folding", "shared_folding"),
    ("last", "shared", "shared_folding", "shared_folding"),
})

TIMESFORMER_MODEL_NAME = "facebook/timesformer-base-finetuned-k400"


def resolve_timesformer_keep_budget(config, total_patches_per_frame, num_frames):
    total_patches = total_patches_per_frame * num_frames
    target_keep_total = max(1, int(total_patches * config.k_keep_rate))
    if config.k_keep is not None:
        keep_per_frame = int(config.k_keep)
    else:
        keep_per_frame = max(
            1,
            int(total_patches_per_frame * config.k_keep_rate),
        )
    effective_keep_total = keep_per_frame * num_frames
    effective_keep_rate = effective_keep_total / total_patches
    return keep_per_frame, target_keep_total, effective_keep_total, effective_keep_rate


def load_timesformer(config, device):
    """Load TimeSFormer and return the typed classification session."""

    device = torch.device(device)
    shared_folding = config.shared_folding is not None
    model_name = config.model_name or TIMESFORMER_MODEL_NAME
    prune_mode = "local" if shared_folding else config.prune_mode or "local"
    i_mode = "folder" if shared_folding else config.i_mode or "folder"
    p_mode = "folder" if shared_folding else config.p_mode or "hevc"
    run_pruned, run_full = resolve_run_modes(config.run_mode)
    if p_mode in BASELINE_MODES:
        # Divided space-time needs equal frames, so baselines draw within each frame.
        prune_mode = "local"
        i_mode = p_mode
    if run_pruned and not shared_folding:
        validate_pruning_modes(
            TIMESFORMER_LEGAL_MODES,
            "timesformer",
            config.prune_stage,
            prune_mode,
            i_mode,
            p_mode,
        )

    model_config = TimesformerConfig.from_pretrained(model_name)
    image_processor = AutoImageProcessor.from_pretrained(
        model_name,
        backend="pil",
    )
    model_num_frames = int(model_config.num_frames)
    num_frames = (
        int(config.num_frames)
        if config.num_frames is not None
        else model_num_frames
    )
    image_size = int(model_config.image_size)
    patch_size = (
        int(model_config.patch_size)
        if model_config.patch_size is not None
        else config.patch_size
    )
    requested_gop_size = int(config.hevc_gop_size)
    encode_scope = resolve_hevc_encode_scope(config.hevc_encode_scope)
    hevc_active = bool(
        run_pruned
        and (
            p_mode == "hevc"
            or (shared_folding and config.shared_folding.mode == "hevc-avg")
        )
    )
    # The effective GOP resolves only once a full source video is encoded.
    effective_gop_size = None
    anchor_policy = (
        str(config.hevc_anchor_policy)
        if hevc_active and not shared_folding
        else "none"
    )
    model_config.num_frames = num_frames
    model = TimesformerForVideoClassification.from_pretrained(
        model_name,
        config=model_config,
    )
    model.to(device).eval()

    total_per_frame = (image_size // patch_size) ** 2
    total_tokens = num_frames * total_per_frame
    (
        regular_keep,
        regular_target_total,
        regular_effective_total,
        regular_effective_rate,
    ) = resolve_timesformer_keep_budget(config, total_per_frame, num_frames)

    resolved_folding = None
    if shared_folding:
        resolved_folding, keep_per_frame = resolve_shared_folding_options(
            replace(
                config.shared_folding,
                patch_width=image_size // patch_size,
            ),
            total_columns=total_per_frame,
            k_keep_rate=config.k_keep_rate,
        )
        target_keep_total = keep_per_frame * num_frames
        effective_keep_total = target_keep_total
        effective_keep_rate = keep_per_frame / total_per_frame
        pruning_config = (
            build_shared_folding_config(
                options=resolved_folding,
                keep_total=target_keep_total,
                model_family="timesformer",
                stage=config.prune_stage,
                legal_modes=TIMESFORMER_LEGAL_MODES,
            )
            if run_pruned
            else None
        )
    else:
        keep_per_frame = regular_keep
        target_keep_total = regular_target_total
        effective_keep_total = regular_effective_total
        effective_keep_rate = regular_effective_rate
        pruning_config = (
            build_model_pruning_config(
                model_family="timesformer",
                stage=config.prune_stage,
                prune_mode=prune_mode,
                i_mode=i_mode,
                p_mode=p_mode,
                legal_modes=TIMESFORMER_LEGAL_MODES,
                uniform_partitions=True,
                keep_per_partition=keep_per_frame,
                keep_total=effective_keep_total,
            )
            if run_pruned
            else None
        )

    prune_layer = resolve_pruning_layer(
        config.prune_layer,
        stage=config.prune_stage,
        num_layers=len(model.timesformer.encoder.layer),
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
        image_shape = (1, 3, num_frames, image_size, image_size)
        selector = (
            spatial_patch_score_selector(
                patch_size,
                image_shape,
                hevc_gop_size=requested_gop_size,
                hevc_anchor_policy=anchor_policy,
            )
            if shared_folding
            else patch_selector(
                patch_size,
                keep_per_frame,
                image_shape,
                hevc_gop_size=requested_gop_size,
                hevc_anchor_policy=anchor_policy,
            )
        )
    elif pruning_config is not None and p_mode in BASELINE_MODES:
        selector = BaselineTokenSelector(
            p_mode,
            total_tokens=total_tokens,
            keep_total=effective_keep_total,
            num_partitions=num_frames,
            seed=config.seed,
            keep_per_partition=keep_per_frame,
            local=True,
        )

    if shared_folding:
        info = (
            f"Model = {model_name}, shared-folding {resolved_folding.mode}, "
            f"K={keep_per_frame}/{total_per_frame} columns per frame "
            f"(rate={effective_keep_rate:.6f}), "
            f"HEVC-score={selector is not None}"
        )
        short_info = (
            f"TimeSFormer shared-folding {resolved_folding.mode}, "
            f"K={keep_per_frame}/{total_per_frame}"
        )
        accuracy_key = "shared_folding"
    elif run_pruned:
        budget_info = (
            f"target K={target_keep_total}/{total_tokens}, "
            f"effective K={effective_keep_total}/{total_tokens} "
            f"(rate={effective_keep_rate:.6f}), "
            f"local K={keep_per_frame}/{total_per_frame} per frame"
        )
        info = (
            f"Model = {model_name}, {budget_info} | "
            f"I:{i_mode}, P:{p_mode} | patch size={patch_size}"
        )
        short_info = (
            f"TimeSFormer {prune_mode},{budget_info} | I:{i_mode}, P:{p_mode}"
        )
        accuracy_key = p_mode
    else:
        info = (
            f"Model = {model_name}, Full "
            f"({total_per_frame}/{total_per_frame} patches per frame) | "
            f"patch size={patch_size}"
        )
        short_info = f"TimeSFormer full, patch size={patch_size}"
        accuracy_key = "full"

    params = {
        "k_keep_rate": config.k_keep_rate,
        "k_keep": keep_per_frame if run_pruned else None,
        "target_keep_total": target_keep_total if run_pruned else None,
        "effective_keep_total": effective_keep_total if run_pruned else None,
        "effective_keep_rate": effective_keep_rate if run_pruned else None,
        "total_patches_per_frame": total_per_frame,
        "total_patches": total_tokens,
        "patch_size": patch_size,
        "i_mode": None if shared_folding or not run_pruned else i_mode,
        "p_mode": None if shared_folding or not run_pruned else p_mode,
        "prune_mode": "shared" if shared_folding else prune_mode,
        "prune_stage": config.prune_stage if run_pruned else None,
        "prune_layer": prune_layer if run_pruned else None,
        "hevc_gop_size": requested_gop_size if hevc_active else None,
        "hevc_encode_scope": encode_scope if hevc_active else None,
        "hevc_effective_gop_size": effective_gop_size if hevc_active else None,
        "hevc_anchor_policy": anchor_policy if hevc_active else None,
        "hevc_anchor_reducer": (
            None if shared_folding else i_mode
        ) if hevc_active else None,
        "hevc_codec_layout": HEVC_CODEC_LAYOUT if hevc_active else None,
        "hevc_signal_policy": HEVC_SIGNAL_POLICY if hevc_active else None,
    }
    if shared_folding:
        params.update(
            {
                "pruning_strategy": "shared-folding",
                "folding_mode": resolved_folding.mode,
                "fold_block_wise": resolved_folding.block_wise,
                "fold_block_size": resolved_folding.block_size,
                "fold_pooling": resolved_folding.pooling,
                "fold_slots_per_block": resolved_folding.slots_per_block,
                "fold_min_slots_per_block": (
                    resolved_folding.min_slots_per_block
                ),
                "shared_folding_uses_hevc": selector is not None,
            }
        )

    return TimeSformerRun(
        model_type="timesformer",
        model_name=model_name,
        model=model,
        model_config=model_config,
        image_processor=image_processor,
        device=device,
        num_frames=num_frames,
        image_size=image_size,
        patch_size=patch_size,
        run_pruned=run_pruned,
        run_full=run_full,
        pruning_plan=pruning_plan,
        selector=selector,
        process_frames_fn=process_timesformer_frames,
        prepare_pruned_fn=prepare_timesformer_state_pruned,
        forward_pruned_fn=forward_timesformer_state_prepared,
        info=info,
        short_info=short_info,
        accuracy_key=accuracy_key,
        total_tokens=total_tokens,
        total_tokens_per_step=total_per_frame,
        keep_per_frame=keep_per_frame,
        hevc_gop_size=requested_gop_size,
        hevc_encode_scope=encode_scope,
        hevc_effective_gop_size=effective_gop_size,
        hevc_anchor_policy=anchor_policy,
        params=params,
    )


def process_timesformer_frames(session, frames_batch, device):
    inputs = session.image_processor(
        frames_batch,
        return_tensors="pt",
        size={"shortest_edge": 224},
        do_normalize=True,
    )
    return inputs["pixel_values"].to(device)


def prepare_timesformer_state_pruned(session, pixel_values, selector_outputs):
    hidden_states, num_frames, num_patches = prepare_timesformer_hidden_states(
        session.model,
        pixel_values,
    )
    plan = (
        session.pruning_plan.bind(selector_outputs)
        if selector_outputs is not None
        else session.pruning_plan
    )
    return hidden_states, num_frames, num_patches, plan


def forward_timesformer_state_prepared(
    session,
    prepared_hidden_states,
    token_reduce=None,
):
    hidden_states, num_frames, num_patches, pruning_plan = prepared_hidden_states
    return forward_timesformer_encoder(
        session.model,
        hidden_states,
        num_frames,
        num_patches,
        pruning_plan,
        token_reduce=token_reduce,
    )


def resize_spatial_position_embeddings(model, num_patches, patch_width):
    embeddings = model.timesformer.embeddings
    position_embeddings = embeddings.position_embeddings
    if num_patches + 1 == position_embeddings.shape[1]:
        return position_embeddings[:, :1, :], position_embeddings[:, 1:, :]

    cls_pos_embed = position_embeddings[:, :1, :]
    patch_pos_embed = position_embeddings[:, 1:, :].transpose(1, 2)
    source_patch_num = int(patch_pos_embed.shape[-1] ** 0.5)
    patch_height = num_patches // patch_width
    patch_pos_embed = patch_pos_embed.reshape(
        1,
        position_embeddings.shape[-1],
        source_patch_num,
        source_patch_num,
    )
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed,
        size=(patch_height, patch_width),
        mode="nearest",
    )
    patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
    return cls_pos_embed, patch_pos_embed


# Follows Transformers' TimesformerLayer.forward (Apache-2.0); see NOTICE.
def forward_timesformer_layer(layer, hidden_states, num_frames, num_patches):
    """Run one TimeSFormer layer with the current spatial-token count."""

    if layer.attention_type != "divided_space_time":
        return layer(hidden_states, output_attentions=False)[0]

    batch_size = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[-1]

    temporal_embedding = hidden_states[:, 1:, :]
    temporal_attention_input = temporal_embedding.reshape(
        batch_size, num_patches, num_frames, hidden_dim
    ).reshape(batch_size * num_patches, num_frames, hidden_dim)
    temporal_attention_outputs = layer.temporal_attention(
        layer.temporal_layernorm(temporal_attention_input),
    )
    residual_temporal = layer.drop_path(temporal_attention_outputs[0])
    residual_temporal = residual_temporal.reshape(
        batch_size, num_patches, num_frames, hidden_dim
    ).reshape(batch_size, num_patches * num_frames, hidden_dim)
    residual_temporal = layer.temporal_dense(residual_temporal)
    temporal_embedding = hidden_states[:, 1:, :] + residual_temporal

    init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
    cls_token = init_cls_token.repeat(1, num_frames, 1).reshape(
        batch_size * num_frames, 1, hidden_dim
    )
    spatial_embedding = (
        temporal_embedding.reshape(batch_size, num_patches, num_frames, hidden_dim)
        .permute(0, 2, 1, 3)
        .reshape(batch_size * num_frames, num_patches, hidden_dim)
    )
    spatial_embedding = torch.cat((cls_token, spatial_embedding), dim=1)

    spatial_attention_outputs = layer.attention(
        layer.layernorm_before(spatial_embedding),
        output_attentions=False,
    )
    residual_spatial = layer.drop_path(spatial_attention_outputs[0])

    cls_token = residual_spatial[:, 0, :]
    cls_token = cls_token.reshape(batch_size, num_frames, hidden_dim).mean(
        1, keepdim=True
    )
    residual_spatial = residual_spatial[:, 1:, :]
    residual_spatial = (
        residual_spatial.reshape(batch_size, num_frames, num_patches, hidden_dim)
        .permute(0, 2, 1, 3)
        .reshape(batch_size, num_patches * num_frames, hidden_dim)
    )

    hidden_states = torch.cat((init_cls_token, temporal_embedding), dim=1) + torch.cat(
        (cls_token, residual_spatial), dim=1
    )
    layer_output = layer.layernorm_after(hidden_states)
    layer_output = layer.intermediate(layer_output)
    layer_output = layer.output(layer_output)
    return hidden_states + layer.drop_path(layer_output)


def prune_timesformer_hidden_states(
    hidden_states,
    num_frames,
    num_patches,
    pruning_plan,
):
    """Adapt TimeSFormer layout to the model-neutral pruning API."""

    batch_size, _, hidden_dim = hidden_states.shape
    cls_token = hidden_states[:, :1]
    patches = (
        hidden_states[:, 1:]
        .reshape(batch_size, num_patches, num_frames, hidden_dim)
        .permute(0, 2, 1, 3)
    )
    selected = prune_tokens(
        PruneTokens.from_partitioned(patches, cls_tokens=cls_token),
        pruning_plan.config,
        pruning_plan.signal,
    ).require_uniform_partitions()
    return pack_timesformer_hidden_states(cls_token, selected)


def forward_timesformer_encoder(
    model,
    hidden_states,
    num_frames,
    num_patches,
    pruning_plan=None,
    token_reduce=None,
):
    """Run the encoder, pruning once at the plan's layer boundary."""

    def prune_now(state, frames, patches):
        def run():
            return prune_timesformer_hidden_states(
                state, frames, patches, pruning_plan
            )

        return token_reduce(run) if token_reduce else run()

    layers = model.timesformer.encoder.layer
    prune_layer = pruning_plan.layer if pruning_plan is not None else None
    for layer_index, layer in enumerate(layers):
        # Pruning changes the per-frame patch count every later layer needs.
        if layer_index == prune_layer:
            hidden_states, num_frames, num_patches = prune_now(
                hidden_states, num_frames, num_patches
            )
        hidden_states = forward_timesformer_layer(
            layer, hidden_states, num_frames, num_patches
        )
    if prune_layer == len(layers):
        hidden_states, num_frames, num_patches = prune_now(
            hidden_states, num_frames, num_patches
        )
    sequence_output = model.timesformer.layernorm(hidden_states)
    return model.classifier(sequence_output[:, 0, :])


def add_timesformer_time_embeddings(embeddings, patches, frames):
    if embeddings.attention_type == "space_only":
        return patches
    time_embeddings = embeddings.time_embeddings
    if frames != time_embeddings.shape[1]:
        time_embeddings = torch.nn.functional.interpolate(
            time_embeddings.transpose(1, 2), size=frames, mode="nearest"
        ).transpose(1, 2)
    return embeddings.time_drop(
        patches + time_embeddings.view(1, frames, 1, patches.shape[-1])
    )


def pack_timesformer_hidden_states(cls_token, patches):
    batch_size, frames, num_patches, hidden_dim = patches.shape
    patch_sequence = patches.permute(0, 2, 1, 3).reshape(
        batch_size, num_patches * frames, hidden_dim
    )
    return torch.cat((cls_token, patch_sequence), dim=1), frames, num_patches


def prepare_timesformer_hidden_states(model, pixel_values):
    embeddings = model.timesformer.embeddings
    patches, _, patch_width = embeddings.patch_embeddings(pixel_values)
    batch_size, frames = pixel_values.shape[:2]
    num_patches, hidden_dim = patches.shape[1:]
    patches = patches.view(batch_size, frames, num_patches, hidden_dim)

    cls_position, patch_position = resize_spatial_position_embeddings(
        model, num_patches, patch_width
    )
    cls_token = embeddings.cls_token.expand(batch_size, -1, -1) + cls_position
    patches = patches + patch_position.view(1, 1, -1, hidden_dim)
    patches = add_timesformer_time_embeddings(embeddings, patches, frames)
    return pack_timesformer_hidden_states(cls_token, patches)


@dataclass
class TimeSformerRun:
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
