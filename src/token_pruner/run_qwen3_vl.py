"""Framework-independent Hugging Face Qwen3-VL backend."""

import math
import os
from pathlib import Path
import types

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
)
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms

from .packed_cells import prune_flat_visual_tokens, kept_cell_indices
from .generation import run_generate_until
from .hevc import resolve_hevc_encode_scope
from .layer_boundary import EncoderPruningContext
from .selection import (
    BASELINE_MODES,
    LayerPruningPlan,
    build_pruning_config,
    normalize_pruning_scope,
    resolve_visual_budget,
    validate_pruning_modes,
)
from .selectors import BaselineTokenSelector, TubeletSelector
from .signals import SignalOptions
from .task_io import (
    STAGE_TIMER_NAMES,
    GenerationStepObserver,
    StageTimer,
    as_bool,
    collect_batch_timing as _collect_batch_timing,
    current_keep_budgets as _current_keep_budgets,
    decode_tokens as _decode_tokens,
    greedy_decoding_enabled,
    inference_dtype,
    inference_dtype_name,
    max_new_tokens_override,
    normalize_until,
    pad_row,
    placeholder_runs,
    prepare_batch as _prepare_batch,
    prepare_sample as _prepare_sample,
    qwen3_vl_decoding_policy,
    resolve_pretrained_source,
    resolve_run_modes,
    stop_strings_enabled,
    _begin_generation as shared_begin_generation,
)
from .tokens import PruningScope, SignalSource, TokenReducer

#: Merged cells per frame side. 12 -> 144 tokens/frame; divides every sweep rate.
DEFAULT_VISUAL_GRID_SIDE = 12

#: Closing tag of a Qwen3-VL reasoning trace.
REASONING_CLOSE_TAG = "</think>"

#: Thinking budget; the Instruct default is spent inside the trace.
THINKING_MAX_NEW_TOKENS = 4096

#: Checkpoint-owned sampling. Task budgets and stopping rules stay outside.
CHECKPOINT_DECODING_FIELDS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
)


def resize_video_placeholders(
    inputs,
    video_token_id,
    group_cells,
    *,
    pad_token_id=0,
    padding_side="left",
):
    """Keep only the surviving cells of each video-placeholder run."""

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    token_types = inputs["mm_token_type_ids"]
    video_token_id = int(video_token_id)

    rows, masks, type_rows, keep_masks = [], [], [], []
    for row_index, row in enumerate(input_ids):
        valid = (
            attention_mask[row_index].bool()
            if attention_mask is not None
            else torch.ones(row.shape[0], dtype=torch.bool, device=row.device)
        )
        tokens = row[valid]
        keep = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        runs = placeholder_runs(tokens, video_token_id)
        sample_cells = group_cells[row_index]
        if len(runs) != len(sample_cells):
            raise RuntimeError(
                f"Sample {row_index} has {len(runs)} video placeholder runs but "
                f"{len(sample_cells)} temporal groups were pruned."
            )
        for (start, end), cells in zip(runs, sample_cells):
            cells = cells.to(keep.device)
            if int(cells.numel()) == 0 or int(cells.max()) >= end - start:
                raise RuntimeError(
                    f"Selected cells do not fit the placeholder run "
                    f"[{start}, {end}) of sample {row_index}."
                )
            keep[start:end] = False
            keep[start + cells] = True
        rows.append(tokens[keep])
        masks.append(torch.ones(int(keep.sum()), dtype=torch.long, device=row.device))
        type_rows.append(token_types[row_index][valid][keep])
        keep_masks.append(keep)

    max_length = max(row.shape[0] for row in rows)

    inputs["input_ids"] = torch.stack([
        pad_row(row, max_length, pad_token_id, padding_side)
        for row in rows
    ])
    if attention_mask is not None:
        inputs["attention_mask"] = torch.stack([
            pad_row(mask, max_length, 0, padding_side)
            for mask in masks
        ]).to(
            dtype=attention_mask.dtype
        )
    inputs["mm_token_type_ids"] = torch.stack([
        pad_row(types, max_length, 0, padding_side)
        for types in type_rows
    ])
    return inputs, keep_masks


class Qwen3VlHfBackend:
    """Load and run Qwen3-VL without depending on an evaluation framework."""

    family = "qwen3_vl"
    vision_backend = "qwen3_vl_vit"
    needs_video_fps = True
    reasoning_close_tag = None
    default_max_new_tokens = 128

    def _resolve_input_length(self, inputs):
        """Prompt length used to split prompt from completion."""
        return inputs["input_ids"].shape[1]

    def _slice_generated(self, sequence, input_length):
        """Completion tokens of one returned sequence."""
        return sequence[input_length:]

    def generate_tokens(self, inputs, generation_kwargs, stop_sequences=None):
        generation_kwargs, stopping = self._begin_generation(generation_kwargs)
        if stop_sequences and stop_strings_enabled():
            generation_kwargs["stop_strings"] = list(stop_sequences)
            generation_kwargs["tokenizer"] = self.processor.tokenizer
        return self.model.generate(
            **inputs,
            **generation_kwargs,
            stopping_criteria=stopping,
        )

    def _install_stage_timers(self, modules):
        """Attach one :class:`StageTimer` per named module."""
        self._stage_timers = {
            name: StageTimer(self.device) for name in STAGE_TIMER_NAMES
        }
        for module, name in modules:
            timer = self._stage_timers[name]
            module.register_forward_pre_hook(timer.pre_hook)
            module.register_forward_hook(timer.post_hook)
        self._step_observer = GenerationStepObserver(self.device)

    def _begin_generation(self, generation_kwargs):
        return shared_begin_generation(self, generation_kwargs)

    @property
    def model_name(self):
        return Path(self.source).name

    @staticmethod
    def clone_inputs(inputs):
        return dict(inputs)

    def _base_prepare_generation_kwargs(self, gen_kwargs):
        generation_kwargs = dict(gen_kwargs or {})
        stop_sequences = normalize_until(generation_kwargs.pop("until", None))
        generation_kwargs.setdefault("max_new_tokens", self.default_max_new_tokens)
        override = max_new_tokens_override()
        if override is not None:
            generation_kwargs["max_new_tokens"] = override
        if greedy_decoding_enabled():
            generation_kwargs["do_sample"] = False
        return generation_kwargs, stop_sequences

    def prepare_sample(self, messages, media, num_frames):
        return _prepare_sample(self, messages, media, num_frames)

    def collect_batch_timing(self, output, inputs):
        return _collect_batch_timing(self, output, inputs)

    def decode_tokens(self, output, inputs, stop_sequences):
        return _decode_tokens(self, output, inputs, stop_sequences)

    def last_untrimmed_texts(self):
        """Return the last batch before stop-sequence trimming."""

        return list(getattr(self, "_untrimmed_texts", []))

    def __init__(
        self,
        *,
        pretrained,
        device,
        load_4bit=False,
        load_8bit=False,
        local_files_only=False,
        visual_grid_side=DEFAULT_VISUAL_GRID_SIDE,
    ):
        self.device = torch.device(device)
        load_4bit = as_bool(load_4bit)
        load_8bit = as_bool(load_8bit)

        self.source, local_files_only = resolve_pretrained_source(
            pretrained,
            local_files_only=local_files_only,
        )
        dtype = inference_dtype(self.device)
        self.model_dtype = dtype
        load_kwargs = {
            "dtype": dtype,
            "local_files_only": local_files_only,
        }
        if load_4bit or load_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                load_in_8bit=load_8bit,
                bnb_4bit_compute_dtype=dtype,
            )
            load_kwargs["device_map"] = {"": str(self.device)}

        self.processor = AutoProcessor.from_pretrained(
            self.source,
            local_files_only=local_files_only,
        )
        self.processor.tokenizer.padding_side = "left"
        self.is_reasoning = self._detect_reasoning_template()
        if self.is_reasoning:
            self.reasoning_close_tag = REASONING_CLOSE_TAG
            self.default_max_new_tokens = THINKING_MAX_NEW_TOKENS
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.source,
            **load_kwargs,
        )
        if not load_4bit and not load_8bit:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.quantization = "4bit" if load_4bit else "8bit" if load_8bit else None

        vision_config = self.model.config.vision_config
        self.patch_size = int(vision_config.patch_size)
        self.spatial_merge_size = int(vision_config.spatial_merge_size)
        self.temporal_patch_size = int(vision_config.temporal_patch_size)
        self.visual_grid_side = int(visual_grid_side)
        #: Side of the pinned square frame, in pixels.
        self.frame_size = (
            self.visual_grid_side * self.spatial_merge_size * self.patch_size
        )
        #: Language-model visual tokens produced by one temporal patch.
        self.cells_per_group = self.visual_grid_side**2
        self.model.model.visual._token_pruner_grid_side = self.visual_grid_side

        # Set per pruned generation; None restores the stock rope path.
        self._shim_state = {}
        self._install_pruning_shims()

        self._install_stage_timers(
            (
                (self.model.model.visual, "vision_encoder"),
                (self.model.model.visual.merger, "projector"),
                (self.model.model.language_model, "language_model"),
            )
        )

    def _detect_reasoning_template(self):
        """Whether this checkpoint's generation prompt pre-opens a trace."""

        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": ""}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered.rstrip().endswith("<think>")

    def vision_tower(self):
        return self.model.model.visual

    def token_geometry(self):
        cell_size = self.patch_size * self.spatial_merge_size
        return self.visual_grid_side * cell_size, cell_size

    def partition_count(self, num_frames):
        return math.ceil(int(num_frames) / self.temporal_patch_size)

    def stage_layer(self, prune_layer, *, stage, num_layers, boundaries):
        """``last`` prunes at the first DeepStack boundary the language model reads."""

        if prune_layer is not None:
            layer = int(prune_layer)
            if layer < 0:
                layer += int(num_layers) + 1
            return layer
        return 0 if stage == "input" else min(boundaries)

    def install_pruning(self, context):
        install_qwen3_vl_encoder_pruning(self.vision_tower(), context)

    def create_signal_provider(
        self,
        *,
        config,
        patch_size,
        num_frames,
        num_partitions,
        image_size,
        selection_seed,
        hevc_gop_size=0,
        hevc_anchor_policy="first",
    ):
        if config.signal_source in {SignalSource.RANDOM, SignalSource.UNIFORM}:
            return BaselineTokenSelector(
                config.signal_source.value,
                total_tokens=int(num_partitions)
                * (int(image_size) // int(patch_size)) ** 2,
                keep_total=config.keep_total,
                num_partitions=num_partitions,
                seed=selection_seed,
                local=config.scope == PruningScope.LOCAL,
                keep_per_partition=(
                    config.keep_per_partition
                    if config.uniform_partitions
                    else None
                ),
                min_per_partition=config.min_per_partition,
            )
        return TubeletSelector(
            patch_size=patch_size,
            tubelet_size=self.temporal_patch_size,
            tubelet_budget=config.keep_per_partition,
            image_shape=(1, 3, int(num_frames), int(image_size), int(image_size)),
            prune_mode=config.scope.value,
            preserve_anchor_tubelets=(
                config.anchor_reducer == TokenReducer.PRESERVE
            ),
            total_tubelet_budget=config.keep_total,
            min_per_group=config.min_per_partition,
            hevc_gop_size=hevc_gop_size,
            hevc_anchor_policy=hevc_anchor_policy,
        )

    def per_group_cells(self, session, num_frames, batch_size=1):
        return kept_cell_indices(
            session.context.plan,
            batch_size=int(batch_size),
            num_groups=int(num_frames),
            cells_per_group=session.total_patches,
            device=self.device,
        )

    def adapt_lm_inputs(self, inputs, session, effective_keep_total):
        pruned_inputs = self.clone_inputs(inputs)
        if not session.pruning_active:
            return pruned_inputs
        batch_size = int(pruned_inputs["input_ids"].shape[0])
        group_cells = self.per_group_cells(
            session,
            session.context.num_groups,
            batch_size,
        )
        for sample_idx, row in enumerate(group_cells):
            kept = sum(int(cells.numel()) for cells in row)
            if kept != int(effective_keep_total):
                raise RuntimeError(
                    f"Sample {sample_idx}'s per-group selections do not add up "
                    f"to the batch budget: {[int(c.numel()) for c in row]} sums "
                    f"to {kept}, expected {int(effective_keep_total)}."
                )
        session.context.expected_group_cells = group_cells
        return self.resize_pruned_inputs(pruned_inputs, group_cells)

    def _install_pruning_shims(self):
        """Teach ``Qwen3VLModel`` that the tower may return fewer tokens."""

        install_lm_shims(self.model, self._shim_state)

    def publish_rope_subset(self, state):
        """Record which visual columns survived, for the rope shim."""

        if state is None:
            self._shim_state.pop("rope", None)
        else:
            self._shim_state["rope"] = state

    def _generation_config(self):
        return self.model.generation_config

    def decoding_parameters(self):
        """Resolved Qwen policy and its model-owned generation fields."""

        policy = qwen3_vl_decoding_policy()
        if policy == "task":
            return {"policy": policy}
        if policy == "greedy":
            return {"policy": policy, "do_sample": False}

        config = self._generation_config()
        parameters = {
            name: getattr(config, name)
            for name in CHECKPOINT_DECODING_FIELDS
            if getattr(config, name, None) is not None
        }
        if parameters.get("do_sample") is not True:
            raise RuntimeError(
                "QWEN3_VL_DECODING_POLICY=checkpoint requires the loaded "
                "checkpoint to declare do_sample=true."
            )
        return {"policy": policy, **parameters}

    def prepare_generation_kwargs(self, gen_kwargs):
        kwargs, stop_sequences = self._base_prepare_generation_kwargs(gen_kwargs)
        parameters = self.decoding_parameters()
        policy = parameters.pop("policy")
        if policy == "checkpoint":
            for name in CHECKPOINT_DECODING_FIELDS:
                kwargs.pop(name, None)
            kwargs.update(parameters)
        elif policy == "greedy":
            kwargs["do_sample"] = False
        return kwargs, stop_sequences

    def _decode_ids(self, token_ids):
        return self.processor.decode(token_ids, skip_special_tokens=True)

    def _prompt_from_messages(self, messages):
        """Render the checkpoint's own chat template."""

        rendered = []
        video_inserted = False
        for message in messages:
            content = []
            if message.get("role") == "user" and not video_inserted:
                content.append({"type": "video"})
                video_inserted = True
            content.extend(
                {"type": "text", "text": block["text"]}
                for block in message.get("content", [])
                if block.get("type") == "text" and block.get("text")
            )
            if content:
                rendered.append({"role": message["role"], "content": content})
        processor = self.processor
        wrapped_video = (
            processor.vision_start_token
            + processor.video_token
            + processor.vision_end_token
        )
        return processor.apply_chat_template(
            rendered,
            tokenize=False,
            add_generation_prompt=True,
        ).replace(wrapped_video, processor.video_token)

    def _pin_frames(self, frames):
        """Resize uint8 ``[T, H, W, C]`` to ``[T, S, S, C]``; temporal grouping is Qwen's."""

        tensor = torch.as_tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(self.frame_size, self.frame_size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        return tensor.numpy()

    def _video_metadata(self, sample):
        from transformers.video_utils import VideoMetadata

        indices = [int(index) for index in sample.frame_indices]
        fps = float(sample.fps) if sample.fps else 24.0
        return VideoMetadata(
            total_num_frames=len(indices),
            fps=fps,
            duration=len(indices) / fps,
            video_backend="decord",
            frames_indices=indices,
        )

    def prepare_batch_inputs(self, samples):
        """Tokenize and transfer one homogeneous generation batch."""

        inputs = self.processor(
            text=[sample.prompt for sample in samples],
            videos=[self._pin_frames(sample.sampled_frames) for sample in samples],
            video_metadata=[self._video_metadata(sample) for sample in samples],
            do_sample_frames=False,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        # Disarm the rope shim: unpruned batch, ids come from inputs later.
        self.publish_rope_subset(None)
        return inputs

    def resize_pruned_inputs(self, inputs, group_cells):
        """Drop the pruned placeholders and arm the rope shim."""

        inputs, rope_state = shrink_visual_placeholders(
            self.model, inputs, group_cells,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            padding_side=self.processor.tokenizer.padding_side,
        )
        self.publish_rope_subset(rope_state)
        return inputs


QWEN_MODE_KEYS = (
    ("local", "preserve", "hevc"),
    ("global", "preserve", "hevc"),
    ("shared", "shared_hevc", "shared_hevc"),
    ("local", "random", "random"),
    ("local", "uniform", "uniform"),
    ("global", "random", "random"),
    ("global", "uniform", "uniform"),
)
QWEN3_VL_LEGAL_MODES = frozenset(
    (stage, *mode) for stage in ("input", "last") for mode in QWEN_MODE_KEYS
)


class Qwen3VlPruning:
    def __init__(self, **values):
        self.__dict__.update(values)
        self.realized_keep_per_partition_min = []
        self.realized_keep_per_partition_max = []
        self.hevc_effective_gop_sizes = []
        self.hevc_artifact_keys = []

    @classmethod
    def create(
        cls,
        backend,
        *,
        num_frames,
        prune_stage,
        prune_mode,
        i_mode,
        p_mode,
        k_keep_rate,
        run_mode,
        hevc_n_parallel,
        hevc_dir,
        hevc_permanent_dir=None,
        hevc_gop_size=0,
        hevc_encode_scope="sampled-clip",
        hevc_anchor_policy="first",
        score_reduce="max",
        folding_mode="temporal-diff",
        fold_block_size=2,
        fold_pooling="weighted",
        prune_layer=None,
        selection_seed=42,
    ):
        device = torch.device(backend.device)
        num_frames = int(num_frames)
        run_pruned, _ = resolve_run_modes(run_mode)
        prune_stage = str(prune_stage)
        p_mode = str(p_mode)
        if p_mode in BASELINE_MODES:
            i_mode = p_mode
        prune_mode = normalize_pruning_scope(
            "qwen3_vl", prune_stage, prune_mode
        ).value
        validate_pruning_modes(
            QWEN3_VL_LEGAL_MODES,
            "qwen3_vl",
            prune_stage,
            prune_mode,
            i_mode,
            p_mode,
        )
        folding_mode = str(folding_mode)
        requested_gop_size = int(hevc_gop_size)
        encode_scope = resolve_hevc_encode_scope(hevc_encode_scope)
        hevc_active = bool(
            run_pruned
            and (
                p_mode == "hevc"
                or (p_mode == "shared_folding" and folding_mode == "hevc-avg")
            )
        )
        anchor_policy = (
            str(hevc_anchor_policy)
            if hevc_active and prune_mode != "shared"
            else "none"
        )

        tower = backend.model.model.visual
        image_size, patch_size = backend.token_geometry()
        num_partitions = backend.partition_count(num_frames)
        budget = resolve_visual_budget(
            image_size=image_size,
            patch_size=patch_size,
            num_partitions=num_partitions,
            k_keep_rate=k_keep_rate,
            run_mode=run_mode,
            run_pruned=run_pruned,
            p_mode=p_mode,
        )
        pruning_config = build_pruning_config(
            family="qwen3_vl",
            stage=prune_stage,
            scope=prune_mode,
            i_mode=i_mode,
            p_mode=p_mode,
            legal_modes=QWEN3_VL_LEGAL_MODES,
            budget=budget,
            patch_width=image_size // patch_size,
            uniform_partitions=False,
            min_per_partition=1,
            folding_mode=folding_mode,
            fold_block_size=fold_block_size,
            fold_pooling=fold_pooling,
        )

        num_layers = len(tower.blocks)
        boundaries = [*tower.deepstack_visual_indexes, num_layers]
        resolved_layer = backend.stage_layer(
            prune_layer,
            stage=prune_stage,
            num_layers=num_layers,
            boundaries=boundaries,
        )
        if budget.pruning_active and resolved_layer > min(boundaries):
            raise ValueError(
                f"prune_layer={resolved_layer} is after the earliest consumed "
                f"vision feature boundary {min(boundaries)}."
            )
        plan = (
            LayerPruningPlan(resolved_layer, pruning_config)
            if pruning_config is not None
            else None
        )
        context = EncoderPruningContext(
            enabled=budget.pruning_active,
            num_groups=num_partitions,
            plan=plan,
        )
        if budget.pruning_active:
            backend.install_pruning(context)

        provider = None
        if budget.pruning_active and pruning_config.signal_source != SignalSource.NONE:
            provider = backend.create_signal_provider(
                config=pruning_config,
                patch_size=patch_size,
                num_frames=num_frames,
                num_partitions=num_partitions,
                image_size=image_size,
                selection_seed=selection_seed,
                hevc_gop_size=requested_gop_size,
                hevc_anchor_policy=anchor_policy,
            )
        return cls(
            context=context,
            signal_provider=provider,
            prune_stage=prune_stage,
            prune_mode=prune_mode,
            prune_layer=resolved_layer,
            i_mode=str(i_mode),
            p_mode=p_mode,
            keep_patches=budget.keep_per_partition,
            total_patches=budget.total_per_partition,
            total_patches_clip=budget.total_tokens,
            target_keep_total=budget.keep_total,
            pruning_active=budget.pruning_active,
            hevc_n_parallel=int(hevc_n_parallel),
            hevc_gop_size=requested_gop_size,
            hevc_encode_scope=encode_scope,
            hevc_effective_gop_size=None,
            hevc_anchor_policy=anchor_policy,
            hevc_dir=Path(hevc_dir).expanduser(),
            hevc_permanent_dir=(
                None if hevc_permanent_dir is None
                else Path(hevc_permanent_dir).expanduser()
            ),
            device=device,
            selection_seed=int(selection_seed),
        )

    @property
    def needs_hevc(self):
        return bool(
            self.context.plan is not None
            and self.context.plan.config.signal_source == SignalSource.HEVC
        )

    def enable(self):
        self.context.enabled = self.pruning_active

    def disable(self):
        self.context.enabled = False

    def current_keep_budgets(self, batch_size):
        return _current_keep_budgets(self, batch_size)

    def _begin_signal(self):
        if self.context.plan is not None:
            self.context.plan = self.context.plan.bind(None)
        self.context.reset_timing()
        return SignalOptions.from_session(self)

    def prepare_batch(self, video_paths, frame_indices, sampled_frames):
        return _prepare_batch(self, video_paths, frame_indices, sampled_frames)

    def prepare_inputs(self, backend, inputs, effective_keep_total):
        return backend.adapt_lm_inputs(inputs, self, effective_keep_total)

    def token_reduce_ms(self):
        return self.context.token_reduce_ms()


class Qwen3VlRegisteredModel(lmms):
    is_simple = False
    model_id = "qwen3_vl"
    family = model_id
    default_pretrained = "Qwen/Qwen3-VL-8B-Instruct"
    default_num_frames = 64
    default_prune_mode = "local"
    default_i_mode = "preserve"
    default_p_mode = "hevc"
    selector_id = "tubelet"
    uniform_partitions = False
    min_cells_per_partition = 1

    def __init__(
        self,
        pretrained=None,
        device="cuda",
        batch_size=1,
        num_frames=None,
        prune_mode=None,
        i_mode=None,
        p_mode=None,
        k_keep_rate=0.5,
        prune_stage="last",
        run_mode="pruned",
        load_4bit=False,
        load_8bit=False,
        results_path=None,
        hevc_n_parallel=6,
        hevc_gop_size=None,
        hevc_encode_scope="sampled-clip",
        hevc_anchor_policy="first",
        hevc_dir="data/inference_hevc",
        hevc_permanent_dir=None,
        score_reduce="max",
        folding_mode="temporal-diff",
        fold_block_size=2,
        fold_pooling="weighted",
        prune_layer=None,
        selection_seed=42,
        local_files_only=False,
        visual_grid_side=None,
        **_kwargs,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_dtype = inference_dtype_name(self.device)
        self.batch_size = int(batch_size)
        self.num_frames = int(self.default_num_frames if num_frames is None else num_frames)
        self.run_mode = str(run_mode)
        self.score_reduce = str(score_reduce)
        self.folding_mode = str(folding_mode)
        self.fold_block_size = int(fold_block_size)
        self.fold_pooling = str(fold_pooling)
        self.pretrained = str(self.default_pretrained if pretrained is None else pretrained)
        self.load_4bit = load_4bit
        self.load_8bit = load_8bit
        self.local_files_only = local_files_only
        self.prune_stage = str(prune_stage)
        self.p_mode = str(self.default_p_mode if p_mode is None else p_mode)
        self.i_mode = str(self.default_i_mode if i_mode is None else i_mode)
        if self.p_mode in BASELINE_MODES:
            self.i_mode = self.p_mode
        scope = self.default_prune_mode if prune_mode is None else prune_mode
        self.prune_mode = normalize_pruning_scope(
            self.family, self.prune_stage, scope
        ).value
        self.prune_layer = int(prune_layer) if prune_layer is not None else None
        self.selection_seed = int(selection_seed)
        self.visual_grid_side = int(visual_grid_side) if visual_grid_side is not None else None
        self.hevc_n_parallel = int(hevc_n_parallel)
        self.hevc_gop_size = int(
            os.getenv("HEVC_SAMPLED_GOP_SIZE", "0")
            if hevc_gop_size is None else hevc_gop_size
        )
        self.hevc_encode_scope = resolve_hevc_encode_scope(hevc_encode_scope)
        self.hevc_anchor_policy = str(hevc_anchor_policy)
        self.hevc_dir = Path(hevc_dir).expanduser()
        self.hevc_permanent_dir = (
            None if hevc_permanent_dir in (None, "")
            else Path(hevc_permanent_dir).expanduser()
        )
        self.run_pruned, self.run_full = resolve_run_modes(self.run_mode)
        self.k_keep_rate = 1.0 if self.run_full and not self.run_pruned else float(k_keep_rate)
        if self.run_pruned and self.k_keep_rate < 1.0:
            validate_pruning_modes(
                QWEN3_VL_LEGAL_MODES,
                self.family,
                self.prune_stage,
                self.prune_mode,
                self.i_mode,
                self.p_mode,
            )
        self.results_path = Path(results_path).expanduser() if results_path else None
        self.effective_keep_totals = []
        self.i_frame_counts = []
        self.p_keep_totals = []
        self.response_store = None
        self.backend = None
        self.pruning = None

    def _ensure_runtime(self):
        if self.backend is not None:
            return
        self.backend = Qwen3VlHfBackend(
            pretrained=self.pretrained,
            device=self.device,
            load_4bit=self.load_4bit,
            load_8bit=self.load_8bit,
            local_files_only=self.local_files_only,
            visual_grid_side=(self.visual_grid_side or DEFAULT_VISUAL_GRID_SIDE),
        )
        self.pruning = Qwen3VlPruning.create(
            self.backend,
            num_frames=self.num_frames,
            prune_stage=self.prune_stage,
            prune_mode=self.prune_mode,
            i_mode=self.i_mode,
            p_mode=self.p_mode,
            k_keep_rate=self.k_keep_rate,
            run_mode=self.run_mode,
            hevc_n_parallel=self.hevc_n_parallel,
            hevc_gop_size=self.hevc_gop_size,
            hevc_encode_scope=self.hevc_encode_scope,
            hevc_anchor_policy=self.hevc_anchor_policy,
            hevc_dir=self.hevc_dir,
            hevc_permanent_dir=self.hevc_permanent_dir,
            score_reduce=self.score_reduce,
            folding_mode=self.folding_mode,
            fold_block_size=self.fold_block_size,
            fold_pooling=self.fold_pooling,
            prune_layer=self.prune_layer,
            selection_seed=self.selection_seed,
        )

    def loglikelihood(self, _requests):
        raise NotImplementedError(f"The {self.family} bridge supports generation tasks only.")

    def generate_until_multi_round(self, _requests):
        raise NotImplementedError(f"The {self.family} bridge does not support multi-round generation.")

    def generate_until(self, requests: list[Instance]) -> list[str]:
        return run_generate_until(self, requests)


class Qwen3VlThinkingRegisteredModel(Qwen3VlRegisteredModel):
    model_id = "qwen3_vl_thinking"
    family = model_id
    default_pretrained = "Qwen/Qwen3-VL-8B-Thinking"


# The pruning forward follows Transformers' Qwen3VLVisionModel.forward
# (Apache-2.0); see NOTICE.
def install_qwen3_vl_encoder_pruning(vision_encoder, context):
    """Install one plan-aware forward on a ``Qwen3VLVisionModel``."""

    vision_encoder._token_pruner_context = context
    if hasattr(vision_encoder, "_token_pruner_original_forward"):
        return

    vision_encoder._token_pruner_original_forward = vision_encoder.forward

    def forward_with_pruning(self, hidden_states, grid_thw, **kwargs):
        pruning_context = self._token_pruner_context
        if not pruning_context.enabled:
            return self._token_pruner_original_forward(
                hidden_states, grid_thw, **kwargs
            )

        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = (
            position_ids.unsqueeze(-1).float() * self.rotary_pos_emb.inv_freq.float()
        ).flatten(1)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Pinned frame geometry gives G equal segments, so this reshapes to [B, G, N, D].
        batch_size = int(grid_thw.shape[0])
        num_groups = (int(cu_seqlens.numel()) - 1) // batch_size
        tokens_per_group = seq_len // (batch_size * num_groups)
        merge_unit = int(self.spatial_merge_unit)
        plan = pruning_context.plan

        def reduce():
            return prune_flat_visual_tokens(
                hidden_states,
                position_embeddings,
                batch_size=batch_size,
                num_groups=num_groups,
                tokens_per_group=tokens_per_group,
                merge_unit=merge_unit,
                plan=plan,
                expected_group_cells=pruning_context.expected_group_cells,
            )

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            if plan.layer == layer_num:
                (
                    hidden_states,
                    position_embeddings,
                    cu_seqlens,
                    kept_cells,
                ) = pruning_context.record_token_reduce(reduce, hidden_states)
                pruning_context.kept_cells = kept_cells
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ]
                deepstack_feature_lists.append(merger(hidden_states))

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
            deepstack_features=deepstack_feature_lists,
        )

    vision_encoder.forward = types.MethodType(forward_with_pruning, vision_encoder)


def install_lm_shims(model, state):
    """Patch the two language-model entry points pruning invalidates."""

    inner = model.model
    visual = inner.visual
    original_rope = inner.get_rope_index
    original_features = getattr(inner, "get_video_features", None)
    shadowed = {
        name: name in inner.__dict__
        for name in ("get_rope_index", "get_video_features")
    }

    def get_video_features(pixel_values_videos, video_grid_thw=None, **kwargs):
        output = visual(
            pixel_values_videos.type(visual.dtype), grid_thw=video_grid_thw
        )
        output.pooler_output = (output.pooler_output,)
        return output

    def get_rope_index(input_ids, mm_token_type_ids, image_grid_thw=None,
                       video_grid_thw=None, attention_mask=None, **kwargs):
        subset = state.get("rope")
        if subset is None:
            return original_rope(
                input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, attention_mask=attention_mask)
        full_ids, full_mask, full_types, keep_masks = subset
        full_positions, _ = original_rope(
            full_ids, full_types, image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw, attention_mask=full_mask)
        position_ids = torch.zeros(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        deltas = []
        for row in range(input_ids.shape[0]):
            valid = full_positions[..., row, full_mask[row] == 1]
            kept = valid[:, keep_masks[row].to(valid.device)].to(position_ids.device)
            position_ids[..., row, attention_mask[row] == 1] = kept
            deltas.append(int(kept.max()) + 1 - kept.shape[-1])
        return position_ids, torch.tensor(
            deltas, device=input_ids.device).unsqueeze(1)

    inner.get_video_features = get_video_features
    inner.get_rope_index = get_rope_index

    def restore():
        if shadowed["get_rope_index"]:
            inner.get_rope_index = original_rope
        else:
            inner.__dict__.pop("get_rope_index", None)
        if shadowed["get_video_features"]:
            inner.get_video_features = original_features
        else:
            inner.__dict__.pop("get_video_features", None)
        state.pop("rope", None)

    return restore


def shrink_visual_placeholders(model, inputs, group_cells, *, pad_token_id,
                               padding_side="left"):
    """Cut placeholders to survivors; returns ``(inputs, rope_state)`` with pre-shrink ids."""

    full_ids = inputs["input_ids"]
    full_mask = inputs.get("attention_mask")
    if full_mask is None:
        full_mask = torch.ones_like(full_ids)
    full_types = inputs["mm_token_type_ids"]
    inputs, keep_masks = resize_video_placeholders(
        dict(inputs), model.config.video_token_id, group_cells,
        pad_token_id=pad_token_id, padding_side=padding_side,
    )
    return inputs, (full_ids, full_mask, full_types, keep_masks)
