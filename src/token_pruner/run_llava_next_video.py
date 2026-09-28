"""LLaVA-NeXT-Video backend: a CLIP tower pooled inside disjoint cells."""

from pathlib import Path
import os

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from transformers import (
    BitsAndBytesConfig,
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
)
import torch

from .cells import gather_cells, gather_kept_cells, predict_cells, cell_view_cls, split_group_features, join_group_features, reduce_cells, lay_down_cells, grid_side_from_tokens
from .generation import run_generate_until
from .hevc import resolve_hevc_encode_scope
from .layer_boundary import EncoderPruningContext, install_group_carrier
from .selection import BASELINE_MODES, LayerPruningPlan, build_pruning_config, normalize_pruning_scope, resolve_pruning_layer, resolve_visual_budget, validate_pruning_modes
from .signals import SignalOptions, build_signal_provider
from .task_io import (
    collect_batch_timing as _collect_batch_timing,
    current_keep_budgets as _current_keep_budgets,
    decode_tokens as _decode_tokens,
    prepare_batch as _prepare_batch,
    prepare_sample as _prepare_sample,
    _begin_generation as shared_begin_generation,
)
from .task_io import (
    GenerationStepObserver,
    STAGE_TIMER_NAMES,
    StageTimer,
    as_bool,
    greedy_decoding_enabled,
    inference_dtype,
    inference_dtype_name,
    max_new_tokens_override,
    normalize_until,
    resolve_pretrained_source,
    resolve_run_modes,
    rewrite_placeholder_runs,
    stop_strings_enabled,
)
from .tokens import SignalSource


class LlavaNextVideoBackend:
    """Load and run LLaVA-NeXT-Video without an evaluation framework."""

    family = "llava_next_video"
    vision_backend = "hf_clip"
    video_placeholder = "<video>\n"
    reasoning_close_tag = None
    default_max_new_tokens = 128
    needs_video_fps = False

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

    def prepare_generation_kwargs(self, gen_kwargs):
        generation_kwargs = dict(gen_kwargs or {})
        stop_sequences = normalize_until(generation_kwargs.pop("until", None))
        generation_kwargs.setdefault("max_new_tokens", self.default_max_new_tokens)
        override = max_new_tokens_override()
        if override is not None:
            generation_kwargs["max_new_tokens"] = override
        if greedy_decoding_enabled():
            generation_kwargs["do_sample"] = False
        return generation_kwargs, stop_sequences

    @classmethod
    def _prompt_from_messages(cls, messages):
        """Build the USER/ASSISTANT conversation prompt for one sample."""
        prompt_parts = []
        video_inserted = False
        for message in messages:
            text = "\n".join(
                content["text"]
                for content in message["content"]
                if content["type"] == "text"
            )
            if message["role"] == "system":
                if text:
                    prompt_parts.append(f"SYSTEM: {text}")
            elif message["role"] == "user":
                media_prefix = ""
                if not video_inserted:
                    media_prefix = cls.video_placeholder
                    video_inserted = True
                prompt_parts.append(f"USER: {media_prefix}{text}")
            elif message["role"] == "assistant" and text:
                prompt_parts.append(f"ASSISTANT: {text}")
        prompt_parts.append("ASSISTANT:")
        return " ".join(prompt_parts)

    def prepare_sample(self, messages, media, num_frames):
        return _prepare_sample(self, messages, media, num_frames)

    def collect_batch_timing(self, output, inputs):
        return _collect_batch_timing(self, output, inputs)

    def decode_tokens(self, output, inputs, stop_sequences):
        return _decode_tokens(self, output, inputs, stop_sequences)

    def last_untrimmed_texts(self):
        """Return the last batch before stop-sequence trimming."""

        return list(getattr(self, "_untrimmed_texts", []))

    def __init__(self, *, pretrained, device, load_4bit=False, load_8bit=False,
                 local_files_only=False):
        self.device = torch.device(device)
        load_4bit = as_bool(load_4bit)
        load_8bit = as_bool(load_8bit)
        self.source, local_files_only = resolve_pretrained_source(
            pretrained, local_files_only=local_files_only)
        dtype = inference_dtype(self.device)
        self.model_dtype = dtype
        load_kwargs = {
            "dtype": dtype,
            "local_files_only": local_files_only,
        }
        if load_4bit or load_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_4bit, load_in_8bit=load_8bit,
                bnb_4bit_compute_dtype=dtype)
            load_kwargs["device_map"] = {"": str(self.device)}

        self.processor = LlavaNextVideoProcessor.from_pretrained(
            self.source, local_files_only=local_files_only)
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.source, **load_kwargs)
        if not load_4bit and not load_8bit:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.quantization = "4bit" if load_4bit else "8bit" if load_8bit else None
        self._shim_state = {}
        self._restore_shim = None

        inner = self.model.model
        self._install_stage_timers((
            (inner.vision_tower, "vision_encoder"),
            (inner.multi_modal_projector, "projector"),
            (inner.language_model, "language_model"),
        ))

    def cell_side(self):
        """Side of one pooled cell, in patches."""

        return int(self.model.config.spatial_pool_stride)

    def grid_side(self):
        vision = self.model.config.vision_config
        image, patch = int(vision.image_size), int(vision.patch_size)
        return image // patch

    def token_geometry(self):
        """Image and cell size, in pixels, of the pooled cell grid the signal selects on."""

        cell = self.cell_side()
        cell_size = int(self.model.config.vision_config.patch_size) * cell
        return (self.grid_side() // cell) * cell_size, cell_size

    def vision_tower(self):
        return self.model.model.vision_tower

    def install_pruning(self, context):
        self._context = context
        self._pruning_cleanup = install_pruning(
            self.vision_tower(), context, cell_side=self.cell_side())
        self._restore_shim = install_feature_shim(
            self.model, self._shim_state)

    def adapt_lm_inputs(self, inputs, session, effective_keep_total):
        """Arm the pooling shim and shrink the prompt to the surviving cells."""

        pruned_inputs = self.clone_inputs(inputs)
        if not session.pruning_active:
            return pruned_inputs
        groups = int(session.context.num_groups)
        videos = inputs["pixel_values_videos"]
        kept = predict_cells(
            session.context.plan, batch_size=videos.shape[0],
            num_groups=groups, grid_side=self.grid_side(),
            cell_side=self.cell_side(), device=videos.device,
        )
        session.context.expected_group_cells = kept
        self._shim_state["cells"] = kept
        self._shim_state["grid_side"] = self.grid_side()
        self._shim_state["num_groups"] = groups
        visual_token_counts = [
            sum(len(cells) for cells in row) for row in kept
        ]
        return self.resize_pruned_inputs(pruned_inputs, visual_token_counts)

    def resize_pruned_inputs(self, inputs, visual_token_counts):
        if isinstance(visual_token_counts, (list, tuple)):
            run_lengths = [[int(count)] for count in visual_token_counts]
        else:
            run_lengths = [int(visual_token_counts)]
        return rewrite_placeholder_runs(
            inputs, self.model.config.video_token_index, run_lengths,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            padding_side=self.processor.tokenizer.padding_side)

    def _generation_config(self):
        return self.model.generation_config

    def _decode_ids(self, token_ids):
        return self.processor.decode(token_ids, skip_special_tokens=True)

    def prepare_batch_inputs(self, samples):
        return self.processor(
            text=[sample.prompt for sample in samples],
            videos=[sample.sampled_frames for sample in samples],
            padding=True, return_tensors="pt",
        ).to(self.device, self.model.dtype)


NEXT_MODE_KEYS = (
    ("global", "global_folder", "global_folder"),
    ("shared", "shared_hevc", "shared_hevc"),
    ("shared", "shared_folding", "shared_folding"),
    ("global", "preserve", "hevc"),
    ("global", "preserve", "folder"),
    ("global", "folder", "hevc"),
    ("local", "preserve", "hevc"),
    ("local", "preserve", "folder"),
    ("local", "folder", "hevc"),
    ("local", "folder", "folder"),
    ("global", "random", "random"),
    ("global", "uniform", "uniform"),
    ("local", "random", "random"),
    ("local", "uniform", "uniform"),
)
LLAVA_NEXT_VIDEO_LEGAL_MODES = frozenset(
    (stage, *mode) for stage in ("input", "last") for mode in NEXT_MODE_KEYS
)


class LlavaNextVideoPruning:
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
            "llava_next_video", prune_stage, prune_mode
        ).value
        validate_pruning_modes(
            LLAVA_NEXT_VIDEO_LEGAL_MODES,
            "llava_next_video",
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

        tower = backend.model.model.vision_tower
        image_size, patch_size = backend.token_geometry()
        budget = resolve_visual_budget(
            image_size=image_size,
            patch_size=patch_size,
            num_partitions=num_frames,
            k_keep_rate=k_keep_rate,
            run_mode=run_mode,
            run_pruned=run_pruned,
            p_mode=p_mode,
        )
        pruning_config = build_pruning_config(
            family="llava_next_video",
            stage=prune_stage,
            scope=prune_mode,
            i_mode=i_mode,
            p_mode=p_mode,
            legal_modes=LLAVA_NEXT_VIDEO_LEGAL_MODES,
            budget=budget,
            patch_width=image_size // patch_size,
            uniform_partitions=False,
            min_per_partition=1,
            folding_mode=folding_mode,
            fold_block_size=fold_block_size,
            fold_pooling=fold_pooling,
        )

        layers = tuple(tower.encoder.layers)
        feature_layers = backend.model.config.vision_feature_layer
        if isinstance(feature_layers, int):
            feature_layers = [feature_layers]
        boundaries = [int(layer) % (len(layers) + 1) for layer in feature_layers]
        resolved_layer = resolve_pruning_layer(
            prune_layer,
            stage=prune_stage,
            num_layers=len(layers),
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
            num_groups=num_frames,
            plan=plan,
        )
        if budget.pruning_active:
            backend.install_pruning(context)

        provider = None
        if budget.pruning_active and pruning_config.signal_source != SignalSource.NONE:
            provider = build_signal_provider(
                config=pruning_config,
                patch_size=patch_size,
                num_frames=num_frames,
                image_size=image_size,
                score_reduce=score_reduce,
                selection_seed=selection_seed,
                selector_id="patch",
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


class LlavaNextVideoRegisteredModel(lmms):
    is_simple = False
    model_id = "llava_next_video"
    family = model_id
    default_pretrained = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    default_num_frames = 8
    default_prune_mode = "local"
    default_i_mode = "preserve"
    default_p_mode = "hevc"
    selector_id = "patch"
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
                LLAVA_NEXT_VIDEO_LEGAL_MODES,
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
        self.backend = LlavaNextVideoBackend(
            pretrained=self.pretrained,
            device=self.device,
            load_4bit=self.load_4bit,
            load_8bit=self.load_8bit,
            local_files_only=self.local_files_only,
        )
        self.pruning = LlavaNextVideoPruning.create(
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


#: Pool modes whose per-cell answer is defined without the surrounding grid.
CELL_LOCAL_MODES = ("average", "max")


def pool_cells(tokens, pooler, *, grid_side, cells=None, kept=False):
    """Pool full or already-pruned cells like the upstream pooler."""

    cell_side = int(
        pooler.pool.kernel_size
        if isinstance(pooler.pool.kernel_size, int)
        else pooler.pool.kernel_size[0]
    )
    gather = gather_kept_cells if kept else gather_cells
    members = gather(tokens, grid_side=grid_side, cell_side=cell_side, cells=cells)
    kind = type(pooler.pool).__name__
    if kind == "AvgPool2d":
        return members.mean(dim=2)
    if kind == "MaxPool2d":
        return members.amax(dim=2)
    raise RuntimeError(
        f"unsupported pooling {kind}; expected one of {CELL_LOCAL_MODES}"
    )


def tower_encoder(tower):
    """Return the tower encoder loop."""

    return tower.encoder


def tower_blocks(tower):
    """Return the CLIP blocks hooked by the pruning plan."""

    return tuple(tower_encoder(tower).layers)


def install_pruning(tower, context, *, cell_side):
    """Prune the CLIP tower whole-cell at ``context.plan.layer``."""

    encoder = tower_encoder(tower)
    num_groups = int(context.num_groups)
    cell_side = int(cell_side)

    def reduce(hidden_states, plan):
        grid = grid_side_from_tokens(hidden_states.shape[1] - 1)

        def run():
            reduced = reduce_cells(
                hidden_states[:, 1:],
                plan,
                num_groups=num_groups,
                grid_side=grid,
                cell_side=cell_side,
                expected=context.expected_group_cells,
                cls_tokens=cell_view_cls(
                    hidden_states[:, :1], num_groups=num_groups, cell_side=cell_side
                ),
            )
            context.kept_cells = [[slots for _cells, slots in row] for row in reduced]
            return lay_down_cells(
                hidden_states,
                reduced,
                num_groups=num_groups,
                grid_side=grid,
                cell_side=cell_side,
                has_cls=True,
            )

        return context.record_token_reduce(run, hidden_states)

    def reset():
        context.kept_cells = None

    return install_group_carrier(
        tower_blocks(tower), encoder, context, num_groups, reduce, reset=reset, output_module=tower
    )


def install_feature_shim(model, state):
    """Redo ``get_video_features``'s pooling from the kept cell list."""

    inner = getattr(model, "model", model)
    original = inner.get_video_features
    shadowed = "get_video_features" in inner.__dict__

    def get_video_features(
        pixel_values,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        **kwargs,
    ):
        cells = state.get("cells")
        if cells is None:
            return original(
                pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                **kwargs,
            )
        layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else inner.config.vision_feature_layer
        )
        batch, frames, channels, height, width = pixel_values.shape
        flat = pixel_values.reshape(batch * frames, channels, height, width)
        outputs = inner.vision_tower(flat, output_hidden_states=True)
        if isinstance(layer, int):
            features = outputs.hidden_states[layer]
        else:
            features = torch.cat(
                [outputs.hidden_states[index] for index in layer], dim=-1
            )
        grid_side = int(state["grid_side"])
        groups = int(state["num_groups"])
        cell = int(
            inner.vision_resampler.pool.kernel_size
            if isinstance(inner.vision_resampler.pool.kernel_size, int)
            else inner.vision_resampler.pool.kernel_size[0]
        )
        rows = split_group_features(
            features, cells, num_groups=groups, cell_side=cell, has_cls=True
        )
        pooled = join_group_features(
            [
                [
                    pool_cells(
                        part[:, 1:],
                        inner.vision_resampler,
                        grid_side=grid_side,
                        cells=cells[sample][group],
                        kept=True,
                    )
                    for group, part in enumerate(row)
                ]
                for sample, row in enumerate(rows)
            ]
        )
        projected = inner.multi_modal_projector(pooled)
        if projected.shape[0] == batch * frames:
            outputs.pooler_output = torch.split(projected, frames, dim=0)
            return outputs
        if projected.shape[0] == batch:
            outputs.pooler_output = tuple(
                projected[sample : sample + 1] for sample in range(batch)
            )
            return outputs
        raise RuntimeError(
            "projected video features have an unexpected leading dimension: "
            f"shape={tuple(projected.shape)}, B={batch}, frames={frames}."
        )

    inner.get_video_features = get_video_features

    def restore():
        if shadowed:
            inner.get_video_features = original
        else:
            inner.__dict__.pop("get_video_features", None)
        state.pop("cells", None)

    return restore


