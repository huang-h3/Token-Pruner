"""LLaVA-OneVision-2 video generation and vision-token pruning."""

from dataclasses import dataclass, field
from pathlib import Path
import types
import os

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from transformers import BitsAndBytesConfig, AutoModelForImageTextToText, AutoProcessor
import torch

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
from .signals import SignalOptions, build_signal_provider
from .task_io import (
    collect_batch_timing as _collect_batch_timing,
    current_keep_budgets as _current_keep_budgets,
    decode_tokens as _decode_tokens,
    prepare_batch as _prepare_batch,
    prepare_sample as _prepare_sample,
    _begin_generation as shared_begin_generation,
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
from .packed_cells import kept_cell_indices, prune_flat_visual_tokens

MERGE_UNIT = 4


class LlavaOnevision2Backend:
    """Load the checkpoint through its official remote-code entry points."""

    family = "llava_onevision2"
    reasoning_close_tag = None
    needs_video_fps = False
    vision_backend = "onevision2"
    default_max_new_tokens = 2048

    def __init__(
        self,
        *,
        pretrained,
        device,
        load_4bit=False,
        load_8bit=False,
        local_files_only=False,
    ):
        self.device = torch.device(device)
        load_4bit = as_bool(load_4bit)
        load_8bit = as_bool(load_8bit)
        assert not (load_4bit and load_8bit), "choose one quantization mode"
        self.source, local_files_only = resolve_pretrained_source(
            pretrained, local_files_only=local_files_only
        )
        dtype = inference_dtype(self.device)
        self.model_dtype = dtype
        load_kwargs = {
            "dtype": dtype,
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
        if load_4bit or load_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                load_in_8bit=load_8bit,
                bnb_4bit_compute_dtype=dtype,
            )
            load_kwargs["device_map"] = {"": str(self.device)}
        self.processor = AutoProcessor.from_pretrained(
            self.source, local_files_only=local_files_only, trust_remote_code=True
        )
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.source, **load_kwargs
        )
        if not load_4bit and (not load_8bit):
            self.model = self.model.to(self.device)
        self.model.eval()
        self.quantization = "4bit" if load_4bit else "8bit" if load_8bit else None
        self._adapter = None
        self._pruning_cleanup = None
        self._feature_cleanup = None
        tower = self.vision_tower()
        self._install_stage_timers(
            (
                (tower, "vision_encoder"),
                (tower.merger, "projector"),
                (self.model.model.language_model, "language_model"),
            )
        )

    def vision_tower(self):
        return self.model.model.visual

    def encoder_layers(self):
        return self.vision_tower().encoder.layers

    def token_geometry(self):
        config = self.vision_tower().config
        if int(config.spatial_merge_size) != 2:
            raise ValueError("OneVision-2 pruning requires 2x2 merger cells")
        cell = int(config.patch_size) * int(config.spatial_merge_size)
        return (int(config.image_size), cell)

    def feature_boundaries(self, num_layers):
        return ([int(num_layers)], "encoder output")

    def stage_layer(self, prune_layer, *, stage, num_layers, boundaries):
        index = (
            (0 if stage == "input" else num_layers)
            if prune_layer is None
            else int(prune_layer)
        )
        if index < 0:
            index += num_layers + 1
        if not 0 <= index <= num_layers:
            raise ValueError("prune_layer out of range")
        return index

    def install_pruning(self, context):
        if self._pruning_cleanup is not None:
            self._pruning_cleanup()
        if self._feature_cleanup is not None:
            self._feature_cleanup()
        self._adapter = LlavaOnevision2TowerAdapter()
        self._pruning_cleanup = self._adapter.install(self.vision_tower(), context)
        self._feature_cleanup = install_feature_shim(
            self.model.model, context, self._adapter.state
        )

    def remove_pruning(self):
        if self._pruning_cleanup is not None:
            self._pruning_cleanup()
        if self._feature_cleanup is not None:
            self._feature_cleanup()
        self._pruning_cleanup = None
        self._feature_cleanup = None

    def adapt_lm_inputs(self, inputs, session, effective_keep_total):
        pruned = self.clone_inputs(inputs)
        if not session.pruning_active:
            return pruned
        batch = int(pruned["input_ids"].shape[0])
        cells = kept_cell_indices(
            session.context.plan,
            batch_size=batch,
            num_groups=int(session.context.num_groups),
            cells_per_group=int(session.total_patches),
            device=self.device,
        )
        budgets = (
            effective_keep_total
            if isinstance(effective_keep_total, (list, tuple))
            else [effective_keep_total] * batch
        )
        if len(budgets) != batch or any(
            sum(group.numel() for group in row) != int(budget)
            for row, budget in zip(cells, budgets)
        ):
            raise ValueError("placeholder budget diverged")
        session.context.expected_group_cells = cells
        lengths = [[int(group.numel()) for group in row] for row in cells]
        return rewrite_placeholder_runs(
            pruned,
            self.model.config.image_token_id,
            lengths,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            padding_side=self.processor.tokenizer.padding_side,
            allow_expand=False,
        )

    def _generation_config(self):
        return self.model.generation_config

    def _decode_ids(self, token_ids):
        return self.processor.decode(token_ids, skip_special_tokens=True)

    def _prompt_from_messages(self, messages):
        rendered = []
        video_inserted = False
        for message in messages:
            content = []
            if message.get("role") == "user" and (not video_inserted):
                content.append({"type": "video"})
                video_inserted = True
            content.extend(
                (
                    {"type": "text", "text": item["text"]}
                    for item in message.get("content", [])
                    if item.get("type") == "text" and item.get("text")
                )
            )
            if content:
                rendered.append({"role": message["role"], "content": content})
        return self.processor.apply_chat_template(
            rendered, tokenize=False, add_generation_prompt=True
        )

    def _pin_frames(self, frames):
        size = int(self.vision_tower().config.image_size)
        tensor = torch.as_tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
        return tensor.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()

    def prepare_batch_inputs(self, samples):
        inputs = self.processor(
            text=[sample.prompt for sample in samples],
            videos=[
                list(self._pin_frames(sample.sampled_frames)) for sample in samples
            ],
            padding=True,
            video_backend="frames",
            return_tensors="pt",
        )
        return inputs.to(self.device, self.model_dtype)

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
            **inputs, **generation_kwargs, stopping_criteria=stopping
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
        return (generation_kwargs, stop_sequences)

    def prepare_sample(self, messages, media, num_frames):
        return _prepare_sample(self, messages, media, num_frames)

    def collect_batch_timing(self, output, inputs):
        return _collect_batch_timing(self, output, inputs)

    def decode_tokens(self, output, inputs, stop_sequences):
        return _decode_tokens(self, output, inputs, stop_sequences)

    def last_untrimmed_texts(self):
        """Return the last batch before stop-sequence trimming."""
        return list(getattr(self, "_untrimmed_texts", []))


LLAVA_ONEVISION2_MODE_KEYS = (
    ("local", "preserve", "hevc"),
    ("global", "preserve", "hevc"),
    ("shared", "shared_hevc", "shared_hevc"),
    ("local", "random", "random"),
    ("local", "uniform", "uniform"),
    ("global", "random", "random"),
    ("global", "uniform", "uniform"),
)
LLAVA_ONEVISION2_LEGAL_MODES = frozenset(
    (stage, *mode) for stage in ("input", "last") for mode in LLAVA_ONEVISION2_MODE_KEYS
)


class LlavaOnevision2Pruning:
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
            "llava_onevision2", prune_stage, prune_mode
        ).value
        validate_pruning_modes(
            LLAVA_ONEVISION2_LEGAL_MODES,
            "llava_onevision2",
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
            family="llava_onevision2",
            stage=prune_stage,
            scope=prune_mode,
            i_mode=i_mode,
            p_mode=p_mode,
            legal_modes=LLAVA_ONEVISION2_LEGAL_MODES,
            budget=budget,
            patch_width=image_size // patch_size,
            uniform_partitions=False,
            min_per_partition=1,
            folding_mode=folding_mode,
            fold_block_size=fold_block_size,
            fold_pooling=fold_pooling,
        )

        layers = tuple(backend.encoder_layers())
        boundaries, _ = backend.feature_boundaries(len(layers))
        resolved_layer = backend.stage_layer(
            prune_layer,
            stage=prune_stage,
            num_layers=len(layers),
            boundaries=boundaries,
        )
        if budget.pruning_active and resolved_layer > min(boundaries):
            raise ValueError(
                "prune_layer is after the consumed vision feature boundary"
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
                None
                if hevc_permanent_dir is None
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


class LlavaOnevision2RegisteredModel(lmms):
    is_simple = False
    model_id = "llava_onevision2"
    family = model_id
    default_pretrained = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
    default_num_frames = 32
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
        self.num_frames = int(
            self.default_num_frames if num_frames is None else num_frames
        )
        self.run_mode = str(run_mode)
        self.score_reduce = str(score_reduce)
        self.folding_mode = str(folding_mode)
        self.fold_block_size = int(fold_block_size)
        self.fold_pooling = str(fold_pooling)
        self.pretrained = str(
            self.default_pretrained if pretrained is None else pretrained
        )
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
            if hevc_gop_size is None
            else hevc_gop_size
        )
        self.hevc_encode_scope = resolve_hevc_encode_scope(hevc_encode_scope)
        self.hevc_anchor_policy = str(hevc_anchor_policy)
        self.hevc_dir = Path(hevc_dir).expanduser()
        self.hevc_permanent_dir = (
            None
            if hevc_permanent_dir in (None, "")
            else Path(hevc_permanent_dir).expanduser()
        )
        self.run_pruned, self.run_full = resolve_run_modes(self.run_mode)
        self.k_keep_rate = (
            1.0 if self.run_full and not self.run_pruned else float(k_keep_rate)
        )
        if self.run_pruned and self.k_keep_rate < 1.0:
            validate_pruning_modes(
                LLAVA_ONEVISION2_LEGAL_MODES,
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
        self.backend = LlavaOnevision2Backend(
            pretrained=self.pretrained,
            device=self.device,
            load_4bit=self.load_4bit,
            load_8bit=self.load_8bit,
            local_files_only=self.local_files_only,
        )
        self.pruning = LlavaOnevision2Pruning.create(
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
        raise NotImplementedError(
            f"The {self.family} bridge supports generation tasks only."
        )

    def generate_until_multi_round(self, _requests):
        raise NotImplementedError(
            f"The {self.family} bridge does not support multi-round generation."
        )

    def generate_until(self, requests: list[Instance]) -> list[str]:
        return run_generate_until(self, requests)


def _cumulative_lengths(lengths, device):
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return cu


def frame_patch_counts(grid_thw):
    """Patch count for each logical frame."""

    return [
        int(h) * int(w)
        for t, h, w in torch.as_tensor(grid_thw).tolist()
        for _ in range(int(t))
    ]


def window_cu_seqlens(grid_thw, kept_cells, window_size, device):
    """Attention segments after frame-wise cell selection."""

    lengths = []
    frames = iter(
        int(cells.numel()) * MERGE_UNIT for row in kept_cells for cells in row
    )
    for t, _h, _w in torch.as_tensor(grid_thw).tolist():
        row = [next(frames) for _ in range(int(t))]
        width = int(window_size)
        lengths.extend(
            sum(row[start : start + width]) for start in range(0, len(row), width)
        )
    assert next(frames, None) is None, "grid and frame selections diverged"
    return _cumulative_lengths(lengths, device)


def row_cell_counts(grid_thw, kept_cells):
    """Merged feature lengths matching grid rows."""

    frames = iter(int(cells.numel()) for row in kept_cells for cells in row)
    counts = [
        sum(next(frames) for _ in range(int(t)))
        for t, _h, _w in torch.as_tensor(grid_thw).tolist()
    ]
    assert next(frames, None) is None, "grid and frame selections diverged"
    return counts


@dataclass
class _ForwardState:
    grid_thw: object | None = None
    patch_positions: object | None = None
    rotary: object | None = None
    cu_seqlens: object | None = None
    row_cells: object | None = None

    def reset(self, grid_thw, patch_positions):
        self.grid_thw = grid_thw
        self.patch_positions = (
            patch_positions.squeeze(0) if patch_positions.ndim == 3 else patch_positions
        )
        self.rotary = None
        self.cu_seqlens = None
        self.row_cells = None


@dataclass
class LlavaOnevision2TowerAdapter:
    """Prune complete merger cells in the packed OneVision sequence."""

    state: _ForwardState = field(default_factory=_ForwardState)

    def install(self, tower, context):
        blocks = tuple(tower.encoder.layers)
        boundary = int(context.plan.layer)
        assert 0 <= boundary <= len(blocks), "vision boundary out of range"
        state = self.state

        def capture(_module, _args, kwargs):
            if context.enabled:
                state.reset(kwargs["grid_thw"], kwargs["patch_positions"])

        def reduce(hidden_states, rotary):
            flat = hidden_states.squeeze(0)
            positions = (rotary.squeeze(0), state.patch_positions)
            groups = int(context.num_groups)
            batch = (
                len(context.expected_group_cells or ())
                or len(frame_patch_counts(state.grid_thw)) // groups
            )
            tokens = flat.shape[0] // (batch * groups)
            patch_counts = frame_patch_counts(state.grid_thw)
            assert (
                tokens * batch * groups == flat.shape[0]
            ), "packed frame geometry changed"
            assert (
                len(set(patch_counts)) == 1 and tokens == patch_counts[0]
            ), "variable frame geometry is unsupported"

            def run():
                return prune_flat_visual_tokens(
                    flat,
                    positions,
                    batch_size=batch,
                    num_groups=groups,
                    tokens_per_group=tokens,
                    merge_unit=MERGE_UNIT,
                    plan=context.plan,
                    expected_group_cells=context.expected_group_cells,
                )

            pruned, selected, _frame_cu, kept = context.record_token_reduce(run, flat)
            state.rotary = selected[0].unsqueeze(0)
            state.patch_positions = selected[1]
            state.cu_seqlens = window_cu_seqlens(
                state.grid_thw,
                kept,
                int(tower.config.frame_windows_size),
                flat.device,
            )
            state.row_cells = row_cell_counts(state.grid_thw, kept)
            context.kept_cells = kept
            return pruned.unsqueeze(0)

        def block_hook(index):
            def hook(_module, args, kwargs):
                if not context.enabled:
                    return None
                state.rotary = (
                    kwargs["rotary_pos_emb"] if state.rotary is None else state.rotary
                )
                hidden = args[0]
                if index == boundary:
                    hidden = reduce(hidden, kwargs["rotary_pos_emb"])
                if state.cu_seqlens is None:
                    return None
                lengths = state.cu_seqlens[1:] - state.cu_seqlens[:-1]
                return (hidden,), {
                    **kwargs,
                    "rotary_pos_emb": state.rotary,
                    "cu_seqlens": state.cu_seqlens,
                    "max_seqlen": int(lengths.max()),
                }

            return hook

        def merger(_module, args, kwargs):
            if not context.enabled:
                return None
            hidden = args[0]
            if boundary == len(blocks):
                hidden = reduce(hidden, state.rotary)
            return (hidden,), {**kwargs, "patch_positions": state.patch_positions}

        handles = [tower.register_forward_pre_hook(capture, with_kwargs=True)]
        handles.extend(
            block.register_forward_pre_hook(block_hook(index), with_kwargs=True)
            for index, block in enumerate(blocks)
        )
        handles.append(tower.merger.register_forward_pre_hook(merger, with_kwargs=True))

        def cleanup():
            for handle in handles:
                handle.remove()

        return cleanup


def install_feature_shim(model, context, state):
    """Split merged features by surviving cells."""

    original = model.get_image_features

    def get_image_features(
        self, pixel_values, image_grid_thw=None, patch_positions=None
    ):
        if not context.enabled:
            return original(
                pixel_values, image_grid_thw, patch_positions=patch_positions
            )
        pixel_values = pixel_values.type(
            self.visual.embeddings.patch_embedding.weight.dtype
        )
        output = self.visual(
            pixel_values, grid_thw=image_grid_thw, patch_positions=patch_positions
        )
        features = output.last_hidden_state.reshape(
            -1, output.last_hidden_state.shape[-1]
        )
        return list(torch.split(features, state.row_cells))

    model.get_image_features = types.MethodType(get_image_features, model)

    def cleanup():
        model.get_image_features = original

    return cleanup
