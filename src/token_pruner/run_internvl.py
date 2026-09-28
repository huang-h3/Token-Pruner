"""InternVL 3.5 video generation and vision-token pruning."""

from pathlib import Path
from types import SimpleNamespace
import os

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .cells import (
    gather_kept_cells,
    predict_cells,
    cell_view_cls,
    split_group_features,
    join_group_features,
    reduce_cells,
    lay_down_cells,
    grid_side_from_tokens,
)
from .generation import run_generate_until
from .hevc import resolve_hevc_encode_scope
from .layer_boundary import EncoderPruningContext, install_group_carrier
from .selection import (
    BASELINE_MODES,
    LayerPruningPlan,
    build_pruning_config,
    normalize_pruning_scope,
    resolve_pruning_layer,
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

# Each pixel-shuffle output joins a complete 2x2 patch cell.
CELL_SIDE = 2

_IMAGE_START = "<img>"
_IMAGE_END = "</img>"
_IMAGE_CONTEXT = "<IMG_CONTEXT>"
_VIDEO_MARKER = "<video>\n"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def internvl_chat_class(source, *, local_files_only):
    config = AutoConfig.from_pretrained(
        source, local_files_only=local_files_only, trust_remote_code=True
    )
    remote = get_class_from_dynamic_module(
        config.auto_map["AutoModel"], source, local_files_only=local_files_only
    )

    def __init__(self, *args, **kwargs):
        remote.__init__(self, *args, **kwargs)
        self.post_init()

    return type(remote.__name__, (remote,), {"__init__": __init__})


class InternVlBackend:
    """Load the official non-thinking InternVL3.5 Instruct checkpoint."""

    family = "internvl"
    reasoning_close_tag = None
    needs_video_fps = False
    default_max_new_tokens = 128
    vision_backend = "internvit_instruct"

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
        self._require_instruct_source(pretrained)
        load_4bit = as_bool(load_4bit)
        load_8bit = as_bool(load_8bit)
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.source,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.processor = SimpleNamespace(
            tokenizer=self.tokenizer, decode=self.tokenizer.decode
        )
        self.model = internvl_chat_class(
            self.source, local_files_only=local_files_only
        ).from_pretrained(self.source, **load_kwargs)
        if not load_4bit and (not load_8bit):
            self.model = self.model.to(self.device)
        self.model.eval()
        self.quantization = "4bit" if load_4bit else "8bit" if load_8bit else None
        if self.model.__class__.__name__ != "InternVLChatModel":
            raise TypeError(
                f"InternVL Instruct requires the official InternVLChatModel, got {self.model.__class__.__name__}."
            )
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(_IMAGE_CONTEXT)
        self.model.img_context_token_id = self.img_context_token_id
        self._shim_state = {}
        self._restore_shim = None
        self._install_stage_timers(
            (
                (self.model.vision_model, "vision_encoder"),
                (self.model.mlp1, "projector"),
                (self.model.language_model, "language_model"),
            )
        )

    @staticmethod
    def _require_instruct_source(pretrained):
        name = Path(str(pretrained).rstrip("/")).name.lower()
        if "instruct" not in name:
            raise ValueError(
                f"The InternVL backend accepts only an Instruct checkpoint; got {pretrained!r}. Use OpenGVLab/InternVL3_5-8B-Instruct."
            )

    def _prompt_from_messages(self, messages):
        """Render the prompt that the checkpoint's ``chat`` builds."""
        template = self.model.conv_template.copy()
        template.system_message = self.model.system_message
        roles = {"user": template.roles[0], "assistant": template.roles[1]}
        video_inserted = False
        for message in messages:
            text = "\n".join(
                item["text"]
                for item in message.get("content", [])
                if item.get("type") == "text" and item.get("text")
            )
            if message.get("role") == "system":
                template.system_message = text
                continue
            if message.get("role") == "user" and not video_inserted:
                text = f"{_VIDEO_MARKER}{text}"
                video_inserted = True
            if text:
                template.append_message(roles[message["role"]], text)
        if not video_inserted:
            raise ValueError("InternVL requires one user turn for the video.")
        template.append_message(template.roles[1], None)
        return template.get_prompt()

    def token_geometry(self):
        """The pruning signal operates on the post-pixel-shuffle cell grid."""
        cell_size = (
            self._square(self.model.config.vision_config.patch_size, "patch_size")
            * CELL_SIDE
        )
        return (self.grid_side() // CELL_SIDE * cell_size, cell_size)

    def vision_tower(self):
        return self.model.vision_model

    def encoder_layers(self):
        return tower_blocks(self.vision_tower())

    def feature_boundaries(self, num_layers):
        layer = self.model.select_layer
        return ([int(layer) % (int(num_layers) + 1)], int(layer))

    def install_pruning(self, context):
        self.validate_pruning_contract()
        self._context = context
        self._shim_state["context"] = context
        self._pruning_cleanup = install_pruning(self.vision_tower(), context)
        self._restore_shim = install_feature_shim(self.model, self._shim_state)

    def remove_pruning(self):
        for teardown in (getattr(self, "_pruning_cleanup", None), self._restore_shim):
            if teardown is not None:
                teardown()
        self._pruning_cleanup = None
        self._restore_shim = None

    @staticmethod
    def _square(value, name):
        if isinstance(value, (list, tuple)):
            if len(value) != 2 or value[0] != value[1]:
                raise ValueError(
                    f"InternVL pruning assumes a square grid; {name}={value!r}."
                )
            value = value[0]
        return int(value)

    def grid_side(self):
        vision = self.model.config.vision_config
        image = self._square(vision.image_size, "image_size")
        patch = self._square(vision.patch_size, "patch_size")
        if patch < 1 or image % patch:
            raise ValueError(
                f"InternVL pruning needs image_size divisible by a positive patch_size, got image_size={image}, patch_size={patch}."
            )
        return image // patch

    def validate_pruning_contract(self):
        ratio = float(self.model.downsample_ratio)
        if ratio != 1 / CELL_SIDE:
            raise ValueError(
                f"pruned InternVL requires downsample_ratio={1 / CELL_SIDE}, got {ratio}."
            )
        grid = self.grid_side()
        if grid % CELL_SIDE:
            raise ValueError(
                f"InternVL's {grid}x{grid} patch grid does not divide into {CELL_SIDE}x{CELL_SIDE} pixel-shuffle cells."
            )
        expected_tokens = (grid // CELL_SIDE) ** 2
        if int(self.model.num_image_token) != expected_tokens:
            raise ValueError(
                f"InternVL's declared image-token count does not match its cell grid: {self.model.num_image_token} != {expected_tokens}."
            )
        if not isinstance(self.model.select_layer, int):
            raise ValueError(
                f"pruned InternVL supports one integer select_layer, got {self.model.select_layer!r}."
            )

    def adapt_lm_inputs(self, inputs, session, effective_keep_total):
        pruned_inputs = self.clone_inputs(inputs)
        if not session.pruning_active:
            return pruned_inputs
        groups = int(session.context.num_groups)
        pixels = inputs["pixel_values"]
        if pixels.ndim != 4 or groups < 1 or pixels.shape[0] % groups:
            raise ValueError(
                f"InternVL pruning expects pixel_values [B*G, C, H, W] with rows divisible by G; shape={tuple(pixels.shape)}, G={groups}."
            )
        kept = predict_cells(
            session.context.plan,
            batch_size=pixels.shape[0] // groups,
            num_groups=groups,
            grid_side=self.grid_side(),
            cell_side=CELL_SIDE,
            device=pixels.device,
        )
        session.context.expected_group_cells = kept
        self._shim_state["cells"] = kept
        self._shim_state["grid_side"] = self.grid_side()
        self._shim_state["num_groups"] = groups
        return shrink_image_placeholders(
            pruned_inputs,
            kept,
            image_token_id=self.img_context_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side=self.tokenizer.padding_side,
        )

    def _generation_config(self):
        return self.model.language_model.generation_config

    def decoding_parameters(self):
        return {"checkpoint_variant": "instruct", "enable_thinking": False}

    def _decode_ids(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _resolve_input_length(self, inputs):
        return 0

    def _expand_video(self, prompt, num_frames):
        if prompt.count(_VIDEO_MARKER) != 1:
            raise ValueError("InternVL prompt must contain exactly one video marker.")
        image = (
            _IMAGE_START + _IMAGE_CONTEXT * int(self.model.num_image_token) + _IMAGE_END
        )
        frames = "".join(
            (f"Frame{index + 1}: {image}\n" for index in range(int(num_frames)))
        )
        return prompt.replace(_VIDEO_MARKER, frames, 1)

    def _preprocess_frames(self, frames):
        side = self.grid_side() * self._square(
            self.model.config.vision_config.patch_size, "patch_size"
        )
        resized = [
            torch.from_numpy(
                np.array(
                    Image.fromarray(frame)
                    .convert("RGB")
                    .resize((side, side), Image.Resampling.BICUBIC),
                    copy=True,
                )
            )
            for frame in frames
        ]
        tensor = torch.stack(resized).permute(0, 3, 1, 2).float()
        tensor = tensor.div_(255.0)
        mean = tensor.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = tensor.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        return tensor.sub_(mean).div_(std)

    def prepare_batch_inputs(self, samples):
        if not samples:
            raise ValueError("samples cannot be empty.")
        num_frames = len(samples[0].frame_indices)
        if any((len(sample.frame_indices) != num_frames for sample in samples)):
            raise ValueError("InternVL batches require one shared frame count.")
        prompts = [self._expand_video(sample.prompt, num_frames) for sample in samples]
        text = self.tokenizer(prompts, padding=True, return_tensors="pt")
        pixel_values = torch.cat(
            [self._preprocess_frames(sample.sampled_frames) for sample in samples],
            dim=0,
        )
        expected = len(samples) * num_frames * int(self.model.num_image_token)
        actual = int((text["input_ids"] == self.img_context_token_id).sum())
        if actual != expected:
            raise ValueError(
                f"InternVL prompt has {actual} image placeholders; expected {expected}."
            )
        return {
            "input_ids": text["input_ids"].to(self.device),
            "attention_mask": text["attention_mask"].to(self.device),
            "pixel_values": pixel_values.to(self.device, self.model_dtype),
        }

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


INTERNVL_MODE_KEYS = (
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
INTERNVL_LEGAL_MODES = frozenset(
    (stage, *mode) for stage in ("input", "last") for mode in INTERNVL_MODE_KEYS
)


class InternVlPruning:
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
        prune_mode = normalize_pruning_scope("internvl", prune_stage, prune_mode).value
        validate_pruning_modes(
            INTERNVL_LEGAL_MODES,
            "internvl",
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
            family="internvl",
            stage=prune_stage,
            scope=prune_mode,
            i_mode=i_mode,
            p_mode=p_mode,
            legal_modes=INTERNVL_LEGAL_MODES,
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
        resolved_layer = resolve_pruning_layer(
            prune_layer, stage=prune_stage, num_layers=len(layers)
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


class InternVlRegisteredModel(lmms):
    is_simple = False
    model_id = "internvl"
    family = model_id
    default_pretrained = "OpenGVLab/InternVL3_5-8B-Instruct"
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
                INTERNVL_LEGAL_MODES,
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
        self.backend = InternVlBackend(
            pretrained=self.pretrained,
            device=self.device,
            load_4bit=self.load_4bit,
            load_8bit=self.load_8bit,
            local_files_only=self.local_files_only,
        )
        self.pruning = InternVlPruning.create(
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


def merge_kept_cells(tokens, *, grid_side, cells):
    """Merge the cells that survived, from a tower output already reduced."""

    members = gather_kept_cells(
        tokens, grid_side=grid_side, cell_side=CELL_SIDE, cells=cells
    )
    return members.reshape(members.shape[0], members.shape[1], -1)


def tower_blocks(tower):
    """Return the InternViT blocks hooked by the pruning plan."""

    return tuple(tower.encoder.layers)


def install_pruning(tower, context):
    """Prune InternViT whole-cell at ``context.plan.layer``; return the teardown."""

    encoder = tower.encoder
    num_groups = int(context.num_groups)

    def reduce(hidden_states, plan):
        grid = grid_side_from_tokens(hidden_states.shape[1] - 1)

        def run():
            reduced = reduce_cells(
                hidden_states[:, 1:],
                plan,
                num_groups=num_groups,
                grid_side=grid,
                cell_side=CELL_SIDE,
                expected=context.expected_group_cells,
                cls_tokens=cell_view_cls(
                    hidden_states[:, :1], num_groups=num_groups, cell_side=CELL_SIDE
                ),
            )
            context.kept_cells = [[slots for _cells, slots in row] for row in reduced]
            return lay_down_cells(
                hidden_states,
                reduced,
                num_groups=num_groups,
                grid_side=grid,
                cell_side=CELL_SIDE,
                has_cls=True,
            )

        return context.record_token_reduce(run, hidden_states)

    def reset():
        context.kept_cells = None

    return install_group_carrier(
        tower_blocks(tower), encoder, context, num_groups, reduce, reset=reset
    )


# Follows InternVLChatModel.extract_feature (MIT); see NOTICE.
def install_feature_shim(model, state):
    original = model.extract_feature
    shadowed = "extract_feature" in model.__dict__

    def extract_feature(pixel_values):
        cells = state.get("cells")
        if cells is None or (
            state.get("context") is not None and not state["context"].enabled
        ):
            return original(pixel_values)

        if model.select_layer == -1:
            features = model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            features = model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[model.select_layer]

        grid_side = int(state["grid_side"])
        groups = int(state["num_groups"])
        rows = split_group_features(
            features,
            cells,
            num_groups=groups,
            cell_side=CELL_SIDE,
            has_cls=True,
        )
        merged = join_group_features(
            [
                [
                    merge_kept_cells(
                        part[:, 1:, :],
                        grid_side=grid_side,
                        cells=cells[sample][group],
                    )
                    for group, part in enumerate(row)
                ]
                for sample, row in enumerate(rows)
            ]
        )
        return model.mlp1(merged)

    model.extract_feature = extract_feature

    def restore():
        if shadowed:
            model.extract_feature = original
        else:
            model.__dict__.pop("extract_feature", None)
        state.pop("cells", None)

    return restore


def shrink_image_placeholders(
    inputs, kept, *, image_token_id, pad_token_id, padding_side="left"
):
    """Resize each prompt run to its surviving cell count."""

    lengths = [[len(cells) for cells in row] for row in kept]
    return rewrite_placeholder_runs(
        inputs,
        image_token_id,
        lengths,
        pad_token_id=pad_token_id,
        padding_side=padding_side,
        allow_expand=False,
    )
