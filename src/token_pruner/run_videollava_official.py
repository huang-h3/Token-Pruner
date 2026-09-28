"""Native Video-LLaVA backend (original non-HF checkpoint, temporal attention)."""

from dataclasses import dataclass
from einops import rearrange
from pathlib import Path
from typing import Optional, Tuple, Union
import os

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from sentencepiece import SentencePieceProcessor
from torch import nn
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIPAttention,
    CLIPMLP,
    CLIPVisionEmbeddings,
)
import torch

from .generation import run_generate_until
from .hevc import resolve_hevc_encode_scope
from .layer_boundary import (
    EncoderPruningContext,
    install_group_carrier,
    prune_frame_sequence,
)
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
)
from .task_io import (
    GenerationStepObserver,
    STAGE_TIMER_NAMES,
    StageTimer,
    as_bool,
    greedy_decoding_enabled,
    inference_dtype,
    PROMPT_STYLE_ENV,
    inference_dtype_name,
    max_new_tokens_override,
    normalize_until,
    placeholder_runs,
    resolve_pretrained_source,
    resolve_run_modes,
    rewrite_placeholder_runs,
    stop_strings_enabled,
)
from .tokens import SignalSource

# CLIP normalization statistics used by the LanguageBind video tower.
_IMAGE_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_IMAGE_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class VideoLlavaOfficialBackend:
    """Run the native Video-LLaVA checkpoint without an evaluation framework."""

    family = "video_llava_official"
    vision_backend = "videobind"
    # One "<image>" per frame; each expands to that frame's tokens.
    NUM_FRAMES = 8
    video_placeholder = "<image>" * NUM_FRAMES + "\n"
    reasoning_close_tag = None
    default_max_new_tokens = 128
    needs_video_fps = False

    def _resolve_input_length(self, inputs):
        """Prompt length used to split prompt from completion."""
        return inputs["input_ids"].shape[1]

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
    def _plain_prompt_from_messages(cls, messages):
        """Build the Video-LLaVA conversation prompt for one sample."""
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

    @classmethod
    def _prompt_from_messages(cls, messages):
        """Build the prompt, optionally using the llava_v1 training template."""

        if official_prompt_style() == "default":
            return cls._plain_prompt_from_messages(messages)
        # llava_v1: system prefix + space-separated sentinels, shared turn builder.
        spaced = " ".join(["<image>"] * cls.NUM_FRAMES) + "\n"
        prompt = cls._plain_prompt_from_messages(messages)
        prompt = prompt.replace(cls.video_placeholder, spaced, 1)
        return f"{LLAVA_V1_SYSTEM_PROMPT} {prompt}"

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
        self.model_dtype = inference_dtype(self.device)
        self.load_4bit = as_bool(load_4bit)
        self.load_8bit = as_bool(load_8bit)

        self.source, local_files_only = resolve_pretrained_source(
            pretrained,
            local_files_only=local_files_only,
        )
        self._checkpoint_dir = ensure_checkpoint_dir(
            self.source,
            local_files_only=local_files_only,
        )
        self.video_tower, self.projector, self.language_model = (
            load_official_checkpoint(
                self._checkpoint_dir,
                device=self.device,
                dtype=self.model_dtype,
                load_4bit=self.load_4bit,
                load_8bit=self.load_8bit,
            )
        )
        self.quantization = (
            "4bit" if self.load_4bit else "8bit" if self.load_8bit else None
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self._checkpoint_dir,
            local_files_only=local_files_only,
        )
        self.sentencepiece = SentencePieceProcessor(
            model_file=str(self._checkpoint_dir / "tokenizer.model")
        )
        self._install_stage_timers(
            (
                (self.video_tower, "vision_encoder"),
                (self.projector, "projector"),
                (self.language_model, "language_model"),
            )
        )

    def vision_tower(self):
        return self.video_tower

    def install_pruning(self, context):
        self._pruning_cleanup = install_pruning(
            self.vision_tower(), context)

    def adapt_lm_inputs(self, inputs, session, effective_keep_total):
        pruned = self.clone_inputs(inputs)
        if session.pruning_active:
            visual_tokens = int(session.context.num_groups) + int(effective_keep_total)
            pruned = self.resize_pruned_inputs(pruned, visual_tokens)
        return pruned

    def _generation_config(self):
        return self.language_model.generation_config

    def _decode_ids(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _slice_generated(self, sequence, input_length):
        # Pure inputs_embeds generation returns only the new tokens, no prefix.
        if sequence.shape[0] > input_length:
            return sequence[input_length:]
        return sequence

    @staticmethod
    def _preprocess_frames(frames):
        """Normalize numpy ``[T,H,W,C]`` uint8 frames for the video tower."""
        tensor = torch.as_tensor(frames, dtype=torch.float32)  # [T, H, W, C]
        tensor = tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
        tensor = torch.nn.functional.interpolate(
            tensor, size=(224, 224), mode="bilinear", align_corners=False
        )
        tensor = tensor / 255.0
        tensor = (tensor - _IMAGE_MEAN.to(tensor.device)) / _IMAGE_STD.to(
            tensor.device
        )
        return tensor.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]

    def _rewrite_frame_runs(self, inputs, per_frame):
        """Set every frame's placeholder run to ``per_frame`` tokens."""

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        run_lengths = []
        for row_index in range(int(input_ids.shape[0])):
            row = input_ids[row_index]
            valid = (
                attention_mask[row_index].bool()
                if attention_mask is not None
                else torch.ones(row.shape[0], dtype=torch.bool, device=row.device)
            )
            runs = placeholder_runs(row[valid], IMAGE_TOKEN_INDEX)
            if len(runs) == 1:
                run_lengths.append([per_frame * self.NUM_FRAMES])
            elif len(runs) == self.NUM_FRAMES:
                run_lengths.append([per_frame] * self.NUM_FRAMES)
            else:
                raise RuntimeError(
                    f"Sample {row_index} has {len(runs)} official video "
                    f"placeholder runs; expected one contiguous run or "
                    f"{self.NUM_FRAMES} frame runs."
                )
        return rewrite_placeholder_runs(
            inputs,
            IMAGE_TOKEN_INDEX,
            run_lengths,
            pad_token_id=self.tokenizer.pad_token_id or 0,
            padding_side="left",
        )

    def prepare_batch_inputs(self, samples):
        """Tokenize with full per-frame sentinel runs and build video tensors."""

        prompts = [sample.prompt for sample in samples]
        batch_ids = [
            tokenizer_image_token(prompt, self.sentencepiece, IMAGE_TOKEN_INDEX)
            for prompt in prompts
        ]
        max_length = max(len(ids) for ids in batch_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full(
            (len(samples), max_length), pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros(
            (len(samples), max_length), dtype=torch.long
        )
        # Rectangle for the [B, L] run rewriter; it re-pads left after stripping.
        for index, ids in enumerate(batch_ids):
            offset = max_length - len(ids)
            input_ids[index, offset:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[index, offset:] = 1

        # One sentinel per frame, however the prompt style spells them out.
        config = self.video_tower.config
        tokens_per_frame = (
            (int(config.image_size) // int(config.patch_size)) ** 2 + 1
        )
        expanded = self._rewrite_frame_runs(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            tokens_per_frame,
        )
        frames_batch = [
            self._preprocess_frames(sample.sampled_frames)
            for sample in samples
        ]
        pixel_values_videos = torch.stack(frames_batch, dim=0).to(
            self.device, self.model_dtype
        )
        return {
            "input_ids": expanded["input_ids"].to(self.device),
            "attention_mask": expanded["attention_mask"].to(self.device),
            "pixel_values_videos": pixel_values_videos,
        }

    def generate_tokens(self, inputs, generation_kwargs, stop_sequences=None):
        generation_kwargs, stopping = self._begin_generation(generation_kwargs)

        video_outputs = self.video_tower(
            inputs["pixel_values_videos"], output_hidden_states=True
        )
        video_embeds = self.projector(video_outputs.hidden_states[-2])
        frames = int(video_embeds.shape[1])
        expected = frames * int(video_embeds.shape[2])

        input_ids = inputs["input_ids"]
        text_embeds = self.language_model.get_input_embeddings()(
            input_ids.clamp(min=0)
        )
        is_video = input_ids == IMAGE_TOKEN_INDEX
        counts = is_video.sum(dim=1).tolist()
        if any(count != expected for count in counts):
            raise RuntimeError(
                f"Placeholder counts {counts} do not match the tower output "
                f"({frames} frames x {video_embeds.shape[2]} tokens)."
            )
        inputs_embeds = text_embeds.masked_scatter(
            is_video.unsqueeze(-1).expand_as(text_embeds),
            video_embeds.flatten(1, 2).to(text_embeds.dtype),
        )

        if stop_sequences and stop_strings_enabled():
            generation_kwargs["stop_strings"] = list(stop_sequences)
            generation_kwargs["tokenizer"] = self.tokenizer
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            **generation_kwargs,
            stopping_criteria=stopping,
        )

    def resize_pruned_inputs(self, inputs, visual_token_count):
        """Shrink every official frame placeholder run to one equal width."""

        per_frame, remainder = divmod(int(visual_token_count), self.NUM_FRAMES)
        if remainder:
            raise RuntimeError(
                f"{visual_token_count} visual tokens do not split evenly over "
                f"{self.NUM_FRAMES} frames."
            )
        return self._rewrite_frame_runs(inputs, per_frame)


OFFICIAL_MODE_KEYS = (
    ("global", "global_folder", "global_folder"),
    ("shared", "shared_hevc", "shared_hevc"),
    ("shared", "shared_folding", "shared_folding"),
    ("local", "folder", "hevc"),
    ("local", "folder", "folder"),
    ("local", "random", "random"),
    ("local", "uniform", "uniform"),
)
VIDEO_LLAVA_OFFICIAL_LEGAL_MODES = frozenset(
    (stage, *mode) for stage in ("input", "last") for mode in OFFICIAL_MODE_KEYS
)


class VideoLlavaOfficialPruning:
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
            "video_llava_official", prune_stage, prune_mode
        ).value
        validate_pruning_modes(
            VIDEO_LLAVA_OFFICIAL_LEGAL_MODES,
            "video_llava_official",
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

        tower = backend.video_tower
        image_size = int(tower.config.image_size)
        patch_size = int(tower.config.patch_size)
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
            family="video_llava_official",
            stage=prune_stage,
            scope=prune_mode,
            i_mode=i_mode,
            p_mode=p_mode,
            legal_modes=VIDEO_LLAVA_OFFICIAL_LEGAL_MODES,
            budget=budget,
            patch_width=image_size // patch_size,
            uniform_partitions=True,
            min_per_partition=0,
            folding_mode=folding_mode,
            fold_block_size=fold_block_size,
            fold_pooling=fold_pooling,
        )

        layers = tuple(tower.encoder.layers)
        feature_layers = -2
        boundaries = [feature_layers % (len(layers) + 1)]
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


class VideoLlavaOfficialRegisteredModel(lmms):
    is_simple = False
    model_id = "video_llava_official"
    family = model_id
    default_pretrained = "LanguageBind/Video-LLaVA-7B"
    default_num_frames = 8
    default_prune_mode = "local"
    default_i_mode = "folder"
    default_p_mode = "hevc"
    selector_id = "patch"
    uniform_partitions = True
    min_cells_per_partition = 0

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
                VIDEO_LLAVA_OFFICIAL_LEGAL_MODES,
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
        self.backend = VideoLlavaOfficialBackend(
            pretrained=self.pretrained,
            device=self.device,
            load_4bit=self.load_4bit,
            load_8bit=self.load_8bit,
            local_files_only=self.local_files_only,
        )
        self.pruning = VideoLlavaOfficialPruning.create(
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


def tower_encoder(tower):
    """The module whose loop drives the blocks; boundary zero prunes here."""

    return tower.encoder


def tower_blocks(tower):
    """The VideoBind encoder blocks, in order."""

    return tuple(tower_encoder(tower).layers)


def install_pruning(tower, context):
    """Prune ``tower`` at ``context.plan.layer`` and return the teardown."""

    encoder = tower_encoder(tower)
    num_groups = int(context.num_groups)

    def reduce(hidden_states, plan):
        return context.record_token_reduce(
            lambda: prune_frame_sequence(hidden_states, num_groups, plan),
            hidden_states,
        )

    return install_group_carrier(
        tower_blocks(tower), encoder, context, num_groups, reduce)


# VideoBind vision tower adapted from LanguageBind's
# languagebind/video/modeling_video.py (MIT); see NOTICE.
@dataclass
class VideoBindVisionConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    patch_size: int = 14
    image_size: int = 224
    num_channels: int = 3
    num_frames: int = 8
    add_time_attn: bool = True
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    hidden_act: str = "gelu"  # native checkpoint config
    force_patch_dropout: float = 0.0
    output_attentions: bool = False
    output_hidden_states: bool = False
    _attn_implementation: str = "eager"


class PatchDropout(nn.Module):
    """https://arxiv.org/abs/2212.00794 — identity when ``prob == 0``."""

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x, B, T):
        if not self.training or self.prob == 0.0:
            return x
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = x[:, :1]
        batch = x.size(0)
        num_tokens = x.size(1)
        batch_indices = torch.arange(batch)[..., None]
        num_patches_keep = max(1, int(num_tokens * (1 - self.prob)))
        if T == 1:
            rand = torch.randn(batch, num_tokens)
            patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices
        else:
            rand = torch.randn(B, num_tokens)
            patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices
            patch_indices_keep = patch_indices_keep.unsqueeze(1).repeat(1, T, 1)
            patch_indices_keep = rearrange(patch_indices_keep, "b t n -> (b t) n")
        x = x[batch_indices, patch_indices_keep]
        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)
        return x


class VideoBindEncoderLayer(nn.Module):
    """One divided space-time attention layer."""

    def __init__(self, config: VideoBindVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)  # type: ignore[arg-type]
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)  # type: ignore[arg-type]
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.add_time_attn = config.add_time_attn
        if self.add_time_attn:
            self.t = config.num_frames
            self.temporal_embedding = nn.Parameter(
                torch.zeros(1, config.num_frames, config.hidden_size)
            )
            nn.init.normal_(self.temporal_embedding, std=config.hidden_size**-0.5)
            self.temporal_attn = CLIPAttention(config)  # type: ignore[arg-type]
            self.temporal_layer_norm1 = nn.LayerNorm(
                self.embed_dim, eps=config.layer_norm_eps
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        if self.add_time_attn:
            bt, n, d = hidden_states.shape
            t = self.t

            # temporal position embedding (frame-level)
            if t != 1:
                hidden_states = rearrange(hidden_states, "(b t) n d -> (b n) t d", t=t)
                hidden_states = hidden_states + self.temporal_embedding[:, :t, :]
                hidden_states = rearrange(hidden_states, "(b n) t d -> (b t) n d", n=n)

            # temporal attention over the frame axis
            residual = hidden_states
            hidden_states = rearrange(hidden_states, "(b t) n d -> (b n) t d", t=t)
            hidden_states = self.temporal_layer_norm1(hidden_states)
            hidden_states, attn_weights = self.temporal_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = residual + rearrange(
                hidden_states, "(b n) t d -> (b t) n d", n=n
            )

        # spatial attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class VideoBindEncoder(nn.Module):
    """Encoder with the same forward signature as the HF CLIP encoder."""

    def __init__(self, config: VideoBindVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [VideoBindEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else True

        encoder_states = ()
        all_attentions = ()

        hidden_states = inputs_embeds
        for layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    encoder_states if output_hidden_states else None,
                    all_attentions if output_attentions else None,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states if output_hidden_states else None,
            attentions=all_attentions if output_attentions else None,
        )


class VideoBindVisionTransformer(nn.Module):
    """Vision tower of the native checkpoint; forward takes ``[B, C, T, H, W]``."""

    def __init__(self, config: VideoBindVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config)  # type: ignore[arg-type]
        self.patch_dropout = PatchDropout(config.force_patch_dropout)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = VideoBindEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else True

        if len(pixel_values.shape) == 5:
            # [B, C, T, H, W] -> [(B*T), C, H, W], frame order matching config.num_frames.
            B, _, T, _, _ = pixel_values.shape
            pixel_values = rearrange(pixel_values, "b c t h w -> (b t) c h w")
        else:
            B, _, _, _ = pixel_values.shape
            T = 1

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.patch_dropout(hidden_states, B, T)
        hidden_states = self.pre_layrnorm(hidden_states)

        # return_dict is not forwarded: the hook follows the HF CLIP contract.
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        pooled_output = pooled_output.reshape(B, T, -1).mean(1)

        if output_hidden_states:
            encoder_outputs.hidden_states = [
                rearrange(i, "(b t) n c -> b t n c", b=B)
                for i in encoder_outputs.hidden_states
            ]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,  # type: ignore[arg-type]
            attentions=encoder_outputs.attentions,
        )


# As in mm_utils.tokenizer_image_token: one non-vocabulary id per frame.
IMAGE_TOKEN_INDEX = -200

#: System prompt of the llava_v1 training template.
LLAVA_V1_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's "
    "questions."
)
PROMPT_STYLES = ("default", "llava_v1")


def official_prompt_style():
    """Return the configured prompt style ("default" or "llava_v1")."""

    style = os.environ.get(PROMPT_STYLE_ENV, "default").strip().lower()
    if style not in PROMPT_STYLES:
        raise RuntimeError(f"{PROMPT_STYLE_ENV} must be one of {PROMPT_STYLES}, got {style!r}.")
    return style


# Follows tokenizer_image_token in LLaVA's llava/mm_utils.py (Apache-2.0); see NOTICE.
def tokenizer_image_token(prompt, processor, image_token_index=IMAGE_TOKEN_INDEX):
    """Split on "<image>" and insert the sentinel id between chunks (BOS kept once)."""
    input_ids = [processor.bos_id()]
    for index, chunk in enumerate(prompt.split("<image>")):
        if index:
            input_ids.append(image_token_index)
        input_ids.extend(processor.encode(chunk))
    return input_ids


def ensure_checkpoint_dir(source, *, local_files_only):
    """A local checkpoint directory, or hub repo ``source`` resolved through the HF cache."""

    checkpoint = Path(source).expanduser()
    if checkpoint.is_dir() and (checkpoint / "config.json").is_file():
        return checkpoint
    from huggingface_hub import snapshot_download

    checkpoint = Path(
        snapshot_download(repo_id=str(source), local_files_only=local_files_only)
    )
    if not (checkpoint / "config.json").is_file():
        raise RuntimeError(f"{source} does not contain config.json.")
    return checkpoint


def _load_submodule(module, state_dict, label):
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{label} weights do not match the checkpoint: "
            f"missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}."
        )
    return module.eval()


def load_official_checkpoint(
    checkpoint_dir, *, device, dtype, load_4bit, load_8bit
):
    """Build and stream-load the tower, projector, and LLM modules."""

    import json

    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    quantized = load_4bit or load_8bit
    config = json.loads(
        (checkpoint_dir / "config.json").read_text()
    )
    llm_config = LlamaConfig(
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        rms_norm_eps=config["rms_norm_eps"],
        vocab_size=config["vocab_size"],
        max_position_embeddings=config["max_position_embeddings"],
    )

    # Sharded repos use model-000NN-of-000NN; small ones ship one model.safetensors.
    shards = sorted(checkpoint_dir.glob("model-*.safetensors"))
    single = checkpoint_dir / "model.safetensors"
    if not shards and single.is_file():
        shards = [single]

    # Construct in the inference dtype: a 7B never materializes as host fp32.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tower = VideoBindVisionTransformer(VideoBindVisionConfig())
        projector = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096),
        )
        # Meta device: a real construction costs ~14GB host RAM.
        if not quantized:
            with torch.device("meta"):
                language_model = LlamaForCausalLM(llm_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    tower = tower.to(device)
    projector = projector.to(device)

    if quantized:
        # Same contract as the HF backend; lm_head stays bf16.
        from transformers import BitsAndBytesConfig

        language_model = LlamaForCausalLM.from_pretrained(
            checkpoint_dir,
            config=llm_config,
            dtype=dtype,
            device_map={"": str(device)},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                load_in_8bit=load_8bit,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["lm_head"],
            ),
        )
    else:
        rotary_path = "model.rotary_emb"
        rotary_parent_path, rotary_attr = rotary_path.rsplit(".", 1)
        rotary_cls = type(language_model.get_submodule(rotary_path))
        setattr(
            language_model.get_submodule(rotary_parent_path),
            rotary_attr,
            rotary_cls(config=llm_config).to(device),
        )

    # Stream tensor by tensor so peak host RAM is the largest tensor.
    tower_sd, projector_sd = {}, {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("model.video_tower.video_tower."):
                    tower_sd[
                        key[len("model.video_tower.video_tower.") :]
                    ] = handle.get_tensor(key).to(dtype=dtype)
                elif key.startswith("model.mm_projector."):
                    projector_sd[key[len("model.mm_projector.") :]] = (
                        handle.get_tensor(key).to(dtype=dtype)
                    )
                elif key.startswith("model.image_tower."):
                    continue
                elif key.endswith("self_attn.rotary_emb.inv_freq"):
                    # Recomputed from the config.
                    continue
                elif quantized:
                    continue
                elif key.startswith("model.") or key.startswith("lm_head."):
                    # The prefixes match LlamaForCausalLM: key == submodule path.
                    module_name, tensor_name = key.rsplit(".", 1)
                    module = language_model.get_submodule(module_name)
                    set_module_tensor_to_device(
                        module,
                        tensor_name,
                        device,
                        handle.get_tensor(key).to(dtype=dtype),
                    )

    return (
        _load_submodule(tower, tower_sd, "video_tower"),
        _load_submodule(projector, projector_sd, "projector"),
        language_model.eval(),
    )
