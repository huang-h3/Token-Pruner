import pytest
import torch

pytest.importorskip("pydantic")
pytest.importorskip(
    "lmms_eval",
    reason=(
        "lmms-eval is not importable, so the whole generation-contract module "
        "would be skipped silently. Install lmms-eval in the test environment "
        "or set LMMS_EVAL_ROOT to its checkout."
    ),
)


def test_expected_records_include_semantics_and_environment_fingerprint(monkeypatch):
    from types import SimpleNamespace

    from token_pruner.generation import _expected_records

    for name in (
        "TOKEN_PRUNER_STOP_STRINGS",
        "TOKEN_PRUNER_ANSWER_FORMAT_SUFFIX",
        "TOKEN_PRUNER_UNTRIMMED_SCORING",
        "TOKEN_PRUNER_REASONING_SEGMENT",
        "TOKEN_PRUNER_MAX_NEW_TOKENS",
        "VIDEO_ABLATION",
        "HEVC_ENCODER",
    ):
        monkeypatch.setenv(name, f"value-{name}")
    monkeypatch.setenv("TOKEN_PRUNER_GREEDY_DECODING", "0")
    monkeypatch.setenv("QWEN3_VL_DECODING_POLICY", "checkpoint")
    owner = SimpleNamespace(
        family="qwen3_vl",
        uniform_partitions=False,
        min_cells_per_partition=1,
        selector_id="tubelet",
        hevc_anchor_policy="first",
        hevc_encode_scope="full-video",
        hevc_gop_size=32,
        run_mode="pruned",
        run_pruned=True,
        prune_stage="last",
        prune_mode="global",
        prune_layer=8,
        i_mode="preserve",
        p_mode="hevc",
        selection_seed=42,
        folding_mode="temporal-diff",
        fold_block_size=2,
        fold_pooling="weighted",
        k_keep_rate=0.5,
        pretrained="dummy",
        num_frames=64,
        batch_size=8,
        score_reduce="max",
        hevc_n_parallel=6,
        model_dtype="bfloat16",
    )
    record = _expected_records(
        owner,
        [("context", lambda _doc: [], {}, 7, "nextqa_mc_test", "test")],
    )[0]
    assert record["partition_rule"] == "clip_free"
    assert record["min_per_partition"] == 1
    assert record["model_dtype"] == "bfloat16"
    assert record["decoding_policy"] == "checkpoint"
    assert record["environment"]["TOKEN_PRUNER_GREEDY_DECODING"] == "0"
    assert record["environment"]["QWEN3_VL_DECODING_POLICY"] == "checkpoint"
    assert record["environment"]["HEVC_ENCODER"] == "value-HEVC_ENCODER"
    assert record["environment"]["TOKEN_PRUNER_UNTRIMMED_SCORING"] == (
        "value-TOKEN_PRUNER_UNTRIMMED_SCORING"
    )
    assert record["environment"]["VIDEO_ABLATION"] == "value-VIDEO_ABLATION"
    assert record["model_source"] == "dummy"


def test_restored_batch_keeps_every_request_scoped_measurement():
    from types import SimpleNamespace

    from token_pruner.generation import (
        _restore_recorded_state,
    )

    batch = {
        "batch_id": 0,
        "sample_count": 2,
        "preprocessing_ms": 5.0,
        "peak_mem_mb": 100.0,
    }
    records = [
        {
            "measurements": {"hevc_encode_ms": 10.0, "ttft_ms": 90.0},
            "batch_measurement": batch,
        },
        {
            "measurements": {"hevc_encode_ms": 20.0, "ttft_ms": 110.0},
            "batch_measurement": batch,
        },
    ]
    owner = SimpleNamespace()
    timing, memory, batches = _restore_recorded_state(owner, records)

    assert timing["preprocessing_ms"] == [5.0]
    assert timing["hevc_encode_ms"] == [10.0, 20.0]
    assert timing["ttft_ms"] == [90.0, 110.0]
    assert memory["peak_mem_mb"] == [100.0]
    assert batches == [batch]


def test_generate_until_sends_the_configured_batch_to_generate(
    monkeypatch,
    capsys,
):
    from pathlib import Path
    from types import SimpleNamespace
    from unittest import mock

    import torch

    from token_pruner import generation
    from token_pruner.hevc_cache import HevcSelectionResult

    class FakeMessages:
        def __init__(self, messages):
            self.messages = messages

        def to_hf_messages(self):
            return []

        def extract_media(self):
            return [], ["video.mp4"], []

    class FakeBackend:
        family = "video_llava_hf"
        model_name = "fake"
        source = "fake"
        quantization = None
        vision_backend = "fake"

        def __init__(self):
            self.sample_index = 0
            self.batch_sizes = []
            self.received_stops = []

        def prepare_sample(self, _messages, _media, _num_frames):
            index = self.sample_index
            self.sample_index += 1
            return SimpleNamespace(
                video_path=Path(f"{index}.mp4"),
                frame_indices=[0],
                sampled_frames=torch.zeros(1),
                sample_index=index,
            )

        @staticmethod
        def prepare_generation_kwargs(kwargs):
            kwargs = dict(kwargs)
            stops = kwargs.pop("until", None)
            return kwargs, [stops] if stops else []

        def prepare_batch_inputs(self, samples):
            self.batch_sizes.append(len(samples))
            input_ids = torch.tensor(
                [[10 + sample.sample_index, 11] for sample in samples]
            )
            return {"input_ids": input_ids}

        def generate_tokens(self, inputs, _kwargs, stop_sequences=None):
            suffix = torch.full((inputs["input_ids"].shape[0], 1), 99)
            self.received_stops.append(stop_sequences)
            return torch.cat((inputs["input_ids"], suffix), dim=1)

        @staticmethod
        def last_untrimmed_texts():
            return []

        @staticmethod
        def decode_tokens(_output, inputs, _stops):
            return [
                f"response-{int(row[0]) - 10}"
                for row in inputs["input_ids"]
            ]

        @staticmethod
        def collect_batch_timing(_output, inputs):
            size = inputs["input_ids"].shape[0]
            return (
                {
                    "vision_encoder_ms": 1.0,
                    "projector_ms": 1.0,
                    "prefill_ms": 1.0,
                    "decode_ms": 0.0,
                },
                [
                    {
                        "output_tokens": 1,
                        "model_ttft_ms": 1.0,
                        "tpot_ms": None,
                    }
                    for _ in range(size)
                ],
            )

    class FakePruning:
        prune_stage = "last"
        prune_mode = "local"
        prune_layer = 11
        i_mode = "preserve"
        p_mode = "hevc"
        keep_patches = 1
        target_keep_total = 3
        total_patches = 4
        total_patches_clip = 12
        hevc_n_parallel = 1
        signal_provider = object()
        context = SimpleNamespace(plan=None, reset_timing=mock.Mock())

        def __init__(self):
            self.prepare_calls = 0

        def prepare_batch(self, paths, _indices, _frames):
            self.prepare_calls += 1
            size = len(paths)
            return HevcSelectionResult(
                hevc_encode_ms_per_sample=[1.0] * size,
                hevc_encode_paid_ms_per_sample=[1.0] * size,
                hevc_cache_hits=[False] * size,
            )

        @staticmethod
        def current_keep_budgets(_batch_size):
            return [(3, 1, 2)] * 3

        @staticmethod
        def enable():
            return None

        @staticmethod
        def prepare_inputs(_backend, inputs, _budget):
            return inputs

        @staticmethod
        def token_reduce_ms():
            return 0.0

    backend = FakeBackend()
    pruning = FakePruning()
    owner = SimpleNamespace(
        family="video_llava_hf",
        uniform_partitions=False,
        min_cells_per_partition=0,
        selector_id="patch",
        hevc_anchor_policy="first",
        hevc_encode_scope="full-video",
        hevc_gop_size=32,
        run_mode="pruned",
        run_pruned=True,
        run_full=False,
        prune_stage="last",
        prune_mode="local",
        prune_layer=11,
        i_mode="preserve",
        p_mode="hevc",
        folding_mode="temporal-diff",
        fold_block_size=2,
        fold_pooling="coverage-hard",
        k_keep_rate=0.5,
        pretrained="fake",
        num_frames=8,
        batch_size=3,
        score_reduce="max",
        model_dtype="float32",
        hevc_n_parallel=1,
        results_path=None,
        effective_keep_totals=[],
        i_frame_counts=[],
        p_keep_totals=[],
        task_dict={"task": {"test": [0, 1, 2]}},
        cache_hook=SimpleNamespace(add_partial=mock.Mock()),
        rank=1,
        device=torch.device("cpu"),
        backend=backend,
        pruning=pruning,
        _ensure_runtime=lambda: None,
    )
    requests = [
        SimpleNamespace(
            args=(
                f"context-{index}",
                lambda _doc: [],
                {"max_new_tokens": 1},
                index,
                "task",
                "test",
            )
        )
        for index in range(3)
    ]
    monkeypatch.setattr(generation, "ChatMessages", FakeMessages)

    responses = generation.run_generate_until(owner, requests)

    assert responses == ["response-0", "response-1", "response-2"]
    assert pruning.prepare_calls == 1
    assert backend.batch_sizes == [3]
    capsys.readouterr()


def test_generate_until_forwards_until_as_stop_strings(monkeypatch, capsys):
    from pathlib import Path
    from types import SimpleNamespace
    from unittest import mock

    import torch

    from token_pruner import generation
    from token_pruner.hevc_cache import HevcSelectionResult

    class FakeMessages:
        def __init__(self, messages):
            self.messages = messages

        def to_hf_messages(self):
            return []

        def extract_media(self):
            return [], ["video.mp4"], []

    class FakeBackend:
        family = "video_llava_hf"
        model_name = "fake"
        source = "fake"
        quantization = None
        vision_backend = "fake"

        def __init__(self):
            self.sample_index = 0
            self.batch_sizes = []
            self.received_stops = []

        def prepare_sample(self, _messages, _media, _num_frames):
            index = self.sample_index
            self.sample_index += 1
            return SimpleNamespace(
                video_path=Path(f"{index}.mp4"),
                frame_indices=[0],
                sampled_frames=torch.zeros(1),
                sample_index=index,
            )

        @staticmethod
        def prepare_generation_kwargs(kwargs):
            kwargs = dict(kwargs)
            stops = kwargs.pop("until", None)
            return kwargs, [stops] if stops else []

        def prepare_batch_inputs(self, samples):
            self.batch_sizes.append(len(samples))
            input_ids = torch.tensor(
                [[10 + sample.sample_index, 11] for sample in samples]
            )
            return {"input_ids": input_ids}

        def generate_tokens(self, inputs, _kwargs, stop_sequences=None):
            self.received_stops.append(stop_sequences)
            suffix = torch.full((inputs["input_ids"].shape[0], 1), 99)
            return torch.cat((inputs["input_ids"], suffix), dim=1)

        @staticmethod
        def last_untrimmed_texts():
            return []

        @staticmethod
        def decode_tokens(_output, inputs, _stops):
            return [
                f"response-{int(row[0]) - 10}"
                for row in inputs["input_ids"]
            ]

        @staticmethod
        def collect_batch_timing(_output, inputs):
            size = inputs["input_ids"].shape[0]
            return (
                {
                    "vision_encoder_ms": 1.0,
                    "projector_ms": 1.0,
                    "prefill_ms": 1.0,
                    "decode_ms": 0.0,
                },
                [
                    {
                        "output_tokens": 1,
                        "model_ttft_ms": 1.0,
                        "tpot_ms": None,
                    }
                    for _ in range(size)
                ],
            )

    class FakePruning:
        prune_stage = "last"
        prune_mode = "local"
        prune_layer = 11
        i_mode = "preserve"
        p_mode = "hevc"
        keep_patches = 1
        target_keep_total = 3
        total_patches = 4
        total_patches_clip = 12
        hevc_n_parallel = 1
        signal_provider = object()
        context = SimpleNamespace(plan=None, reset_timing=mock.Mock())

        def __init__(self):
            self.prepare_calls = 0

        def prepare_batch(self, paths, _indices, _frames):
            self.prepare_calls += 1
            return HevcSelectionResult(
                hevc_encode_ms_per_sample=[1.0] * len(paths),
                hevc_encode_paid_ms_per_sample=[1.0] * len(paths),
                hevc_cache_hits=[False] * len(paths),
            )

        @staticmethod
        def current_keep_budgets(_batch_size):
            return [(3, 1, 2)] * 3

        @staticmethod
        def enable():
            return None

        @staticmethod
        def prepare_inputs(_backend, inputs, _budget):
            return inputs

        @staticmethod
        def token_reduce_ms():
            return 0.0

    backend = FakeBackend()
    owner = SimpleNamespace(
        family="video_llava_hf",
        uniform_partitions=False,
        min_cells_per_partition=0,
        selector_id="patch",
        hevc_anchor_policy="first",
        hevc_encode_scope="full-video",
        hevc_gop_size=32,
        run_mode="pruned",
        run_pruned=True,
        run_full=False,
        prune_stage="last",
        prune_mode="local",
        prune_layer=11,
        i_mode="preserve",
        p_mode="hevc",
        folding_mode="temporal-diff",
        fold_block_size=2,
        fold_pooling="coverage-hard",
        k_keep_rate=0.5,
        pretrained="fake",
        num_frames=8,
        batch_size=3,
        score_reduce="max",
        model_dtype="float32",
        hevc_n_parallel=1,
        results_path=None,
        effective_keep_totals=[],
        i_frame_counts=[],
        p_keep_totals=[],
        task_dict={"task": {"test": [0, 1, 2]}},
        cache_hook=SimpleNamespace(add_partial=mock.Mock()),
        rank=1,
        device=torch.device("cpu"),
        backend=backend,
        pruning=FakePruning(),
        _ensure_runtime=lambda: None,
    )
    requests = [
        SimpleNamespace(
            args=(
                f"context-{index}",
                lambda _doc: [],
                {"max_new_tokens": 1, "until": "\n\n"},
                index,
                "task",
                "test",
            )
        )
        for index in range(3)
    ]
    monkeypatch.setattr(generation, "ChatMessages", FakeMessages)

    generation.run_generate_until(owner, requests)

    assert backend.received_stops == [["\n\n"]]
    capsys.readouterr()


def test_videollava_hf_generate_tokens_forwards_stop_strings(monkeypatch):
    from unittest import mock

    from token_pruner.task_io import STOP_STRINGS_ENV
    from token_pruner.run_videollava_hf import (
        VideoLlavaHfBackend,
    )

    backend = VideoLlavaHfBackend.__new__(VideoLlavaHfBackend)
    backend._stage_timers = {"vision_encoder": mock.Mock()}
    backend._step_observer = mock.Mock()
    backend.processor = mock.Mock()
    backend.processor.tokenizer = "fake-tokenizer"
    backend.model = mock.Mock()

    monkeypatch.setenv(STOP_STRINGS_ENV, "1")
    backend.generate_tokens(
        {"input_ids": "x"}, {"max_new_tokens": 8}, ["\n\n"]
    )

    kwargs = backend.model.generate.call_args.kwargs
    assert kwargs["stop_strings"] == ["\n\n"]
    assert kwargs["tokenizer"] == "fake-tokenizer"


def test_videollava_hf_generate_tokens_omits_stops_when_empty():
    from unittest import mock

    from token_pruner.run_videollava_hf import (
        VideoLlavaHfBackend,
    )

    backend = VideoLlavaHfBackend.__new__(VideoLlavaHfBackend)
    backend._stage_timers = {"vision_encoder": mock.Mock()}
    backend._step_observer = mock.Mock()
    backend.processor = mock.Mock()
    backend.model = mock.Mock()

    backend.generate_tokens({"input_ids": "x"}, {"max_new_tokens": 8})

    kwargs = backend.model.generate.call_args.kwargs
    assert "stop_strings" not in kwargs
    assert "tokenizer" not in kwargs


def test_videollava_official_generate_tokens_forwards_stop_strings(monkeypatch):
    from types import SimpleNamespace
    from unittest import mock

    import torch

    from token_pruner.task_io import STOP_STRINGS_ENV
    from token_pruner.run_videollava_official import (
        VideoLlavaOfficialBackend,
    )

    monkeypatch.setenv(STOP_STRINGS_ENV, "1")

    backend = VideoLlavaOfficialBackend.__new__(VideoLlavaOfficialBackend)
    backend._stage_timers = {"language_model": mock.Mock()}
    backend._step_observer = mock.Mock()
    backend.tokenizer = "fake-tokenizer"
    backend.language_model = mock.Mock()
    backend.language_model.get_input_embeddings.return_value = (
        lambda input_ids: torch.zeros(*input_ids.shape, 4)
    )
    backend.projector = mock.Mock(return_value=torch.zeros(1, 1, 2, 4))
    backend.video_tower = mock.Mock(
        return_value=SimpleNamespace(
            hidden_states=[None, torch.zeros(1, 1, 2, 4)]
        )
    )

    backend.generate_tokens(
        {
            "input_ids": torch.tensor([[-200, -200]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values_videos": torch.zeros(1, 1, 1, 1, 1),
        },
        {"max_new_tokens": 8},
        ["\n\n"],
    )

    kwargs = backend.language_model.generate.call_args.kwargs
    assert kwargs["stop_strings"] == ["\n\n"]
    assert kwargs["tokenizer"] == "fake-tokenizer"


def test_official_masked_scatter_preserves_frame_major_visual_order():
    from types import SimpleNamespace
    from unittest import mock

    import torch

    from token_pruner.run_videollava_official import (
        VideoLlavaOfficialBackend,
    )

    backend = VideoLlavaOfficialBackend.__new__(VideoLlavaOfficialBackend)
    backend._stage_timers = {"language_model": mock.Mock()}
    backend._step_observer = mock.Mock()
    backend.tokenizer = "fake-tokenizer"
    backend.language_model = mock.Mock()
    backend.language_model.get_input_embeddings.return_value = (
        lambda input_ids: input_ids.to(torch.float32).unsqueeze(-1).expand(
            *input_ids.shape, 3
        )
    )
    visual = torch.tensor(
        [[[[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
          [[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]]]]
    )
    backend.projector = mock.Mock(return_value=visual)
    backend.video_tower = mock.Mock(
        return_value=SimpleNamespace(hidden_states=[None, torch.zeros(1)])
    )

    backend.generate_tokens(
        {
            "input_ids": torch.tensor([[5, -200, -200, 6, -200, -200, 7]]),
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
            "pixel_values_videos": torch.zeros(1, 1, 1, 1, 1),
        },
        {},
    )

    expected = torch.tensor(
        [[[5.0, 5.0, 5.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0],
          [6.0, 6.0, 6.0], [30.0, 31.0, 32.0], [40.0, 41.0, 42.0],
          [7.0, 7.0, 7.0]]]
    )
    actual = backend.language_model.generate.call_args.kwargs["inputs_embeds"]
    assert torch.equal(actual, expected)


def test_stop_strings_is_off_unless_explicitly_enabled(monkeypatch):
    """An unset environment measures; early stopping on ``until`` is opt-in."""

    from unittest import mock

    from token_pruner.task_io import (
        STOP_STRINGS_ENV,
        stop_strings_enabled,
    )
    from token_pruner.run_videollava_hf import (
        VideoLlavaHfBackend,
    )

    def call():
        backend = VideoLlavaHfBackend.__new__(VideoLlavaHfBackend)
        backend._stage_timers = {"vision_encoder": mock.Mock()}
        backend._step_observer = mock.Mock()
        backend.processor = mock.Mock()
        backend.processor.tokenizer = "fake-tokenizer"
        backend.model = mock.Mock()
        backend.generate_tokens({"input_ids": "x"}, {"max_new_tokens": 8}, ["\n\n"])
        return backend.model.generate.call_args.kwargs

    monkeypatch.delenv(STOP_STRINGS_ENV, raising=False)
    assert stop_strings_enabled() is False
    kwargs = call()
    assert "stop_strings" not in kwargs
    assert "tokenizer" not in kwargs

    # Values that read as "off" must not enable it either.
    for value in ("0", "", "false", "no", "off", "FALSE"):
        monkeypatch.setenv(STOP_STRINGS_ENV, value)
        assert stop_strings_enabled() is False, value
        assert "stop_strings" not in call()

    for value in ("1", "true", "yes"):
        monkeypatch.setenv(STOP_STRINGS_ENV, value)
        assert stop_strings_enabled() is True, value
        assert call()["stop_strings"] == ["\n\n"]


def test_official_backend_resolves_a_hub_checkpoint(tmp_path, monkeypatch):
    from token_pruner.run_videollava_official import (
        ensure_checkpoint_dir,
    )

    snapshot = tmp_path / "snapshot"
    downloads = []

    def fake_download(repo_id, local_files_only):
        downloads.append((repo_id, local_files_only))
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text("{}")
        return str(snapshot)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_download
    )
    result = ensure_checkpoint_dir("LanguageBind/Video-LLaVA-7B", local_files_only=True)

    assert downloads == [("LanguageBind/Video-LLaVA-7B", True)]
    assert result == snapshot


def test_official_backend_reuses_existing_checkpoint(tmp_path, monkeypatch):
    from unittest import mock

    from token_pruner.run_videollava_official import (
        ensure_checkpoint_dir,
    )

    existing = tmp_path / "ckpt"
    existing.mkdir()
    (existing / "config.json").write_text("{}")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        mock.Mock(side_effect=AssertionError("should not download")),
    )
    result = ensure_checkpoint_dir(str(existing), local_files_only=False)

    assert result == existing


def test_official_backend_rejects_a_snapshot_without_config(tmp_path, monkeypatch):
    from token_pruner.run_videollava_official import (
        ensure_checkpoint_dir,
    )

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, local_files_only: str(tmp_path),
    )
    with pytest.raises(RuntimeError, match="config.json"):
        ensure_checkpoint_dir("LanguageBind/Video-LLaVA-7B", local_files_only=False)


_PROMPT_MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "Why did the boy clap?"}]}
]


def test_official_prompt_uses_configured_template(monkeypatch):
    from token_pruner.run_videollava_official import (
        VideoLlavaOfficialBackend,
    )

    monkeypatch.delenv("VIDEOLLAVA_OFFICIAL_PROMPT", raising=False)
    prompt = VideoLlavaOfficialBackend._prompt_from_messages(_PROMPT_MESSAGES)

    assert prompt.startswith("USER: ")
    assert "<image>" * 8 in prompt
    assert "A chat between" not in prompt


def test_official_prompt_llava_v1_adds_system_and_spaces(monkeypatch):
    """The native checkpoint is vicuna-v1.5 based and trained with this prefix."""

    from token_pruner.run_videollava_official import (
        LLAVA_V1_SYSTEM_PROMPT,
        VideoLlavaOfficialBackend,
    )

    monkeypatch.setenv("VIDEOLLAVA_OFFICIAL_PROMPT", "llava_v1")
    prompt = VideoLlavaOfficialBackend._prompt_from_messages(_PROMPT_MESSAGES)

    assert prompt.startswith(f"{LLAVA_V1_SYSTEM_PROMPT} USER: ")
    assert " ".join(["<image>"] * 8) in prompt
    # The concatenated spelling must be gone, otherwise the sentinels double up.
    assert "<image>" * 8 not in prompt
    assert prompt.endswith("ASSISTANT:")


def test_official_prompt_styles_keep_one_sentinel_per_frame(monkeypatch):
    """Both styles must expand to exactly num_frames image placeholders."""

    from token_pruner.run_videollava_official import (
        VideoLlavaOfficialBackend,
    )

    for style in ("default", "llava_v1"):
        monkeypatch.setenv("VIDEOLLAVA_OFFICIAL_PROMPT", style)
        prompt = VideoLlavaOfficialBackend._prompt_from_messages(_PROMPT_MESSAGES)
        assert prompt.count("<image>") == 8, style


def test_official_batch_inputs_pad_on_the_left():
    """Decoder-only generation resumes from the final position, so pads go left."""

    from types import SimpleNamespace

    import torch

    from token_pruner.run_videollava_official import (
        IMAGE_TOKEN_INDEX,
        VideoLlavaOfficialBackend,
    )

    class FakeSentencePiece:
        def bos_id(self):
            return 1

        def encode(self, text):
            # One id per character, offset away from bos/pad and the sentinel.
            return [ord(c) % 100 + 10 for c in text]

    backend = VideoLlavaOfficialBackend.__new__(VideoLlavaOfficialBackend)
    backend.tokenizer = SimpleNamespace(pad_token_id=0)
    backend.sentencepiece = FakeSentencePiece()
    backend.video_tower = SimpleNamespace(
        config=SimpleNamespace(image_size=224, patch_size=14)
    )
    backend.device = torch.device("cpu")
    backend.model_dtype = torch.float32
    backend._preprocess_frames = lambda frames: torch.zeros(3, 2, 2, 2)

    samples = [
        SimpleNamespace(
            prompt=backend.video_placeholder + "short",
            sampled_frames=None,
        ),
        SimpleNamespace(
            prompt=backend.video_placeholder + "a much longer prompt than the other",
            sampled_frames=None,
        ),
    ]
    inputs = backend.prepare_batch_inputs(samples)
    input_ids, mask = inputs["input_ids"], inputs["attention_mask"]

    assert input_ids.shape[0] == 2
    lengths = mask.sum(dim=1).tolist()
    assert lengths[0] < lengths[1], "the fixture must have unequal lengths"

    for row, length in enumerate(lengths):
        pad_width = input_ids.shape[1] - length
        # Padding sits at the front, real tokens run to the end.
        assert torch.all(mask[row, :pad_width] == 0)
        assert torch.all(mask[row, pad_width:] == 1)
        # The last position must be a real token, never padding.
        assert mask[row, -1] == 1
        assert input_ids[row, -1] != backend.tokenizer.pad_token_id

    # Every sample keeps its sentinel count so visual assembly still lines up.
    for row in range(2):
        assert int((input_ids[row] == IMAGE_TOKEN_INDEX).sum()) == 8 * 257


def test_official_batch_inputs_expand_contiguous_frame_sentinels():
    from types import SimpleNamespace

    import torch

    from token_pruner.run_videollava_official import (
        IMAGE_TOKEN_INDEX,
        VideoLlavaOfficialBackend,
    )

    class FakeSentencePiece:
        def bos_id(self):
            return 1

        def encode(self, text):
            return [ord(c) % 100 + 10 for c in text]

    backend = VideoLlavaOfficialBackend.__new__(VideoLlavaOfficialBackend)
    backend.tokenizer = SimpleNamespace(pad_token_id=0)
    backend.sentencepiece = FakeSentencePiece()
    backend.video_tower = SimpleNamespace(
        config=SimpleNamespace(image_size=224, patch_size=14)
    )
    backend.device = torch.device("cpu")
    backend.model_dtype = torch.float32
    backend._preprocess_frames = lambda frames: torch.zeros(3, 2, 2, 2)

    inputs = backend.prepare_batch_inputs([
        SimpleNamespace(
            prompt=backend.video_placeholder + "short",
            sampled_frames=None,
        )
    ])
    assert int((inputs["input_ids"][0] == IMAGE_TOKEN_INDEX).sum()) == 8 * 257


def _official_backend_for_placeholder_tests():
    """A backend able to tokenize and rewrite placeholders, with no weights."""

    from types import SimpleNamespace

    import torch

    from token_pruner.run_videollava_official import (
        VideoLlavaOfficialBackend,
    )

    class FakeSentencePiece:
        def bos_id(self):
            return 1

        def encode(self, text):
            return [ord(c) % 100 + 10 for c in text]

    backend = VideoLlavaOfficialBackend.__new__(VideoLlavaOfficialBackend)
    backend.tokenizer = SimpleNamespace(pad_token_id=0)
    backend.sentencepiece = FakeSentencePiece()
    backend.video_tower = SimpleNamespace(
        config=SimpleNamespace(image_size=224, patch_size=14)
    )
    backend.device = torch.device("cpu")
    backend.model_dtype = torch.float32
    backend._preprocess_frames = lambda frames: torch.zeros(3, 2, 2, 2)
    return backend


def test_official_resize_shrinks_both_prompt_style_run_layouts(monkeypatch):
    """Both prompt styles shrink to the same total visual width."""

    from types import SimpleNamespace

    from token_pruner.task_io import placeholder_runs
    from token_pruner.run_videollava_official import (
        IMAGE_TOKEN_INDEX,
    )

    backend = _official_backend_for_placeholder_tests()
    messages = _PROMPT_MESSAGES

    for style, expected_runs in (("default", 1), ("llava_v1", 8)):
        monkeypatch.setenv("VIDEOLLAVA_OFFICIAL_PROMPT", style)
        prompt = type(backend)._prompt_from_messages(messages)
        samples = [
            SimpleNamespace(prompt=prompt, sampled_frames=None),
            SimpleNamespace(prompt=prompt + " padded out", sampled_frames=None),
        ]
        inputs = backend.prepare_batch_inputs(samples)
        for row in inputs["input_ids"]:
            assert int((row == IMAGE_TOKEN_INDEX).sum()) == 8 * 257, style

        pruned = backend.resize_pruned_inputs(inputs, 8 * 33)
        for row_index, row in enumerate(pruned["input_ids"]):
            valid = row[pruned["attention_mask"][row_index].bool()]
            runs = placeholder_runs(valid, IMAGE_TOKEN_INDEX)
            assert len(runs) == expected_runs, style
            assert sum(end - start for start, end in runs) == 8 * 33, style
            # Decoding must resume from a real token, never from a pad.
            assert int(pruned["attention_mask"][row_index][-1]) == 1, style


def test_skip_existing_off_disables_response_replay(monkeypatch):
    """SKIP_EXISTING=0 must defeat the response replay as well as the markers."""

    from token_pruner.records import (
        RESUME_ENV,
        resume_enabled,
    )

    monkeypatch.delenv(RESUME_ENV, raising=False)
    assert resume_enabled() is True, "resume stays the default for crash recovery"

    for value in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv(RESUME_ENV, value)
        assert resume_enabled() is False, value

    monkeypatch.setenv(RESUME_ENV, "1")
    assert resume_enabled() is True


_SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def _mcq_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "clip.mp4"},
                {"type": "text", "text": "What is X?\nA. foo\nB. bar\nC. baz"},
            ],
        }
    ]


def test_answer_format_suffix_is_off_by_default(monkeypatch):
    from token_pruner.generation import (
        apply_answer_format_suffix,
    )

    monkeypatch.delenv("TOKEN_PRUNER_ANSWER_FORMAT_SUFFIX", raising=False)
    messages = _mcq_messages()
    apply_answer_format_suffix(messages)

    assert messages[0]["content"][1]["text"] == "What is X?\nA. foo\nB. bar\nC. baz"


def test_answer_format_suffix_appends_to_multiple_choice(monkeypatch):
    from token_pruner.generation import (
        apply_answer_format_suffix,
    )

    monkeypatch.setenv("TOKEN_PRUNER_ANSWER_FORMAT_SUFFIX", _SUFFIX)
    messages = _mcq_messages()
    apply_answer_format_suffix(messages)

    assert messages[0]["content"][1]["text"].endswith(_SUFFIX)
    # The media block and the question itself must survive untouched.
    assert messages[0]["content"][0] == {"type": "video", "video": "clip.mp4"}
    assert messages[0]["content"][1]["text"].startswith("What is X?\nA. foo")


def test_answer_format_suffix_skips_open_ended_prompts(monkeypatch):
    """Video-MMMU mixes open-ended items into the same task."""

    from token_pruner.generation import (
        apply_answer_format_suffix,
    )

    monkeypatch.setenv("TOKEN_PRUNER_ANSWER_FORMAT_SUFFIX", _SUFFIX)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Name the algorithm in the video."}],
        }
    ]
    apply_answer_format_suffix(messages)

    assert messages[0]["content"][0]["text"] == "Name the algorithm in the video."


def test_answer_format_suffix_targets_the_last_user_turn(monkeypatch):
    from token_pruner.generation import (
        apply_answer_format_suffix,
    )

    monkeypatch.setenv("TOKEN_PRUNER_ANSWER_FORMAT_SUFFIX", _SUFFIX)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        *_mcq_messages(),
    ]
    apply_answer_format_suffix(messages)

    assert messages[0]["content"][0]["text"] == "sys"
    assert messages[1]["content"][1]["text"].endswith(_SUFFIX)


# Scoring operators: which part of one generation reaches the scorer.


class _ReasoningBackend:
    """Minimal backend exposing only what ``decode_tokens`` needs."""

    from token_pruner.run_videollava_hf import VideoLlavaHfBackend as _base

    reasoning_close_tag = "</think>"
    decode_tokens = _base.decode_tokens
    last_untrimmed_texts = _base.last_untrimmed_texts

    def __init__(self, text):
        self._text = text

    def _resolve_input_length(self, _inputs):
        return 0

    def _slice_generated(self, sequence, _input_length):
        return sequence

    def _decode_ids(self, _token_ids):
        return self._text


_TRACE = "weighing the options\n\nstill weighing\n</think>\n\nB"


@pytest.mark.parametrize(
    ("segment", "untrimmed", "expected"),
    [
        ("raw", False, "weighing the options"),
        ("raw", True, _TRACE),
        ("answer", False, "B"),
        ("answer", True, "B"),
    ],
)
def test_reasoning_segment_and_trim_compose(monkeypatch, segment, untrimmed, expected):
    monkeypatch.setenv("TOKEN_PRUNER_REASONING_SEGMENT", segment)
    monkeypatch.setenv("TOKEN_PRUNER_UNTRIMMED_SCORING", "1" if untrimmed else "0")
    backend = _ReasoningBackend(_TRACE)

    scored = backend.decode_tokens(torch.zeros(1, 1), {}, [["\n\n"]])

    assert scored == [expected]
    # The recorded generation stays complete, so another operator can replay it.
    assert backend.last_untrimmed_texts() == [_TRACE]


def test_reasoning_segment_is_inert_without_a_close_tag(monkeypatch):
    monkeypatch.setenv("TOKEN_PRUNER_REASONING_SEGMENT", "answer")
    monkeypatch.setenv("TOKEN_PRUNER_UNTRIMMED_SCORING", "1")
    backend = _ReasoningBackend("plain answer")
    backend.reasoning_close_tag = None

    assert backend.decode_tokens(torch.zeros(1, 1), {}, [["\n\n"]]) == ["plain answer"]


def test_unclosed_trace_scores_as_empty_under_the_answer_segment(monkeypatch):
    monkeypatch.setenv("TOKEN_PRUNER_REASONING_SEGMENT", "answer")
    monkeypatch.setenv("TOKEN_PRUNER_UNTRIMMED_SCORING", "1")
    backend = _ReasoningBackend("ran out of budget mid-thought")

    # An empty string means the budget expired before any answer.
    assert backend.decode_tokens(torch.zeros(1, 1), {}, [["\n\n"]]) == [""]


def _qwen_backend_for_decoding_tests():
    from types import SimpleNamespace

    from token_pruner.run_qwen3_vl import Qwen3VlHfBackend

    backend = object.__new__(Qwen3VlHfBackend)
    backend.model = SimpleNamespace(
        generation_config=SimpleNamespace(
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
        )
    )
    return backend


def test_qwen_checkpoint_policy_overrides_task_decoding_but_keeps_task_budget(
    monkeypatch,
):
    monkeypatch.delenv("TOKEN_PRUNER_GREEDY_DECODING", raising=False)
    monkeypatch.delenv("TOKEN_PRUNER_MAX_NEW_TOKENS", raising=False)
    monkeypatch.setenv("QWEN3_VL_DECODING_POLICY", "checkpoint")
    backend = _qwen_backend_for_decoding_tests()

    kwargs, stops = backend.prepare_generation_kwargs(
        {
            "until": ["stop"],
            "max_new_tokens": 64,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.2,
        }
    )

    assert stops == ["stop"]
    assert kwargs == {
        "max_new_tokens": 64,
        "num_beams": 1,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
    }


def test_qwen_task_policy_and_greedy_override(monkeypatch):
    monkeypatch.delenv("TOKEN_PRUNER_GREEDY_DECODING", raising=False)
    monkeypatch.delenv("TOKEN_PRUNER_MAX_NEW_TOKENS", raising=False)
    monkeypatch.setenv("QWEN3_VL_DECODING_POLICY", "task")
    backend = _qwen_backend_for_decoding_tests()

    kwargs, _ = backend.prepare_generation_kwargs(
        {"max_new_tokens": 64, "do_sample": False, "temperature": 0.0}
    )
    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] == 0.0

    monkeypatch.setenv("QWEN3_VL_DECODING_POLICY", "checkpoint")
    monkeypatch.setenv("TOKEN_PRUNER_GREEDY_DECODING", "1")
    kwargs, _ = backend.prepare_generation_kwargs(
        {"max_new_tokens": 64, "do_sample": True}
    )
    assert backend.decoding_parameters() == {"policy": "greedy", "do_sample": False}
    assert kwargs["do_sample"] is False
