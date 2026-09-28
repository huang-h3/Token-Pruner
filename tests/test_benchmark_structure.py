from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy
import torch

from benchmark.classification.profiling import profiled_active_steps
from benchmark.reporting.io import result_params
from token_pruner.hevc import EncodeResult
from token_pruner.run_videollava_hf import VideoLlavaHfBackend
from token_pruner.run_videollava_hf import resize_video_placeholder
from token_pruner.tokens import SharedFoldingOptions
from token_pruner.shared_folding import (
    resolve_shared_folding_options,
)
from token_pruner.records import save_benchmark_result
from token_pruner.progress import ProgressReporter
from token_pruner.records import (
    ResponseStore,
)
from token_pruner.measurements import (
    BATCH_SCOPED_METRICS,
    build_request_measurement,
    new_generation_measurements,
    record_measurement,
)
from token_pruner.measurements import TimedInferenceResult
from token_pruner.hevc_cache import HevcArtifactStore
from token_pruner.hevc_cache import (
    prepare_hevc_batch_selection,
)
from token_pruner.video_io import (
    _apply_ablation,
    _donor_path,
    sample_video_uniform,
)


def test_video_ablations_keep_the_frame_tensor_shape_and_are_deterministic():
    frames = numpy.arange(3 * 4 * 4 * 3, dtype=numpy.uint8).reshape(3, 4, 4, 3)
    for mode in ("static", "black", "noise"):
        ablated = _apply_ablation(frames, mode, "/videos/a.mp4")
        assert ablated.shape == frames.shape
        assert ablated.dtype == frames.dtype
    # "static" removes motion only: every frame is the first one.
    static = _apply_ablation(frames, "static", "/videos/a.mp4")
    assert (static[0] == frames[0]).all()
    assert (static[1] == static[0]).all()
    # Noise is keyed on the path, so a rerun reproduces it.
    first = _apply_ablation(frames, "noise", "/videos/a.mp4")
    assert (first == _apply_ablation(frames, "noise", "/videos/a.mp4")).all()
    assert not (first == _apply_ablation(frames, "noise", "/videos/b.mp4")).all()


def test_shuffle_donor_is_another_video_and_is_stable(tmp_path):
    paths = []
    for name in ("a.mp4", "b.mp4", "c.mkv", "notes.txt"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(path)
    donor = _donor_path(paths[0])
    assert donor != str(paths[0])
    assert donor in {str(paths[1]), str(paths[2])}
    assert donor == _donor_path(paths[0])
    lonely = tmp_path / "solo"
    lonely.mkdir()
    only = lonely / "only.mp4"
    only.write_bytes(b"x")
    assert _donor_path(only) is None


def test_long_decord_reads_use_fresh_bounded_batches(monkeypatch):
    class Batch:
        def __init__(self, indices):
            self.indices = indices

        def asnumpy(self):
            return numpy.asarray(self.indices, dtype=numpy.int64)[:, None, None, None]

    class Reader:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.calls = []
            self.instances.append(self)

        def __len__(self):
            return 65

        def get_avg_fps(self):
            return 24.0

        def get_batch(self, indices):
            self.calls.append(indices)
            return Batch(indices)

    monkeypatch.delenv("VIDEO_ABLATION", raising=False)
    monkeypatch.setattr("token_pruner.video_io.decord.VideoReader", Reader)
    monkeypatch.setattr("token_pruner.video_io.decord.cpu", lambda _index: None)

    frames, indices, fps = sample_video_uniform("video.mp4", 32, with_fps=True)

    assert frames[:, 0, 0, 0].tolist() == indices
    assert fps == 24.0
    assert [len(reader.calls) for reader in Reader.instances] == [0, 1, 1]
    assert [len(reader.calls[0]) for reader in Reader.instances[1:]] == [16, 16]


def test_shared_folding_resolver_handles_global_and_block_budgets():
    global_options, global_keep = resolve_shared_folding_options(
        SharedFoldingOptions(
            patch_width=4,
            block_wise=False,
            global_k=None,
        ),
        total_columns=16,
        k_keep_rate=0.5,
    )
    assert global_keep == 8
    assert global_options.global_k == 8

    block_options, block_keep = resolve_shared_folding_options(
        SharedFoldingOptions(
            patch_width=4,
            block_size=2,
            slots_per_block=2,
            block_wise=True,
        ),
        total_columns=16,
        k_keep_rate=0.25,
    )
    assert block_keep == 8
    assert block_options.global_k is None


def test_result_writer_schema():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "result.pt"
        save_benchmark_result(
            path,
            timing={"forward_ms": [1.0, 2.0]},
            memory={"peak_mem_mb": [128.0]},
            params={"model_type": "timesformer"},
            accuracy={"pruned": [0.5, 0.75]},
        )
        saved = torch.load(path, map_location="cpu", weights_only=False)

    assert saved["schema_version"] == 1
    assert result_params(saved)["model_type"] == "timesformer"
    assert torch.equal(saved["timing"]["forward_ms"], torch.tensor([1.0, 2.0]))


def test_profiler_average_uses_actual_active_steps():
    assert profiled_active_steps(1) == 0
    assert profiled_active_steps(2) == 0
    assert profiled_active_steps(5) == 3
    assert profiled_active_steps(20) == 8


def test_batch_timing_records_decode_steps_from_lm_forward_spans():
    backend = VideoLlavaHfBackend.__new__(VideoLlavaHfBackend)
    backend._stage_timers = {
        "vision_encoder": SimpleNamespace(elapsed=lambda: [1.0]),
        "projector": SimpleNamespace(elapsed=lambda: [2.0]),
        "language_model": SimpleNamespace(elapsed=lambda: [10.0, 4.0, 5.0]),
    }
    backend._step_observer = SimpleNamespace(elapsed_steps=lambda: [10.0, 14.0, 19.0])
    backend._generation_config = lambda: SimpleNamespace(eos_token_id=None)
    backend._resolve_input_length = lambda _inputs: 1
    output = torch.tensor([[7, 8, 9], [7, 8, 9]])
    batch, requests = backend.collect_batch_timing(
        output,
        {"input_ids": output},
    )
    assert batch["prefill_ms"] == 10.0
    assert batch["decode_ms"] == 9.0
    assert batch["decode_steps"] == 2
    assert all(item["output_tokens"] == 2 for item in requests)


def _single(record):
    """A response record completed as its own one-request batch."""

    index = record["request_index"]
    return {
        **record,
        "batch_measurement": {"batch_id": index, "request_indices": [index]},
    }


def test_lmms_response_store_round_trips_completed_responses():
    with tempfile.TemporaryDirectory() as temporary:
        store = ResponseStore(Path(temporary) / "timing.pt")
        expected = [
            {
                "request_index": 0,
                "doc_id": 7,
                "task": "nextqa_mc_test",
                "split": "test",
            }
        ]
        store.begin()
        store.append(_single({**expected[0], "response": "answer"}))

        assert store.load(expected) == ["answer"]
        assert store.path.name == "timing.responses.jsonl"


def test_lmms_response_store_resumes_a_valid_prefix():
    with tempfile.TemporaryDirectory() as temporary:
        store = ResponseStore(Path(temporary) / "timing.pt")
        expected = [
            {"request_index": 0, "doc_id": 7},
            {"request_index": 1, "doc_id": 8},
        ]
        store.begin()
        store.append(_single({**expected[0], "response": "first"}))

        assert store.load_prefix(expected) == ["first"]
        store.begin(preserve=True)
        store.append(_single({**expected[1], "response": "second"}))
        assert store.load(expected) == ["first", "second"]


def test_lmms_response_store_rejects_records_without_a_batch():
    with tempfile.TemporaryDirectory() as temporary:
        store = ResponseStore(Path(temporary) / "timing.pt")
        expected = [{"request_index": 0, "doc_id": 7}]
        store.begin()
        store.append({**expected[0], "response": "first"})

        assert store.load_prefix(expected) is None


def test_lmms_response_store_discards_only_a_torn_trailing_line():
    with tempfile.TemporaryDirectory() as temporary:
        store = ResponseStore(Path(temporary) / "timing.pt")
        expected = [
            {"request_index": 0, "doc_id": 7},
            {"request_index": 1, "doc_id": 8},
        ]
        store.begin()
        store.append(_single({**expected[0], "response": "first"}))
        with store.path.open("a", encoding="utf-8") as output_file:
            output_file.write('{"request_index": 1')

        assert store.load_prefix(expected) == ["first"]
        store.begin(preserve=True)
        store.append(_single({**expected[1], "response": "second"}))
        assert store.load(expected) == ["first", "second"]


def test_lmms_response_store_trims_an_incomplete_batch():
    with tempfile.TemporaryDirectory() as temporary:
        store = ResponseStore(Path(temporary) / "timing.pt")
        expected = [
            {"request_index": index, "batch_size": 4}
            for index in range(6)
        ]
        batch = {
            "batch_id": 0,
            "request_indices": [0, 1, 2, 3],
            "sample_count": 4,
        }
        store.begin()
        store.append_many([
            {
                **expected[index],
                "response": str(index),
                "batch_measurement": batch,
            }
            for index in range(2)
        ])

        assert store.load_prefix(expected) == []
        assert store.records == []


def test_video_placeholder_resize_preserves_left_padding_and_pad_id():

    video_token_id = 99
    inputs = {
        "input_ids": torch.tensor([
            [32002, 32002, 1, 99, 99, 2],
            [1, 99, 99, 2, 3, 4],
        ]),
        "attention_mask": torch.tensor([
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]),
    }
    resized = resize_video_placeholder(
        inputs,
        video_token_id,
        1,
        pad_token_id=32002,
        padding_side="left",
    )

    assert resized["attention_mask"].tolist() == [
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    assert resized["input_ids"][0, :2].tolist() == [32002, 32002]
    assert resized["input_ids"][0, -3:].tolist() == [1, 99, 2]


def test_vlm_request_measurement_uses_disjoint_stages():
    preparation = SimpleNamespace(
        hevc_encode_ms=4.0,
        hevc_encode_paid_ms=4.0,
        hevc_cache_hits=[False],
        selector_compute_ms=6.0,
    )
    inference = TimedInferenceResult(
        output=None,
        elapsed_ms=20.0,
        peak_mem_mb=100.0,
        delta_peak_mem_mb=10.0,
    )
    record = build_request_measurement(
        variant="pruned",
        preprocessing_ms=3.0,
        inference=inference,
        fine={
            "vision_encoder_ms": 2.0,
            "projector_ms": 1.0,
            "prefill_ms": 7.0,
            "decode_ms": 8.0,
            "output_tokens": 3,
            "ttft_ms": 11.0,
            "tpot_ms": 4.0,
        },
        preparation=preparation,
        vlm_input_adapt_ms=2.0,
        token_reduce_ms=1.0,
        signal_prepare_wall_ms=15.0,
        e2e_wall_ms=40.0,
    )
    timing, memory = new_generation_measurements()
    record_measurement(timing, memory, record)

    assert record["measurement_scope"] == "request"
    assert record["signal_to_mask_ms"] == 6.0
    # 15 wall - 4 paid encode - 6 selector = 5 unattributed.
    assert record["signal_overhead_ms"] == 5.0
    assert record["signal_ms"] == 15.0
    assert record["model_ms"] == 22.0
    assert record["inference_ms"] == 37.0
    assert record["e2e_ms"] == 40.0
    assert record["e2e_paid_ms"] == 40.0
    assert record["e2e_wall_ms"] == 40.0
    assert timing["ttft_ms"] == [11.0]
    assert timing["output_tokens"] == [3]
    assert memory["delta_peak_mem_mb"] == []


def test_vlm_stage_totals_sum_to_e2e_without_a_residual():
    """e2e must equal the sum of its stages, including the signal remainder."""

    preparation = SimpleNamespace(
        hevc_encode_ms=30.0,
        hevc_encode_paid_ms=0.0,  # cache hit: cold cost still charged to e2e_ms
        hevc_cache_hits=[True],
        selector_compute_ms=50.0,
    )
    inference = TimedInferenceResult(
        output=None,
        elapsed_ms=100.0,
        peak_mem_mb=1.0,
        delta_peak_mem_mb=1.0,
    )
    record = build_request_measurement(
        variant="pruned",
        preprocessing_ms=200.0,
        inference=inference,
        fine={"output_tokens": 1},
        preparation=preparation,
        vlm_input_adapt_ms=5.0,
        token_reduce_ms=1.0,
        signal_prepare_wall_ms=120.0,
        e2e_wall_ms=425.0,
    )

    assert record["signal_overhead_ms"] == 70.0  # 120 - 0 paid - 50 selector
    assert record["signal_ms"] == 150.0  # cold 30 + 50 + 70
    assert record["signal_paid_ms"] == 120.0  # paid 0 + 50 + 70
    assert record["model_ms"] == 105.0
    assert record["e2e_ms"] == record["preprocessing_ms"] + record["inference_ms"]
    assert record["e2e_ms"] == 455.0
    # The cache saved the encode, so the paid total matches the wall clock.
    assert record["e2e_paid_ms"] == 425.0
    assert record["e2e_paid_ms"] == record["e2e_wall_ms"]


def test_vlm_batched_request_measurement_omits_batch_owned_metrics():
    """A batched request record omits what the batch owns; it does not null it."""

    inference = TimedInferenceResult(
        output=None,
        elapsed_ms=20.0,
        peak_mem_mb=100.0,
        delta_peak_mem_mb=10.0,
    )
    fine = {
        "vision_encoder_ms": 2.0,
        "projector_ms": 1.0,
        "output_tokens": 3,
        "hit_token_cap": False,
        "model_ttft_ms": 9.0,
        "ttft_ms": 11.0,
        "tpot_ms": 4.0,
    }
    arguments = dict(
        variant="pruned",
        preprocessing_ms=3.0,
        inference=inference,
        fine=fine,
        preparation=None,
        vlm_input_adapt_ms=2.0,
        token_reduce_ms=1.0,
        e2e_wall_ms=40.0,
    )
    batched = build_request_measurement(**arguments, batch_scoped=True)

    assert not (set(batched) & BATCH_SCOPED_METRICS)
    # Absence, not a null: a null would still claim the field was measured here.
    assert all(value is not None for value in batched.values())
    # Everything genuinely per-request survives.
    assert batched["measurement_scope"] == "request"
    assert batched["output_tokens"] == 3 and batched["tpot_ms"] == 4.0
    assert batched["hevc_encode_ms"] == 0.0

    # The unbatched record is untouched and still balances.
    single = build_request_measurement(**arguments, batch_scoped=False)
    assert BATCH_SCOPED_METRICS <= set(single)
    assert single["model_ms"] == 22.0
    assert single["e2e_ms"] == single["preprocessing_ms"] + single["inference_ms"]

    # The numeric view keeps the scopes apart either way.
    timing, memory = new_generation_measurements()
    record_measurement(timing, memory, batched)
    assert timing["output_tokens"] == [3]
    assert timing["vlm_generate_ms"] == [] and memory["peak_mem_mb"] == []


def test_progress_reporter_prints_persisted_values_every_twenty_samples():
    stream = StringIO()
    reporter = ProgressReporter(total=21, every=20, stream=stream)
    measurement = {
        "delta_peak_mem_mb": 10.0,
        "ttft_ms": 11.0,
        "tpot_ms": 4.0,
        "signal_ms": 6.0,
        "token_reduce_ms": 1.0,
        "model_ms": 22.0,
        "inference_ms": 28.0,
        "e2e_ms": 36.0,
    }
    for _ in range(21):
        reporter.update(measurement)
    reporter.close()

    output = stream.getvalue()
    assert "[20/21 samples]" in output
    assert "[21/21 samples]" in output
    assert "TTFT=11.0ms" in output
    assert "signal=6.0ms" in output

    empty_stream = StringIO()
    ProgressReporter(total=10, stream=empty_stream).close()
    assert empty_stream.getvalue() == ""


def test_batch_hevc_preparation_reports_failed_sources():
    with tempfile.TemporaryDirectory() as temporary:
        selector = mock.Mock(return_value="signal")
        class FakeStore:
            def __init__(self, permanent):
                self.permanent_dir = Path(permanent)
                self.calls = 0

            def get_or_encode(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 2:
                    return None
                return SimpleNamespace(
                    path=Path(temporary) / "a.mp4", encode_ms=1.5,
                    paid_encode_ms=1.5, cache_hit=False, cache_key="a",
                    effective_gop_size=8,
                )

        with mock.patch(
            "token_pruner.hevc_cache.HevcArtifactStore", FakeStore
        ):
            result = prepare_hevc_batch_selection(
                ["/videos/a.mp4", "/videos/b.mp4"],
                temporary, selector, torch.device("cpu"),
            )

    assert result.selector_outputs == "signal"
    assert result.kept_indices == [0]
    assert result.failed_indices == [1]
    assert result.hevc_encode_ms == 1.5
    assert selector.call_args.args[1] == [0]


def test_task_hevc_cache_separates_cold_and_paid_encode_cost(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    cache = tmp_path / "cache"
    calls = []

    class SampledFrames:
        shape = (2, 2, 2, 3)
        dtype = "uint8"

        @staticmethod
        def tobytes(order="C"):
            assert order == "C"
            return bytes(range(24))

    def encode(_source, destination, *, sampled_frames, preferred_encoder, **_kwargs):
        calls.append((Path(destination), sampled_frames))
        Path(destination).write_bytes(b"encoded")
        return EncodeResult(12.5, preferred_encoder, preferred_encoder)

    store = HevcArtifactStore(cache)
    first = store.get_or_encode(
        source, sampled_frames=SampledFrames(), sampled_gop_size=2,
        encode_fn=encode, requested_encoder="libx265",
    )
    second = store.get_or_encode(
        source, sampled_frames=SampledFrames(), sampled_gop_size=2,
        encode_fn=encode, requested_encoder="libx265",
    )

    assert len(calls) == 1
    assert first.cache_hit is False
    assert first.encode_ms == first.paid_encode_ms == 12.5
    assert second.cache_hit is True
    assert second.encode_ms == 12.5
    assert second.paid_encode_ms == 0.0
    assert first.path == second.path
