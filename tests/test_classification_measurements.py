from types import SimpleNamespace
from unittest import mock

import torch

from benchmark.classification.measurements import (
    ClassificationMeasurements,
    TIMING_FIELDS,
)
from token_pruner.run_vivit import VivitRun
from token_pruner.run_timesformer import (
    forward_timesformer_encoder,
)
from token_pruner.run_vivit import forward_vivit_encoder


def test_classification_record_uses_the_batch_schema():
    measurements = ClassificationMeasurements()

    record = measurements.record_batch(
        variant="pruned",
        batch_index=3,
        sample_count=4,
        preprocessing_ms=10.0,
        hevc_encode_ms=2.0,
        signal_to_mask_ms=3.0,
        token_reduce_ms=4.0,
        vision_classification_forward_ms=20.0,
        peak_mem_mb=100.0,
        delta_peak_mem_mb=25.0,
    )

    assert record["measurement_scope"] == "batch"
    assert record["sample_count"] == 4
    assert record["signal_ms"] == 5.0
    assert record["model_ms"] == 20.0
    assert record["inference_ms"] == 25.0
    # token_reduce is nested in the complete forward and is not added twice.
    assert record["e2e_ms"] == 35.0
    assert record["e2e_paid_ms"] == 35.0
    assert set(TIMING_FIELDS) <= record.keys()

    timing = measurements.timing()
    assert timing["variant_code"] == [1]
    assert timing["hevc_encode_paid_ms"] == [2.0]
    assert timing["sample_count"] == [4]
    assert not any(
        name.startswith(("pruned_", "full_")) or "prepare" in name
        for name in timing
    )


def test_full_and_pruned_records_have_the_same_fields():
    measurements = ClassificationMeasurements()
    common = dict(
        batch_index=1,
        sample_count=2,
        preprocessing_ms=1.0,
        vision_classification_forward_ms=5.0,
        peak_mem_mb=10.0,
        delta_peak_mem_mb=2.0,
    )
    pruned = measurements.record_batch(
        variant="pruned",
        hevc_encode_ms=2.0,
        signal_to_mask_ms=3.0,
        token_reduce_ms=1.0,
        **common,
    )
    full = measurements.record_batch(variant="full", **common)

    assert pruned.keys() == full.keys()
    assert full["signal_ms"] == 0.0
    assert full["token_reduce_ms"] == 0.0


def test_e2e_uses_cold_encode_while_preserving_paid_latency():
    measurements = ClassificationMeasurements()
    record = measurements.record_batch(
        variant="pruned",
        batch_index=1,
        sample_count=1,
        preprocessing_ms=10.0,
        hevc_encode_ms=7.0,
        hevc_encode_paid_ms=0.0,
        signal_to_mask_ms=3.0,
        token_reduce_ms=2.0,
        vision_classification_forward_ms=20.0,
        peak_mem_mb=100.0,
        delta_peak_mem_mb=10.0,
    )

    assert record["signal_ms"] == 10.0
    assert record["signal_paid_ms"] == 3.0
    assert record["e2e_paid_ms"] == 33.0
    assert record["e2e_ms"] == 40.0


def test_classification_signal_overhead_is_attributed_to_the_signal_stage():
    measurements = ClassificationMeasurements()
    record = measurements.record_batch(
        variant="pruned",
        batch_index=0,
        sample_count=1,
        preprocessing_ms=100.0,
        hevc_encode_ms=30.0,
        hevc_encode_paid_ms=30.0,
        signal_to_mask_ms=50.0,
        signal_prepare_wall_ms=200.0,
        token_reduce_ms=1.0,
        vision_classification_forward_ms=10.0,
        peak_mem_mb=1.0,
        delta_peak_mem_mb=1.0,
    )

    # 200 wall - 30 encode - 50 selector = 120 that no inner timer covered.
    assert record["signal_overhead_ms"] == 120.0
    assert record["signal_ms"] == 200.0
    assert record["e2e_ms"] == record["preprocessing_ms"] + record["inference_ms"]
    assert record["e2e_ms"] == 310.0


def test_request_record_keeps_unattributable_batch_metrics_missing():
    measurements = ClassificationMeasurements()
    record = measurements.record_request(
        {
            "variant": "pruned",
            "measurement_scope": "request",
            "request_index": 7,
            "batch_index": 2,
            "batch_size_effective": 4,
            "decode_ms": 1.5,
            "hevc_encode_ms": 3.0,
            "signal_to_mask_ms": None,
            "token_reduce_ms": None,
            "vision_classification_forward_ms": None,
            "e2e_ms": None,
        }
    )

    assert record["hevc_encode_ms"] == 3.0
    assert record["vision_classification_forward_ms"] is None
    assert measurements.request_records == [record]


def test_session_complete_pruned_forward_keeps_token_reduce_callback():
    prepare = mock.Mock(return_value="prepared")
    forward = mock.Mock(return_value="logits")
    timer = object()
    session = VivitRun(
        model_type="test",
        model_name="test",
        model=object(),
        model_config=object(),
        image_processor=object(),
        device=torch.device("cpu"),
        num_frames=1,
        image_size=1,
        patch_size=1,
        run_pruned=True,
        run_full=False,
        pruning_plan=object(),
        selector=None,
        process_frames_fn=mock.Mock(),
        prepare_pruned_fn=prepare,
        forward_pruned_fn=forward,
        info="test",
        short_info="test",
        accuracy_key="test",
        total_tokens=1,
        total_tokens_per_step=1,
    )

    assert session.forward_pruned("pixels", "signal", timer) == "logits"
    prepare.assert_called_once_with(session, "pixels", "signal")
    forward.assert_called_once_with(session, "prepared", timer)


def test_vivit_times_complete_adapter_boundary():
    hidden_states = torch.randn(1, 3, 2)
    model = SimpleNamespace(
        vivit=SimpleNamespace(
            layers=[],
            layernorm=torch.nn.Identity(),
        ),
        classifier=torch.nn.Identity(),
    )
    plan = SimpleNamespace(layer=0)
    calls = []

    def timer(operation):
        calls.append("start")
        output = operation()
        calls.append("end")
        return output

    with mock.patch(
        "token_pruner.run_vivit."
        "prune_vivit_hidden_states",
        return_value=hidden_states,
    ) as prune:
        output = forward_vivit_encoder(
            model,
            hidden_states,
            plan,
            token_reduce=timer,
        )

    assert output.shape == (1, 2)
    assert calls == ["start", "end"]
    prune.assert_called_once_with(model, hidden_states, plan)


def test_timesformer_times_complete_adapter_boundary():
    hidden_states = torch.randn(1, 3, 2)
    model = SimpleNamespace(
        timesformer=SimpleNamespace(
            encoder=SimpleNamespace(layer=[]),
            layernorm=torch.nn.Identity(),
        ),
        classifier=torch.nn.Identity(),
    )
    plan = SimpleNamespace(layer=0)
    calls = []

    def timer(operation):
        calls.append("start")
        output = operation()
        calls.append("end")
        return output

    with mock.patch(
        "token_pruner.run_timesformer."
        "prune_timesformer_hidden_states",
        return_value=(hidden_states, 1, 2),
    ) as prune:
        output = forward_timesformer_encoder(
            model,
            hidden_states,
            1,
            2,
            plan,
            token_reduce=timer,
        )

    assert output.shape == (1, 2)
    assert calls == ["start", "end"]
    prune.assert_called_once_with(hidden_states, 1, 2, plan)
