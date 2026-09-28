import re
from unittest import mock

import pytest

from benchmark.reporting.summary import (
    add_average_score,
    add_measurements,
    merge_task_rows,
    write_csv,
)
from token_pruner.hevc import EncodeResult
from token_pruner.hevc_cache import HevcArtifactStore
from token_pruner.measurements import (
    new_generation_measurements,
    print_generation_summary,
)


def test_batch_measurements_are_sample_weighted():
    result = {
        "measurements": {
            "requests": [],
            "batches": [
                {"variant": "pruned", "sample_count": 2, "e2e_ms": 10.0, "e2e_paid_ms": 10.0, "delta_peak_mem_mb": 4.0},
                {"variant": "pruned", "sample_count": 1, "e2e_ms": 40.0, "e2e_paid_ms": 40.0, "delta_peak_mem_mb": 8.0},
            ],
        }
    }
    row = {}
    add_measurements(row, result)
    assert row["timing__e2e_ms__mean"] == pytest.approx(20.0)
    assert "timing__e2e_ms__n" not in row
    assert "timing__e2e_ms__p50" not in row
    assert "timing__e2e_ms__p95" not in row
    assert row["memory__delta_peak_mem_mb__max"] == 8.0


def test_batch_scoped_csv_measurements_are_explicitly_labeled(tmp_path):
    result = {
        "measurements": {
            "requests": [],
            "batches": [
                {
                    "variant": "pruned",
                    "sample_count": 2,
                    "measurement_scope": "batch",
                    "e2e_ms": 10.0,
                    "e2e_paid_ms": 10.0,
                    "peak_mem_mb": 128.0,
                }
            ],
        }
    }
    row = {}
    add_measurements(row, result)
    output = tmp_path / "summary.csv"
    write_csv(output, [row], transpose=False)
    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert "timing__e2e_ms__batch__mean" in header
    assert "timing__e2e_ms__mean" not in header
    assert "memory__peak_mem_mb__batch__mean" in header
    assert "memory__peak_mem_mb__batch__max" in header


def test_uses_request_scope_for_per_request_metrics():
    result = {
        "measurements": {
            "requests": [
                {"variant": "pruned", "hevc_encode_ms": 10.0, "output_tokens": 2},
                {"variant": "pruned", "hevc_encode_ms": 20.0, "output_tokens": 4},
            ],
            "batches": [
                {
                    "variant": "pruned",
                    "sample_count": 2,
                    "hevc_encode_ms": 30.0,
                    "output_tokens": 3.0,
                    "e2e_ms": 100.0,
                    "e2e_paid_ms": 100.0,
                    "request_indices": [0, 1],
                    "batch_output_tokens": 6,
                }
            ],
        }
    }
    row = {}
    add_measurements(row, result)
    assert row["timing__hevc_encode_ms__mean"] == 15.0
    assert row["timing__output_tokens__mean"] == 3.0
    assert row["timing__e2e_ms__mean"] == 100.0
    assert "timing__request_indices__mean" not in row
    assert "timing__batch_output_tokens__mean" not in row


def test_terminal_summary_signal_breakdown_adds_up(capsys):
    timing, memory = new_generation_measurements()
    batch = {
        "sample_count": 2,
        "preprocessing_ms": 5.0,
        "hevc_encode_ms": 30.0,
        "signal_to_mask_ms": 2.0,
        "signal_overhead_ms": 3.0,
        "signal_ms": 35.0,
        "vlm_input_adapt_ms": 1.0,
        "vlm_generate_ms": 20.0,
        "ttft_ms": 100.0,
        "tpot_ms": 4.0,
        "e2e_ms": 40.0,
        "peak_mem_mb": 100.0,
        "delta_peak_mem_mb": 10.0,
    }
    requests = [
        {"hevc_encode_ms": 10.0, "ttft_ms": 90.0, "tpot_ms": 3.0},
        {"hevc_encode_ms": 20.0, "ttft_ms": 110.0, "tpot_ms": 5.0},
    ]

    print_generation_summary(
        timing,
        memory,
        request_measurements=requests,
        batch_measurements=[batch],
    )

    output = capsys.readouterr().out
    # The printed parts must be the parts of the printed signal total.
    assert "hevc=30.0" in output
    signal, parts = re.search(
        r"signal=([\d.]+)ms \(hevc=([\d.]+), to_mask=([\d.]+), overhead=([\d.]+)\)",
        output,
    ).group(1), re.findall(
        r"(?:hevc|to_mask|overhead)=([\d.]+)", output
    )
    assert float(signal) == pytest.approx(sum(float(part) for part in parts))
    # TTFT and TPOT are labelled per-request rates, not parts of a sum.
    assert "TTFT=100.0ms" in output
    assert "TPOT=4.0ms" in output


def test_task_first_rows_merge_with_equal_task_score():
    common = {
        "result_type": "lmms_eval",
        "model_name": "Video-LLaVA-7B-hf",
        "model_path": "LanguageBind/Video-LLaVA-7B-hf",
        "run_mode": "pruned",
        "batch_size": 1,
    }
    rows = [
        {
            **common,
            "experiment": "nextqa_video_llava_hf_pruned_case",
            "task": "nextqa_mc_test",
            "source_path": "nextqa.pt",
            "metric__nextqa_mc_test__exact_match": 0.6,
            "timing__e2e_ms__mean": 10.0,
            "_measurement_values": {"timing__e2e_ms": [10.0, 10.0]},
        },
        {
            **common,
            "experiment": "vinoground_video_llava_hf_pruned_case",
            "task": "vinoground",
            "source_path": "vinoground.pt",
            "metric__vinoground__vinoground_score_text": 60.0,
            "metric__vinoground__vinoground_score_group": 40.0,
            "timing__e2e_ms__mean": 40.0,
            "_measurement_values": {"timing__e2e_ms": [40.0]},
        },
    ]
    merged = merge_task_rows(rows)
    assert len(merged) == 1
    assert merged[0]["timing__e2e_ms__mean"] == pytest.approx(20.0)
    assert "timing__e2e_ms__n" not in merged[0]
    assert "timing__e2e_ms__p50" not in merged[0]
    assert "timing__e2e_ms__p95" not in merged[0]
    assert merged[0]["avg_score"] == pytest.approx((0.6 + 0.4) / 2)


def test_average_score_uses_only_explicit_primary_metrics():
    row = {
        "task": (
            "nextqa_mc_test+nextqa_oe_test+mvbench_action"
            "+motionbench_full"
        ),
        "metric__nextqa_mc_test__exact_match": 0.5,
        "metric__nextqa_oe_test__WUPS": 70.0,
        "metric__mvbench_action__mvbench_accuracy": 90.0,
        "metric__motionbench_full__motionbench_acc": 0.4,
        "metric__motionbench_full__motionbench_answered_rate": 1.0,
        "metric__motionbench_full__motionbench_tracking": 0.95,
    }
    add_average_score(row)
    assert row["task_score__nextqa_oe_test"] == pytest.approx(0.7)
    assert "suite_score__nextqa_oe" not in row
    assert row["suite_score__motionbench"] == pytest.approx(0.4)
    assert row["avg_score"] == pytest.approx((0.5 + 0.9 + 0.4) / 3)
    assert row["n_suites"] == 3
    assert row["expected_n_suites"] == 3


@pytest.mark.parametrize(
    ("task", "metric", "native", "normalized"),
    [
        ("intphys2", "intphys2_accuracy", 0.5, 0.5),
        ("motionbench_full", "motionbench_acc", 0.4, 0.4),
        ("mvbench_action", "mvbench_accuracy", 0.75, 0.0075),
        ("nextqa_mc_test", "exact_match", 0.6, 0.6),
        ("nextqa_oe_test", "WUPS", 70.0, 0.7),
        ("nextqa_oe_val", "WUPS", 65.0, 0.65),
        ("video_mmmu_perception", "mmmu_acc", 0.3, 0.3),
        ("vitatecs_occlusion", "accuracy", 55.0, 0.55),
        ("vinoground", "vinoground_score_group", 41.25, 0.4125),
    ],
)
def test_primary_score_contracts_cover_observed_tasks(
    task, metric, native, normalized
):
    row = {f"metric__{task}__{metric}": native}
    add_average_score(row)
    assert row[f"task_score__{task}"] == pytest.approx(normalized)


def test_average_score_is_absent_when_a_requested_suite_is_missing():
    row = {
        "task": "nextqa_mc_test+mvbench_action",
        "metric__nextqa_mc_test__exact_match": 0.6,
    }
    add_average_score(row)
    assert "avg_score" not in row
    assert row["missing_suites"] == "mvbench"
    assert row["n_suites"] == 1
    assert row["expected_n_suites"] == 2


def test_mean6_matches_equal_suite_aggregate():
    row = {
        "task": (
            "intphys2+motionbench_full+mvbench_action+nextqa_mc_test"
            "+video_mmmu_perception+vitatecs_occlusion"
        ),
        "metric__intphys2__intphys2_accuracy": 0.5,
        "metric__motionbench_full__motionbench_acc": 0.46875,
        "metric__mvbench_action__mvbench_accuracy": 43.984375,
        "metric__nextqa_mc_test__exact_match": 0.59375,
        "metric__video_mmmu_perception__mmmu_acc": 0.1354166667,
        "metric__vitatecs_occlusion__accuracy": 51.04166667,
    }
    add_average_score(row)
    expected = (
        0.5 + 0.46875 + 0.43984375 + 0.59375 + 0.1354166667 + 0.5104166667
    ) / 6
    assert row["avg_score"] == pytest.approx(expected)
    assert row["mean6_score"] == pytest.approx(expected)
    assert row["n_suites"] == 6


def test_task_cache_encodes_once_and_reuses_atomic_artifact(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    cache = tmp_path / "cache"

    def encode(_source, destination, *, preferred_encoder, **_kwargs):
        destination.write_bytes(b"encoded")
        return EncodeResult(12.5, preferred_encoder, preferred_encoder)

    encoder = mock.Mock(side_effect=encode)
    store = HevcArtifactStore(cache)
    first = store.get_or_encode(
        source, sampled_gop_size=8, effective_gop_size=8,
        encode_scope="full-video", encode_fn=encoder,
    )
    second = store.get_or_encode(
        source, sampled_gop_size=8, effective_gop_size=8,
        encode_scope="full-video", encode_fn=encoder,
    )

    assert encoder.call_count == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.encode_ms == 12.5
    assert second.paid_encode_ms == 0.0


def test_task_first_rows_merge_official_prefix():
    """The official backend's artifact prefix merges like the HF one."""
    from benchmark.reporting.summary import merge_task_rows

    rows = [
        {
            "result_type": "lmms_eval",
            "model_name": "Video-LLaVA-7B",
            "model_path": "LanguageBind/Video-LLaVA-7B",
            "run_mode": "pruned",
            "batch_size": 1,
            "experiment": "nextqa_video_llava_official_pruned_case",
            "task": "nextqa_mc_test",
            "source_path": "nextqa.pt",
            "metric__nextqa_mc_test__exact_match": 0.6,
            "timing__e2e_ms__mean": 10.0,
            "_measurement_values": {"timing__e2e_ms": [10.0]},
        },
        {
            "result_type": "lmms_eval",
            "model_name": "Video-LLaVA-7B",
            "model_path": "LanguageBind/Video-LLaVA-7B",
            "run_mode": "pruned",
            "batch_size": 1,
            "experiment": "vinoground_video_llava_official_pruned_case",
            "task": "vinoground",
            "source_path": "vinoground.pt",
            "metric__vinoground__vinoground_score_text": 60.0,
            "metric__vinoground__vinoground_score_group": 40.0,
            "timing__e2e_ms__mean": 40.0,
            "_measurement_values": {"timing__e2e_ms": [40.0]},
        },
    ]
    merged = merge_task_rows(rows)
    assert len(merged) == 1
    assert merged[0]["timing__e2e_ms__mean"] == pytest.approx(25.0)
    assert merged[0]["avg_score"] == pytest.approx((0.6 + 0.4) / 2)
