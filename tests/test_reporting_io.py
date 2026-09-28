"""Shared result-reader behavior."""

import json
import os
from pathlib import Path

import pytest
import torch

from benchmark.reporting import io, summary


ROOT = Path(__file__).resolve().parents[1]
TASK = "nextqa_mc_test"


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def response(doc=0, task=TASK, tokens=10, batch=0):
    return {
        "task": task, "split": "test", "doc_id": doc,
        "run_mode": "pruned", "prune_stage": "input",
        "prune_mode": "local", "i_mode": "random", "p_mode": "random",
        "k_keep_rate": 0.5, "raw_response": "A",
        "measurements": {"output_tokens": tokens, "batch_id": batch},
        "batch_measurement": {"batch_id": batch, "sample_count": 1, "decode_ms": 20},
    }


@pytest.fixture
def arm_path(tmp_path):
    return tmp_path / "hf-example" / "Video-LLaVA-7B-hf" / "nextqa" / "case.responses.jsonl"


def test_all_response_readers_keep_first_copy_and_batch_identity(arm_path):
    first = response()
    duplicate = {**response(tokens=999, batch=1), "raw_response": "B"}
    write_jsonl(arm_path, [first, duplicate, response(doc=1)])
    stats = {}
    rows = list(io.iter_responses(arm_path, stats=stats))
    assert [row["measurements"]["output_tokens"] for row in rows] == [10, 10]
    assert stats["duplicates"] == 1
    assert [row["raw_response"] for row in rows] == ["A", "A"]
    assert rows[0]["batch_measurement"]["batch_id"] == 0


def test_response_identity_includes_task_and_split(arm_path):
    first = response()
    second = {**response(), "split": "validation"}
    third = response(task="nextqa_oe_test")
    fourth = response(doc=1)
    write_jsonl(arm_path, [first, second, third, fourth])
    rows = list(io.iter_responses(arm_path))
    assert len(rows) == 4
    assert io.document_key(rows[-1]) == (TASK, "test", 1)
    incomplete = {k: v for k, v in response().items() if k != "doc_id"}
    with pytest.raises(ValueError, match="missing task, split, or doc_id"):
        io.document_key(incomplete)


def test_jsonl_tolerates_only_a_broken_final_record(arm_path):
    write_jsonl(arm_path, [response()])
    with arm_path.open("a") as handle:
        handle.write('{"task":')
    with pytest.warns(UserWarning, match="incomplete final record"):
        assert len(list(io.iter_responses(arm_path))) == 1
    with arm_path.open("a") as handle:
        handle.write("\n" + json.dumps(response(doc=1)) + "\n")
    with pytest.raises(ValueError, match="Invalid JSON.*:2"):
        list(io.iter_responses(arm_path))


def test_every_score_reader_uses_latest_native_result_and_matching_samples(arm_path):
    write_jsonl(arm_path, [response()])
    directory = io.artifact_dir(arm_path)
    first = directory / "z" / "z_results.json"
    new = directory / "a" / "nested" / "a_results.json"
    for path, score, mtime in [(first, 0.1, 100), (new, 0.75, 200)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "results": {TASK: {"exact_match,none": score}},
            "n-samples": {TASK: {"effective": 1}},
        }))
        os.utime(path, (mtime, mtime))
        write_jsonl(path.with_name(path.name.replace("_results.json", f"_samples_{TASK}.jsonl")),
                    [{"doc_id": 0, "exact_match": score > 0.5}])
    assert io.load_native_result(directory)[0] == new
    assert io.arm_accuracy(arm_path) == {TASK: 75.0}
    assert io.primary_scores(json.loads(new.read_text()))[TASK] == {
        "metric": "exact_match", "raw": 0.75, "score": 75.0, "effective": 1}
    row = {}
    summary.add_lmms_metrics(row, arm_path.parent, "case")
    assert row["metric__nextqa_mc_test__exact_match"] == 0.75


def test_an_arm_without_a_native_result_has_no_scores(arm_path):
    write_jsonl(arm_path, [response()])
    assert io.arm_accuracy(arm_path) == {}
    assert io.load_native_result(io.artifact_dir(arm_path))[1] == {}


def test_pt_reader_returns_params_and_tolerates_their_absence(tmp_path):
    params = {"model_type": "vivit", "num_frames": 32}
    path = tmp_path / "result.pt"
    torch.save({"params": params}, path)
    assert io.result_params(io.load_torch_result(path)) == params
    torch.save({"timing": {}}, path)
    assert io.result_params(io.load_torch_result(path)) == {}
