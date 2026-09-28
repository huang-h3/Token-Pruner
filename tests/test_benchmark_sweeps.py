import os
from pathlib import Path
import tempfile
from unittest import mock

import pytest

from benchmark.sweeps import classification_sweep, vlm_sweep
from token_pruner.run_videollava_official import VIDEO_LLAVA_OFFICIAL_LEGAL_MODES


def test_classification_sweep_expands_modes_stages_and_rates():
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        permanent_dirs = []
        environment = {
            "MODEL_TYPE": "timesformer",
            "RUN_MODE": "pruned",
            "SUITE": "timesformer_modes",
            "PRUNE_STAGE": "input last",
            "PRUNE_LAYER": "-2",
            "K_KEEP_RATE": "0.25 0.5",
            "OUTPUT_DIR": str(output_dir),
            "RESULTS_CSV": str(output_dir / "summary.csv"),
            "HEVC_DIR": str(output_dir / "hevc"),
            "HEVC_PERMANENT_DIR": str(output_dir / "store"),
            "SLURM_JOB_ID": "1234",
            "SLURM_ARRAY_TASK_ID": "7",
            "RANDOM_SAMPLES": "",
            "SKIP_EXISTING": "0",
        }

        def record_run(command):
            if command[1:3] == ["-m", "benchmark.infer_clf"]:
                permanent_dirs.append(Path(command[command.index("--hevc-permanent-dir") + 1]))
                work_dir = Path(command[command.index("--hevc-dir") + 1])
                assert work_dir == output_dir / "hevc"

        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("benchmark.sweeps.run", side_effect=record_run) as run,
        ):
            classification_sweep()

    commands = [call.args[0] for call in run.call_args_list]
    inference_commands = [
        command
        for command in commands
        if command[1:3] == ["-m", "benchmark.infer_clf"]
    ]
    # 3 regular modes x 2 stages x 2 rates.
    assert len(inference_commands) == 12
    global_folder_commands = [
        cmd for cmd in inference_commands
        if cmd[cmd.index("--p-mode") + 1] == "global_folder"
    ]
    assert len(global_folder_commands) == 4
    assert all(cmd[cmd.index("--i-mode") + 1] == "global_folder"
               and cmd[cmd.index("--prune-mode") + 1] == "global"
               for cmd in global_folder_commands)
    assert all(
        command[command.index("--prune-mode") + 1] in {"global", "local"}
        for command in inference_commands
    )
    result_paths = {
        command[command.index("--results-path") + 1]
        for command in inference_commands
    }
    assert any("input_layer-neg2_k-0p25" in path for path in result_paths)
    assert any("last_layer-neg2_k-0p5" in path for path in result_paths)
    assert permanent_dirs == [output_dir / "store"] * len(inference_commands)

    summary = commands[-1]
    assert summary[1:3] == ["-m", "benchmark.reporting.summary"]
    assert summary[summary.index("--output") + 1] == str(output_dir / "summary.csv")


def test_vlm_sweep_builds_artifacts_and_marks_completed_case():
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        environment = {
            "TASKS": "nextqa_mc_test mvbench_action",
            "RUN_MODE": "pruned",
            "PRUNE_STAGE": "last",
            "PRUNE_LAYER": "-1",
            "K_KEEP_RATE": "0.25",
            "OUTPUT_DIR": str(output_dir),
            "RESULTS_CSV": str(output_dir / "summary.csv"),
            "HEVC_DIR": str(output_dir / "hevc"),
            "HEVC_PERMANENT_DIR": str(output_dir / "store"),
            "SLURM_JOB_ID": "5678",
            "SLURM_ARRAY_TASK_ID": "3",
            "SLURM_TMPDIR": str(output_dir / "slurm-tmp"),
            "LOG_SAMPLES": "0",
            "HEVC_ENCODE_SCOPE": "full-video",
            "HEVC_SAMPLED_GOP_SIZE": "16",
            "HEVC_ANCHOR_POLICY": "all",
            "SKIP_EXISTING": "0",
        }
        commands = []
        permanent_dirs = []

        def fake_run(command):
            if "--results-path" in command:
                permanent_dirs.append(Path(command[command.index("--hevc-permanent-dir") + 1]))
                work_dir = Path(command[command.index("--hevc-dir") + 1])
                assert work_dir == output_dir / "hevc"
            commands.append(command)
            if "--results-path" in command:
                Path(command[command.index("--results-path") + 1]).touch()
                native = Path(command[command.index("--output-path") + 1])
                native.mkdir(parents=True, exist_ok=True)
                (native / "fake_results.json").write_text(
                    '{"results": {"fake": {"accuracy": 1.0}}}',
                    encoding="utf-8",
                )

        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("benchmark.sweeps.run", side_effect=fake_run),
        ):
            vlm_sweep()

        inference_commands = [
            command for command in commands if "--results-path" in command
        ]
        assert [
            command[command.index("--tasks") + 1]
            for command in inference_commands
        ] == ["nextqa_mc_test", "mvbench_action"]
        assert permanent_dirs == [output_dir / "store"] * 2
        assert permanent_dirs[0].is_dir()

        inference = commands[0]
        assert inference[1:3] == ["-m", "benchmark.infer_vlm"]
        assert (
            inference[inference.index("--hevc-encode-scope") + 1] == "full-video"
        )
        native_dir = Path(inference[inference.index("--output-path") + 1])
        assert "_gop-16_anchor-all_layer-neg1" in native_dir.name
        assert native_dir.name.endswith("_layer-neg1_k-0p25_full-split")
        assert inference[inference.index("--prune-layer") + 1] == "-1"
        marker = native_dir / ".complete.json"
        assert marker.is_file()
        assert '"complete": true' in marker.read_text(encoding="utf-8")
        # Results, artifacts and summary all sit under the task's own directory.
        assert native_dir.parent.parent == output_dir / "nextqa_mc_test"
        results_path = Path(inference[inference.index("--results-path") + 1])
        assert results_path.parent == output_dir / "nextqa_mc_test"

        summaries = [
            command
            for command in commands
            if command[1:3] == ["-m", "benchmark.reporting.summary"]
        ]
        assert len(summaries) == 2, "one summary per task, not one per sweep"
        roots = [
            Path(command[command.index("--results-root") + 1])
            for command in summaries
        ]
        assert roots == [
            output_dir / "nextqa_mc_test",
            output_dir / "mvbench_action",
        ]
        for command, root in zip(summaries, roots):
            assert Path(command[command.index("--output") + 1]).parent == root
        # RESULTS_CSV pins one path, so it cannot name two summaries.
        assert all(
            command[command.index("--output") + 1] != str(output_dir / "summary.csv")
            for command in summaries
        )


def test_vlm_sweep_honours_results_csv_for_a_single_task():
    """One task produces one summary, so an explicit path can still pin it."""

    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        environment = {
            "TASKS": "nextqa_mc_test",
            "RUN_MODE": "full",
            "OUTPUT_DIR": str(output_dir),
            "RESULTS_CSV": str(output_dir / "summary.csv"),
            "HEVC_DIR": str(output_dir / "hevc"),
            "HEVC_PERMANENT_DIR": str(output_dir / "store"),
            "LOG_SAMPLES": "0",
            "SKIP_EXISTING": "0",
        }
        commands = []

        def fake_run(command):
            commands.append(command)
            if "--results-path" in command:
                Path(command[command.index("--results-path") + 1]).touch()
                native = Path(command[command.index("--output-path") + 1])
                native.mkdir(parents=True, exist_ok=True)
                (native / "fake_results.json").write_text(
                    '{"results": {"fake": {"accuracy": 1.0}}}', encoding="utf-8")

        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("benchmark.sweeps.run", side_effect=fake_run),
        ):
            vlm_sweep()

        summary = commands[-1]
        assert summary[1:3] == ["-m", "benchmark.reporting.summary"]
        assert summary[summary.index("--results-root") + 1] == str(
            output_dir / "nextqa_mc_test")
        assert summary[summary.index("--output") + 1] == str(
            output_dir / "summary.csv")


@pytest.mark.parametrize("model_id", [
    "video_llava_hf", "video_llava_official", "qwen3_vl",
    "llava_next_video",
])
def test_every_swept_mode_is_accepted_by_the_vlm_cli(model_id):
    """Every swept mode must survive argparse: the sweep runs with check=True."""

    from benchmark.infer_vlm import build_parser

    parser = build_parser()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        commands = []
        environment = {
            "MODEL_ID": model_id, "MODEL_PATH": "x", "CKPT_NAME": "ck",
            "TASKS": "nextqa", "RUN_MODE": "both", "PRUNE_STAGE": "input last",
            "SUITE": "all_modes", "K_KEEP_RATE": "0.125 0.5",
            "OUTPUT_DIR": str(output_dir), "HEVC_DIR": str(output_dir / "hevc"),
            "HEVC_PERMANENT_DIR": str(output_dir / "store"),
            "LOG_SAMPLES": "0", "SKIP_EXISTING": "0", "LIMIT": "8",
        }

        def fake_run(command):
            commands.append(command)
            if "--results-path" in command:
                Path(command[command.index("--results-path") + 1]).touch()
                native = Path(command[command.index("--output-path") + 1])
                native.mkdir(parents=True, exist_ok=True)
                (native / "fake_results.json").write_text(
                    '{"results": {"fake": {"accuracy": 1.0}}}', encoding="utf-8")

        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("benchmark.sweeps.run", side_effect=fake_run),
        ):
            vlm_sweep()

    inference = [c for c in commands if "--results-path" in c]
    assert inference, f"{model_id} swept nothing"
    for command in inference:
        try:
            parser.parse_args(command[command.index("--model-id"):])
        except SystemExit:  # argparse exits on an invalid choice
            triple = tuple(
                command[command.index(f"--{name}") + 1]
                for name in ("prune-mode", "i-mode", "p-mode")
                if f"--{name}" in command
            )
            raise AssertionError(
                f"{model_id} sweeps {triple}, which benchmark/infer_vlm.py rejects")


def test_concurrent_cases_cannot_share_one_arm(tmp_path):
    """The lock keeps a second writer out of one arm's response file."""

    import subprocess
    import sys
    import textwrap

    from benchmark.sweeps import _exclusive

    arm = tmp_path / "lmms_artifacts" / "one_arm"
    probe = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
        from pathlib import Path
        from benchmark.sweeps import _exclusive
        with _exclusive(Path({str(arm)!r})) as owned:
            print("owned" if owned else "refused")
        """
    )

    with _exclusive(arm) as owned:
        assert owned
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert "refused" in result.stdout
        assert str(os.getpid()) in (arm / ".writer.lock").read_text()

    # Released, so the next case may claim it.
    with _exclusive(arm) as owned:
        assert owned


def test_shared_hevc_completed_run_invalidates_on_encoding_change(tmp_path):
    """Changing CRF must not reuse a shared-HEVC result with a different encoding configuration."""
    environment = {
        "MODEL_ID": "video_llava_hf", "MODEL_PATH": "x",
        "TASKS": "nextqa_mc_test", "RUN_MODE": "pruned",
        "PRUNE_STAGE": "last", "PRUNE_MODE": "shared",
        "I_MODE": "shared_hevc", "P_MODE": "shared_hevc",
        "OUTPUT_DIR": str(tmp_path), "HEVC_DIR": str(tmp_path / "hevc"),
        "HEVC_PERMANENT_DIR": str(tmp_path / "store"),
        "SKIP_EXISTING": "1", "LOG_SAMPLES": "0", "LIMIT": "2",
        "HEVC_ENCODE_SCOPE": "full-video", "HEVC_SAMPLED_GOP_SIZE": "32",
        "CRF": "23",
    }
    inference = []

    def execute(command):
        if "--results-path" not in command:
            return
        inference.append(command)
        Path(command[command.index("--results-path") + 1]).touch()
        native = Path(command[command.index("--output-path") + 1])
        native.mkdir(parents=True, exist_ok=True)
        (native / "fake_results.json").write_text(
            '{"results": {"nextqa_mc_test": {"exact_match": 1.0}}}')

    with (
        mock.patch.dict(os.environ, environment, clear=True),
        mock.patch("benchmark.sweeps.run", side_effect=execute),
    ):
        vlm_sweep()
        vlm_sweep()
        assert len(inference) == 1
        os.environ["CRF"] = "24"
        vlm_sweep()
        assert len(inference) == 2


def test_all_modes_covers_official_legal_modes():
    from benchmark.vlm_cases import MODEL_RUNS

    profile = MODEL_RUNS["video_llava_official"]
    for stage in ("input", "last"):
        swept = {
            (stage, mode.prune_mode, mode.i_mode, mode.p_mode)
            for mode in profile.modes
        }
        assert swept == {
            mode for mode in VIDEO_LLAVA_OFFICIAL_LEGAL_MODES if mode[0] == stage
        }
