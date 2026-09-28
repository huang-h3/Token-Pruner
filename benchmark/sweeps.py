#!/usr/bin/env python3
"""Experiment-matrix orchestration for the benchmark entry points."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from benchmark.paths import (
    CLF_RESULTS_ROOT,
    HEVC_STORE_DIR,
    PROJECT_ROOT,
    VLM_HEVC_DIR as DEFAULT_VLM_HEVC_DIR,
    VLM_RESULTS_ROOT as DEFAULT_VLM_RESULTS_ROOT,
)
from benchmark.vlm_cases import MODEL_RUNS, Mode
from token_pruner.hevc import HEVC_SIGNAL_POLICY, resolve_hevc_encode_scope
from token_pruner.records import RESUME_ENV, atomic_write_text
from token_pruner.shared_folding import FOLDING_MODES
from token_pruner.task_io import (
    inference_dtype_name,
    qwen3_vl_decoding_policy,
)


VIVIT_MODES = (
    Mode("global", "global_folder", "global_folder"),
    Mode("global", "preserve", "hevc"),
    Mode("global", "preserve", "folder"),
    Mode("local", "preserve", "hevc"),
    Mode("local", "preserve", "folder"),
    Mode("local", "folder", "hevc"),
    Mode("local", "folder", "folder"),
)
VIVIT_BASELINE_MODES = (
    Mode("global", "random", "random"),
    Mode("global", "uniform", "uniform"),
)
# Shared folding is reached through `shared_modes(stage)`.
TIMESFORMER_MODES = (
    Mode("global", "global_folder", "global_folder"),
    Mode("local", "folder", "hevc"),
    Mode("local", "folder", "folder"),
)
# Per-frame baselines: divided space-time attention needs equal tokens per frame.
TIMESFORMER_BASELINE_MODES = (
    Mode("local", "random", "random"),
    Mode("local", "uniform", "uniform"),
)


def setting(name: str, default: str) -> str:
    return os.environ.get(name) or default


def flag(name: str, default: bool = False) -> bool:
    raw = setting(name, "1" if default else "0")
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{name} must be 0 or 1, got {raw!r}")
    return raw == "1"


def words(name: str, default: str) -> list[str]:
    values = setting(name, default).replace(",", " ").split()
    if not values:
        raise RuntimeError(f"{name} must contain at least one value")
    return values


def choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise RuntimeError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def run(arguments: list[str]) -> None:
    print(f"+ {shlex.join(arguments)}", flush=True)
    subprocess.run(arguments, cwd=PROJECT_ROOT, check=True)


def task_dir_name(task: str) -> str:
    """Directory a task's artifacts live in, under the model directory."""

    return task.replace("/", "-")


def summarize(results_root: Path, *, allow_env_override: bool = True) -> None:
    """Summarize every ``.pt`` under ``results_root`` into one CSV; the reader recurses."""

    timestamp = setting("RUN_TIMESTAMP", dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    output = results_root / f"results_summary_{timestamp}.csv"
    if allow_env_override:
        output = Path(setting("RESULTS_CSV", str(output)))
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "benchmark.reporting.summary",
            "--results-root",
            str(results_root),
            "--output",
            str(output),
        ]
    )


def boundary_tag(stage: str, prune_layer: str, *, include_stage: bool) -> str:
    if not prune_layer:
        return stage
    layer = f"neg{prune_layer[1:]}" if prune_layer.startswith("-") else prune_layer
    return f"{stage}_layer-{layer}" if include_stage else f"layer-{layer}"


def rate_tag(rate: str) -> str:
    return rate.replace(".", "p")


def _case_fingerprint(payload) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _valid_native_result(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("results"), dict)
        and bool(payload["results"])
    )


@contextlib.contextmanager
def _exclusive(native_dir: Path):
    """Hold the single writer slot for one arm, or report that it is taken."""

    native_dir.mkdir(parents=True, exist_ok=True)
    handle = (native_dir / ".writer.lock").open("a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN, errno.ENOLCK):
                raise
            if error.errno == errno.ENOLCK:
                # No lock manager, as on some network mounts; refuse rather than run unguarded.
                raise RuntimeError(
                    f"the filesystem holding {native_dir} cannot lock"
                ) from error
            handle.seek(0)
            print(f"Another process holds this arm, skipping: {native_dir}"
                  f" (owner: {handle.read().strip() or 'unknown'})")
            yield False
            return
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid {os.getpid()}\n")
        handle.flush()
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def _completed(marker: Path, result_path: Path, fingerprint: str) -> bool:
    if not marker.is_file() or not result_path.is_file():
        return False
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    native_results = manifest.get("native_results")
    if not isinstance(native_results, list) or not native_results:
        return False
    paths = [marker.parent / item["path"] for item in native_results]
    if not all(
        path.is_file()
        and path.stat().st_size == item["size"]
        and _valid_native_result(path)
        for path, item in zip(paths, native_results)
    ):
        return False
    return (
        manifest.get("schema_version") == 1
        and manifest.get("fingerprint") == fingerprint
        and manifest.get("result_size") == result_path.stat().st_size
        and manifest.get("complete") is True
    )


def _mark_completed(marker: Path, result_path: Path, fingerprint: str) -> None:
    native_paths = [
        path
        for path in sorted(marker.parent.rglob("*_results.json"))
        if _valid_native_result(path)
    ]
    if not native_paths:
        raise RuntimeError(
            "lmms-eval did not create a valid non-empty results table under: "
            f"{marker.parent}"
        )
    native_results = [
        {
            "path": str(path.relative_to(marker.parent)),
            "size": path.stat().st_size,
        }
        for path in native_paths
    ]
    payload = {
        "schema_version": 1,
        "native_results": native_results,
        "fingerprint": fingerprint,
        "result_size": result_path.stat().st_size,
        "complete": True,
    }
    atomic_write_text(marker, json.dumps(payload, sort_keys=True, indent=2))


def classification_sweep() -> None:
    model_type = choice(
        "MODEL_TYPE",
        setting("MODEL_TYPE", "vivit"),
        ("timesformer", "vivit"),
    )
    defaults = {
        "timesformer": {
            "model_name": "facebook/timesformer-base-finetuned-k400",
            "num_frames": "8",
            "prune_mode": "local",
            "i_mode": "folder",
        },
        "vivit": {
            "model_name": "google/vivit-b-16x2-kinetics400",
            "num_frames": "32",
            "prune_mode": "global",
            "i_mode": "preserve",
        },
    }[model_type]
    model_name = setting("MODEL_NAME", defaults["model_name"])
    num_frames = setting("NUM_FRAMES", defaults["num_frames"])
    prune_mode = choice(
        "PRUNE_MODE",
        setting("PRUNE_MODE", defaults["prune_mode"]),
        ("local", "global"),
    )
    i_mode = choice(
        "I_MODE",
        setting("I_MODE", defaults["i_mode"]),
        ("preserve", "folder", "global_folder"),
    )
    p_mode = choice(
        "P_MODE",
        setting("P_MODE", "hevc"),
        ("folder", "global_folder", "hevc", "shared_folding", "random", "uniform"),
    )
    run_mode = choice(
        "RUN_MODE",
        setting("RUN_MODE", "both"),
        ("pruned", "full", "both"),
    )
    stages = words("PRUNE_STAGE", "input")
    for stage in stages:
        choice("PRUNE_STAGE", stage, ("input", "last"))
    rates = [] if run_mode == "full" else words("K_KEEP_RATE", "0.5")
    prune_layer = os.environ.get("PRUNE_LAYER", "")
    suite = os.environ.get("SUITE", "")

    dataset = setting("DATASET", "k400")
    random_samples = os.environ.get("RANDOM_SAMPLES", "128")
    seed = setting("SEED", "42")
    batch_size = setting("BATCH_SIZE", "16")
    hevc_n_parallel = setting("HEVC_N_PARALLEL", "4")
    profiler = flag("ENABLE_PROFILER")
    skip_existing = flag("SKIP_EXISTING", True)

    checkpoint = setting("CKPT_NAME", model_name.rstrip("/").rsplit("/", 1)[-1])
    output_root = Path(
        setting(
            "OUTPUT_ROOT",
            str(CLF_RESULTS_ROOT / model_type),
        )
    )
    output_dir = Path(setting("OUTPUT_DIR", str(output_root / checkpoint)))
    profiler_root = Path(
        setting("PROFILER_ROOT", str(output_dir / "profiler"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    hevc_root = Path(
        setting("HEVC_DIR", str(output_dir / "hevc_tmp"))
    ).expanduser()
    hevc_root.mkdir(parents=True, exist_ok=True)
    hevc_permanent_dir = Path(
        setting("HEVC_PERMANENT_DIR", str(HEVC_STORE_DIR))
    ).expanduser()
    hevc_permanent_dir.mkdir(parents=True, exist_ok=True)
    base_args = [
        sys.executable,
        "-m",
        "benchmark.infer_clf",
        "--model-type",
        model_type,
        "--dataset",
        dataset,
        "--model-name",
        model_name,
        "--num-frames",
        num_frames,
        "--seed",
        seed,
        "--batch-size",
        batch_size,
        "--hevc-n-parallel",
        hevc_n_parallel,
        "--hevc-encode-scope",
        setting("HEVC_ENCODE_SCOPE", "sampled-clip"),
        "--hevc-gop-size",
        setting("HEVC_SAMPLED_GOP_SIZE", "0"),
        "--hevc-anchor-policy",
        setting("HEVC_ANCHOR_POLICY", "first"),
        "--hevc-permanent-dir",
        str(hevc_permanent_dir),
    ]
    if random_samples:
        base_args.extend(("--random-samples", random_samples))
    tubelet_size = os.environ.get("TUBELET_SIZE", "")
    patch_size = os.environ.get("PATCH_SIZE", "")
    if model_type == "vivit" and tubelet_size:
        base_args.extend(("--tubelet-size", tubelet_size))
    if model_type == "timesformer" and patch_size:
        base_args.extend(("--patch-size", patch_size))
    if profiler:
        base_args.append("--profiler")

    def execute_case(arguments: list[str], result_path: Path) -> None:
        if skip_existing and result_path.is_file():
            print(f"Skipping completed result: {result_path}")
            return
        run(
            [
                *base_args,
                *arguments,
                "--hevc-dir",
                str(hevc_root),
                "--results-path",
                str(result_path),
            ]
        )

    def regular_case(stage: str, mode: Mode, rate: str) -> None:
        boundary = boundary_tag(stage, prune_layer, include_stage=True)
        case_name = mode.name
        if mode.p_mode == "hevc":
            case_name += (
                f"_scope-{resolve_hevc_encode_scope(setting('HEVC_ENCODE_SCOPE', 'sampled-clip'))}"
                f"_gop-{setting('HEVC_SAMPLED_GOP_SIZE', '0')}"
                f"_anchor-{setting('HEVC_ANCHOR_POLICY', 'first')}"
            )
        if mode.p_mode == "random":
            case_name += f"_seed-{seed}"
        result_path = output_dir / (
            f"inference_{model_type}_pruned_{case_name}_{boundary}_"
            f"k-{rate_tag(rate)}_results.pt"
        )
        profiler_dir = (
            profiler_root / "pruned" / boundary / case_name / f"k-{rate_tag(rate)}"
        )
        arguments = [
            "--run-mode",
            "pruned",
            "--prune-stage",
            stage,
            "--prune-mode",
            mode.prune_mode,
            "--i-mode",
            mode.i_mode,
            "--p-mode",
            mode.p_mode,
            "--k-keep-rate",
            rate,
            "--profiler-dir",
            str(profiler_dir),
        ]
        if prune_layer:
            arguments.extend(("--prune-layer", prune_layer))
        execute_case(arguments, result_path)

    folding_mode_setting = setting("FOLDING_MODE", "temporal-diff")
    folding_modes = (
        list(FOLDING_MODES)
        if suite == "folding_modes"
        or folding_mode_setting == "all"
        else folding_mode_setting.split()
    )
    for folding_mode in folding_modes:
        choice("FOLDING_MODE", folding_mode, FOLDING_MODES)
    block_wise_setting = choice(
        "FOLD_BLOCK_WISE",
        setting("FOLD_BLOCK_WISE", "0"),
        ("0", "1", "all"),
    )
    block_wise_values = (
        ("0", "1") if block_wise_setting == "all" else (block_wise_setting,)
    )
    fold_block_size = setting("FOLD_BLOCK_SIZE", "2")
    fold_pooling = choice(
        "FOLD_POOLING",
        setting("FOLD_POOLING", "coverage-hard"),
        ("weighted", "representative", "coverage-hard"),
    )
    fold_slots = setting("FOLD_SLOTS_PER_BLOCK", "2")
    fold_min_slots = setting("FOLD_MIN_SLOTS_PER_BLOCK", "1")

    def shared_case(
        stage: str,
        folding_mode: str,
        block_wise: str,
        rate: str,
    ) -> None:
        boundary = boundary_tag(stage, prune_layer, include_stage=True)
        fold_tag = "global"
        arguments = [
            "--run-mode",
            "pruned",
            "--prune-stage",
            stage,
            "--shared-folding",
            "--folding-mode",
            folding_mode,
            "--k-keep-rate",
            rate,
            "--fold-block-size",
            fold_block_size,
            "--fold-pooling",
            fold_pooling,
            "--fold-slots-per-block",
            fold_slots,
            "--fold-min-slots-per-block",
            fold_min_slots,
        ]
        if block_wise == "1":
            fold_tag = (
                f"block-b{fold_block_size}-s{fold_slots}-"
                f"{fold_pooling}-min{fold_min_slots}"
            )
            arguments.append("--fold-block-wise")
        if prune_layer:
            arguments.extend(("--prune-layer", prune_layer))
        case_name = f"shared-folding_fold-{folding_mode.replace('_', '-')}_{fold_tag}"
        if folding_mode == "hevc-avg":
            case_name += (
                f"_scope-{resolve_hevc_encode_scope(setting('HEVC_ENCODE_SCOPE', 'sampled-clip'))}"
                f"_gop-{setting('HEVC_SAMPLED_GOP_SIZE', '0')}"
                f"_anchor-{setting('HEVC_ANCHOR_POLICY', 'first')}"
            )
        result_path = output_dir / (
            f"inference_timesformer_pruned_{case_name}_{boundary}_"
            f"k-{rate_tag(rate)}_results.pt"
        )
        arguments.extend(
            (
                "--profiler-dir",
                str(
                    profiler_root
                    / "pruned"
                    / boundary
                    / case_name
                    / f"k-{rate_tag(rate)}"
                ),
            )
        )
        execute_case(arguments, result_path)

    def shared_modes(stage: str) -> None:
        if model_type != "timesformer":
            raise RuntimeError("shared folding requires MODEL_TYPE=timesformer")
        for folding_mode in folding_modes:
            for block_wise in block_wise_values:
                selected_rates = rates[:1] if block_wise == "1" else rates
                for rate in selected_rates:
                    shared_case(stage, folding_mode, block_wise, rate)

    if run_mode in {"pruned", "both"}:
        for stage in stages:
            if not suite:
                if p_mode == "shared_folding":
                    shared_modes(stage)
                else:
                    mode = Mode(prune_mode, i_mode, p_mode)
                    for rate in rates:
                        regular_case(stage, mode, rate)
            elif suite == "vivit_modes":
                if model_type != "vivit":
                    raise RuntimeError(
                        f"SUITE={suite} requires MODEL_TYPE=vivit"
                    )
                for mode in VIVIT_MODES:
                    for rate in rates:
                        regular_case(stage, mode, rate)
            elif suite == "timesformer_modes":
                if model_type != "timesformer":
                    raise RuntimeError(
                        "SUITE=timesformer_modes requires "
                        "MODEL_TYPE=timesformer"
                    )
                for mode in TIMESFORMER_MODES:
                    for rate in rates:
                        regular_case(stage, mode, rate)
            elif suite == "folding_modes":
                shared_modes(stage)
            elif suite == "all_modes":
                modes = (
                    VIVIT_MODES + VIVIT_BASELINE_MODES
                    if model_type == "vivit"
                    else TIMESFORMER_MODES + TIMESFORMER_BASELINE_MODES
                )
                for mode in modes:
                    for rate in rates:
                        regular_case(stage, mode, rate)
                if model_type == "timesformer":
                    shared_modes(stage)
            else:
                raise RuntimeError(f"unknown classification SUITE={suite!r}")

    if run_mode in {"full", "both"}:
        result_path = output_dir / f"inference_{model_type}_full_results.pt"
        execute_case(
            [
                "--run-mode",
                "full",
                "--profiler-dir",
                str(profiler_root / "full"),
            ],
            result_path,
        )
    summarize(output_dir)


def vlm_sweep() -> None:
    task_names = words("TASKS", "nextqa_mc_test")
    limit = os.environ.get("LIMIT", "")
    sample_tag = f"limit-{limit}" if limit else "full-split"
    run_mode = choice(
        "RUN_MODE",
        setting("RUN_MODE", "pruned"),
        ("pruned", "full", "both"),
    )
    stages = words("PRUNE_STAGE", "last")
    for stage in stages:
        choice("PRUNE_STAGE", stage, ("input", "last"))
    rates = words("K_KEEP_RATE", "0.5")
    prune_layer = os.environ.get("PRUNE_LAYER", "")
    visual_grid_side = os.environ.get("VISUAL_GRID_SIDE", "")
    model_id = setting("MODEL_ID", "video_llava_hf")
    profile = MODEL_RUNS.get(model_id)
    if profile is None:
        raise RuntimeError(f"Unknown MODEL_ID={model_id!r}; choose from {tuple(MODEL_RUNS)}")
    default_mode = Mode(
        choice(
            "PRUNE_MODE",
            setting("PRUNE_MODE", profile.default_prune_mode),
            ("global", "local", "shared"),
        ),
        choice(
            "I_MODE",
            setting("I_MODE", profile.default_i_mode),
            # Baselines carry no anchor, so a single run is requestable.
            ("preserve", "global_folder", "folder", "shared_hevc", "shared_folding", "random", "uniform"),
        ),
        choice(
            "P_MODE",
            setting("P_MODE", profile.default_p_mode),
            ("hevc", "global_folder", "folder", "shared_hevc", "shared_folding", "random", "uniform"),
        ),
    )
    suite = os.environ.get("SUITE", "")
    if suite not in {"", "all_modes"}:
        raise RuntimeError(f"unknown VLM SUITE={suite!r}")

    default_num_frames = str(profile.default_num_frames)
    model_path = setting("MODEL_PATH", profile.default_pretrained)
    checkpoint = setting(
        "CKPT_NAME",
        profile.default_pretrained.rstrip("/").rsplit("/", 1)[-1],
    )
    output_root = Path(
        setting(
            "OUTPUT_ROOT",
            str(DEFAULT_VLM_RESULTS_ROOT),
        )
    )
    output_dir = Path(setting("OUTPUT_DIR", str(output_root / checkpoint)))
    hevc_dir = Path(setting("HEVC_DIR", str(DEFAULT_VLM_HEVC_DIR))).expanduser()
    hevc_permanent_dir = Path(
        setting("HEVC_PERMANENT_DIR", str(HEVC_STORE_DIR))
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    hevc_dir.mkdir(parents=True, exist_ok=True)
    hevc_permanent_dir.mkdir(parents=True, exist_ok=True)

    load_4bit = flag("LOAD_4BIT")
    load_8bit = flag("LOAD_8BIT")
    if load_4bit and load_8bit:
        raise RuntimeError("LOAD_4BIT and LOAD_8BIT cannot both be 1")
    log_samples = flag("LOG_SAMPLES", True)
    local_files_only = flag("LOCAL_FILES_ONLY")
    skip_existing = flag("SKIP_EXISTING", True)
    if not skip_existing:
        # Response replay keys on configuration alone, so disable it too.
        os.environ[RESUME_ENV] = "0"

    def execute_case(
        task: str,
        run_case_mode: str,
        mode: Mode,
        stage: str,
        case_name: str,
        rate: str,
    ) -> None:
        task_tag = task_dir_name(task)
        if mode.p_mode == "random":
            case_name += f"_seed-{setting('SEED', '42')}"
        if mode.p_mode == "shared_folding":
            case_name += (
                f"_fold-{setting('FOLDING_MODE', 'temporal-diff').replace('_', '-')}"
                f"_pool-{setting('FOLD_POOLING', 'coverage-hard').replace('_', '-')}"
                f"_b-{setting('FOLD_BLOCK_SIZE', '2')}"
            )
        hevc_active = bool(
            run_case_mode == "pruned"
            and (
                mode.p_mode in {"hevc", "shared_hevc"}
                or (
                    mode.p_mode == "shared_folding"
                    and setting("FOLDING_MODE", "temporal-diff") == "hevc-avg"
                )
            )
        )
        if hevc_active:
            case_name += (
                f"_scope-{resolve_hevc_encode_scope(setting('HEVC_ENCODE_SCOPE', 'sampled-clip'))}"
                f"_gop-{setting('HEVC_SAMPLED_GOP_SIZE', '0')}"
                f"_anchor-{setting('HEVC_ANCHOR_POLICY', 'first')}"
            )
        boundary = boundary_tag(stage, prune_layer, include_stage=False)
        # Artifact names use the lmms-eval model id.
        model_prefix = profile.model_id
        if run_case_mode == "full":
            rate = "1.0"
            output_name = f"{task_tag}_{model_prefix}_full_{sample_tag}"
        else:
            output_name = (
                f"{task_tag}_{model_prefix}_pruned_{case_name}_{boundary}_"
                f"k-{rate_tag(rate)}_{sample_tag}"
            )
        # One directory per task: the summary reader recurses into whatever it is given.
        task_dir = output_dir / task_tag
        native_dir = task_dir / "lmms_artifacts" / output_name
        marker = native_dir / ".complete.json"
        result_path = task_dir / f"{output_name}.pt"

        native_dir.mkdir(parents=True, exist_ok=True)
        device = setting("DEVICE", "cuda:0")
        arguments = [
            sys.executable,
            "-m",
            "benchmark.infer_vlm",
            "--model-id",
            model_id,
            "--model-path",
            model_path,
            "--tasks",
            task,
            "--batch-size",
            setting("BATCH_SIZE", "1"),
            "--device",
            device,
            "--output-path",
            str(native_dir),
            "--results-path",
            str(result_path),
            "--verbosity",
            setting("VERBOSITY", "INFO"),
            "--run-mode",
            run_case_mode,
            "--prune-stage",
            stage,
            "--prune-mode",
            mode.prune_mode,
            "--i-mode",
            mode.i_mode,
            "--p-mode",
            mode.p_mode,
            "--seed",
            setting("SEED", "42"),
            "--k-keep-rate",
            rate,
            "--num-frames",
            setting("NUM_FRAMES", default_num_frames),
            "--hevc-n-parallel",
            setting("HEVC_N_PARALLEL", "6"),
            "--hevc-encode-scope",
            setting("HEVC_ENCODE_SCOPE", "sampled-clip"),
            "--hevc-gop-size",
            setting("HEVC_SAMPLED_GOP_SIZE", "0"),
            "--hevc-anchor-policy",
            setting("HEVC_ANCHOR_POLICY", "first"),
            "--hevc-dir",
            str(hevc_dir),
            "--hevc-permanent-dir",
            str(hevc_permanent_dir),
            "--score-reduce",
            setting("SCORE_REDUCE", "max"),
            "--folding-mode",
            setting("FOLDING_MODE", "temporal-diff"),
            "--fold-block-size",
            setting("FOLD_BLOCK_SIZE", "2"),
            "--fold-pooling",
            setting("FOLD_POOLING", "coverage-hard"),
        ]
        if prune_layer:
            arguments.extend(("--prune-layer", prune_layer))
        if visual_grid_side:
            arguments.extend(("--visual-grid-side", visual_grid_side))
        if limit:
            arguments.extend(("--limit", limit))
        if log_samples:
            arguments.append("--log-samples")
        if load_4bit:
            arguments.append("--load-4bit")
        if load_8bit:
            arguments.append("--load-8bit")
        if local_files_only:
            arguments.append("--local-files-only")
        fingerprint_arguments = list(arguments)
        if not hevc_active:
            for option in (
                "--hevc-encode-scope",
                "--hevc-gop-size",
                "--hevc-anchor-policy",
            ):
                position = fingerprint_arguments.index(option)
                del fingerprint_arguments[position : position + 2]
        fingerprint = _case_fingerprint(
            {
                "arguments": fingerprint_arguments,
                "hevc_permanent_dir": str(hevc_permanent_dir),
                "model_dtype": inference_dtype_name(device),
                "decoding_policy": (
                    qwen3_vl_decoding_policy()
                    if model_id in {"qwen3_vl", "qwen3_vl_thinking"}
                    else None
                ),
                "hevc": (
                    {
                        "encoder": os.getenv("HEVC_ENCODER", ""),
                        "gop_size": os.getenv("GOP_SIZE", "8"),
                        "encode_scope": resolve_hevc_encode_scope(
                            os.getenv("HEVC_ENCODE_SCOPE", "sampled-clip")
                        ),
                        "sampled_gop_size": os.getenv("HEVC_SAMPLED_GOP_SIZE", "0"),
                        "anchor_policy": os.getenv("HEVC_ANCHOR_POLICY", "first"),
                        "signal_policy": HEVC_SIGNAL_POLICY,
                        "crf": os.getenv("CRF", "23"),
                        "min_dim": os.getenv("HEVC_MIN_DIM", "128"),
                    }
                    if hevc_active
                    else None
                ),
            }
        )
        if skip_existing and _completed(marker, result_path, fingerprint):
            print(f"Skipping completed result: {native_dir}")
            return
        with _exclusive(native_dir) as owned:
            if not owned:
                return
            # Whoever held the lock may have just finished this very case.
            if skip_existing and _completed(marker, result_path, fingerprint):
                print(f"Skipping completed result: {native_dir}")
                return
            run(arguments)
            if not result_path.is_file():
                raise RuntimeError(f"timing result was not created: {result_path}")
            _mark_completed(marker, result_path, fingerprint)

    cases = []
    if run_mode in {"pruned", "both"}:
        for stage in stages:
            modes = profile.modes if suite == "all_modes" else (default_mode,)
            for mode in modes:
                for rate in rates:
                    cases.append(("pruned", mode, stage, mode.name, rate))
    if run_mode in {"full", "both"}:
        cases.append(("full", default_mode, "last", "full", "1.0"))

    for task in task_names:
        for case in cases:
            execute_case(task, *case)
        summarize(
            output_dir / task_dir_name(task),
            allow_env_override=len(task_names) == 1,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("classification", "vlm"))
    args = parser.parse_args(argv)
    if args.benchmark == "classification":
        classification_sweep()
    else:
        vlm_sweep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
