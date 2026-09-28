#!/usr/bin/env python3
"""Video-language benchmark entry point backed by lmms-eval."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

try:  # puts src/ on sys.path when the package is not installed
    from . import bootstrap  # noqa: F401
except ImportError:  # Running the script by path.
    import bootstrap  # noqa: F401

from benchmark.paths import HEVC_STORE_DIR, VLM_HEVC_DIR
from token_pruner.task_io import GREEDY_DECODING_ENV
from token_pruner.hevc import resolve_hevc_encode_scope


DEFAULT_MODEL_ID = "video_llava_hf"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one VLM benchmark case through lmms-eval."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--results-path", required=True, type=Path)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verbosity", default="INFO")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--log-samples", action="store_true")

    parser.add_argument(
        "--run-mode",
        choices=("pruned", "full"),
        default="pruned",
    )
    parser.add_argument(
        "--prune-stage",
        choices=("input", "last"),
        default="last",
    )
    parser.add_argument(
        "--prune-layer",
        type=int,
        default=None,
        help=(
            "Encoder boundary to prune at; -1 is before the last layer "
            "and -2 before the penultimate layer."
        ),
    )
    parser.add_argument(
        "--prune-mode",
        choices=("global", "local", "shared"),
        default=None,
    )
    # Shared scope treats every frame alike, so i_mode repeats p_mode.
    parser.add_argument(
        "--i-mode",
        choices=("preserve", "global_folder", "folder", "shared_hevc", "shared_folding",
                 "random", "uniform"),
        default=None,
    )
    parser.add_argument(
        "--p-mode",
        choices=("hevc", "global_folder", "folder", "shared_hevc", "shared_folding",
                 "random", "uniform"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-keep-rate", type=float, default=0.5)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--hevc-n-parallel", type=int, default=6)
    parser.add_argument(
        "--hevc-encode-scope",
        choices=("sampled-clip", "full-video"),
        default=os.getenv("HEVC_ENCODE_SCOPE", "sampled-clip"),
    )
    parser.add_argument(
        "--hevc-gop-size",
        type=int,
        default=int(os.getenv("HEVC_SAMPLED_GOP_SIZE", "0")),
    )
    parser.add_argument(
        "--hevc-anchor-policy", choices=("first", "all"), default=os.getenv("HEVC_ANCHOR_POLICY", "first")
    )
    parser.add_argument(
        "--hevc-dir", type=Path, default=Path(os.getenv("HEVC_DIR", str(VLM_HEVC_DIR))),
        help="Per-job HEVC work directory (env: HEVC_DIR).",
    )
    parser.add_argument(
        "--hevc-permanent-dir", type=Path,
        default=Path(os.getenv("HEVC_PERMANENT_DIR", str(HEVC_STORE_DIR))),
        help="Shared HEVC artifact store (env: HEVC_PERMANENT_DIR).",
    )
    parser.add_argument("--score-reduce", default="max")
    parser.add_argument("--folding-mode", default="temporal-diff")
    parser.add_argument("--fold-block-size", type=int, default=2)
    parser.add_argument("--fold-pooling", default="coverage-hard")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help=f"Force greedy decoding (same as {GREEDY_DECODING_ENV}=1).",
    )
    parser.add_argument(
        "--visual-grid-side",
        type=int,
        default=None,
        help=(
            "Merged cells per frame side for towers with dynamic resolution "
            "(Qwen3-VL). Ignored by fixed-resolution backends."
        ),
    )

    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--load-4bit", action="store_true")
    quantization.add_argument("--load-8bit", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.hevc_encode_scope = resolve_hevc_encode_scope(args.hevc_encode_scope)
    if args.hevc_gop_size < 0:
        raise RuntimeError("--hevc-gop-size must be non-negative.")
    if int(args.batch_size) <= 0:
        raise RuntimeError("--batch-size must be positive.")
    if args.greedy:
        os.environ[GREEDY_DECODING_ENV] = "1"
    return args


def model_arguments(args: argparse.Namespace) -> str:
    values = (
        ("pretrained", args.model_path),
        ("run_mode", args.run_mode),
        ("prune_stage", args.prune_stage),
        ("load_4bit", int(args.load_4bit)),
        ("load_8bit", int(args.load_8bit)),
        ("num_frames", args.num_frames),
        ("prune_mode", args.prune_mode),
        ("i_mode", args.i_mode),
        ("p_mode", args.p_mode),
        ("selection_seed", args.seed),
        ("k_keep_rate", args.k_keep_rate),
        ("hevc_n_parallel", args.hevc_n_parallel),
        ("hevc_gop_size", args.hevc_gop_size),
        ("hevc_encode_scope", args.hevc_encode_scope),
        ("hevc_anchor_policy", args.hevc_anchor_policy),
        ("hevc_dir", args.hevc_dir.expanduser()),
        ("hevc_permanent_dir", args.hevc_permanent_dir.expanduser()),
        ("score_reduce", args.score_reduce),
        ("folding_mode", args.folding_mode),
        ("fold_block_size", args.fold_block_size),
        ("fold_pooling", args.fold_pooling),
        ("local_files_only", int(args.local_files_only)),
        ("results_path", args.results_path.expanduser()),
    )
    parts = [f"{name}={value}" for name, value in values if value is not None]
    if args.prune_layer is not None:
        parts.append(f"prune_layer={args.prune_layer}")
    if args.visual_grid_side is not None:
        parts.append(f"visual_grid_side={args.visual_grid_side}")
    return ",".join(parts)


def lmms_arguments(args: argparse.Namespace) -> list[str]:
    arguments = [
        "--model",
        args.model_id,
        "--model_args",
        model_arguments(args),
        "--tasks",
        args.tasks,
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--output_path",
        str(args.output_path.expanduser()),
        "--verbosity",
        args.verbosity,
    ]
    if args.limit is not None:
        arguments.extend(("--limit", str(args.limit)))
    if args.log_samples:
        arguments.append("--log_samples")
    return arguments


def resolve_chat_model(model_id: str) -> None:
    from lmms_eval.models import MODEL_REGISTRY_V2

    resolved = MODEL_REGISTRY_V2.resolve(model_id)
    print(f"lmms model: {resolved.model_id} -> {resolved.class_path}")


def run_lmms(arguments: Sequence[str]) -> None:
    from lmms_eval.__main__ import cli_evaluate

    original_argv = sys.argv
    try:
        sys.argv = ["lmms_eval", *arguments]
        cli_evaluate()
    finally:
        sys.argv = original_argv


def main(argv: Sequence[str] | None = None) -> int:
    args = validate_args(build_parser().parse_args(argv))
    resolve_chat_model(args.model_id)
    run_lmms(lmms_arguments(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
