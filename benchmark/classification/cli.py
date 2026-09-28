"""CLI translation for classification experiments."""

import argparse
import os

from token_pruner.classification_data import VisionEncoderConfig
from token_pruner.hevc import resolve_hevc_encode_scope
from token_pruner.tokens import SharedFoldingOptions
from token_pruner.shared_folding import (
    FOLDING_MODES,
    FOLDING_POOLING_MODES,
)

from benchmark.paths import CLF_HEVC_DIR, HEVC_STORE_DIR, RESULTS_ROOT

from .data import DEFAULT_DATASET
from .datasets import DATASET_NAMES


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=("timesformer", "vivit"),
        default="timesformer",
    )
    parser.add_argument("--dataset", choices=DATASET_NAMES, default=DEFAULT_DATASET)
    parser.add_argument("--random-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--k-keep-rate", type=float, default=0.5)
    parser.add_argument(
        "--k-keep",
        type=int,
        default=None,
        help="Optional explicit per-frame patch count for TimeSFormer.",
    )
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--tubelet-size", type=int, default=None)
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
        "--hevc-dir",
        default=os.getenv("HEVC_DIR", str(CLF_HEVC_DIR)),
        help="Per-job HEVC work directory (env: HEVC_DIR).",
    )
    parser.add_argument(
        "--hevc-permanent-dir",
        default=os.getenv("HEVC_PERMANENT_DIR", str(HEVC_STORE_DIR)),
        help="Shared HEVC artifact store (env: HEVC_PERMANENT_DIR).",
    )
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--skipped-videos-path", default=None)
    parser.add_argument(
        "--run-mode",
        choices=("pruned", "full", "both"),
        default="pruned",
    )
    parser.add_argument("--prune-mode", choices=("global", "local"), default=None)
    parser.add_argument("--prune-stage", choices=("input", "last"), default="input")
    parser.add_argument(
        "--prune-layer",
        type=int,
        default=None,
        help=(
            "Encoder boundary to prune at; 0 is before the first layer, "
            "-1 before the last, and -2 before the penultimate layer."
        ),
    )
    parser.add_argument(
        "--i-mode",
        choices=("preserve", "global_folder", "folder", "random", "uniform"),
        default=None,
    )
    parser.add_argument(
        "--p-mode",
        choices=("hevc", "global_folder", "folder", "random", "uniform"),
        default=None,
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument(
        "--profiler-dir",
        default=str(RESULTS_ROOT / "profiler_logs"),
    )

    folding = parser.add_argument_group("TimeSFormer shared folding")
    folding.add_argument(
        "--shared-folding",
        action="store_true",
        help="Use TimeSFormer shared folding instead of regular token pruning.",
    )
    folding.add_argument(
        "--folding-mode",
        choices=FOLDING_MODES,
        default="temporal-diff",
    )
    folding.add_argument("--fold-block-wise", action="store_true", default=False)
    folding.add_argument("--fold-block-size", type=int, default=2)
    folding.add_argument(
        "--fold-pooling",
        choices=FOLDING_POOLING_MODES,
        default="coverage-hard",
    )
    folding.add_argument("--fold-slots-per-block", type=int, default=2)
    folding.add_argument("--fold-min-slots-per-block", type=int, default=1)
    return parser


def validate_args(args):
    args.hevc_encode_scope = resolve_hevc_encode_scope(args.hevc_encode_scope)
    if args.hevc_gop_size < 0:
        raise RuntimeError("--hevc-gop-size must be non-negative.")
    if args.batch_size <= 0:
        raise RuntimeError(f"batch-size must be positive, got {args.batch_size}.")
    if args.random_samples is not None and args.random_samples <= 0:
        raise RuntimeError(
            f"random-samples must be positive, got {args.random_samples}."
        )
    if args.hevc_n_parallel <= 0:
        raise RuntimeError(
            f"hevc-n-parallel must be positive, got {args.hevc_n_parallel}."
        )
    if args.shared_folding and args.model_type != "timesformer":
        raise RuntimeError("--shared-folding requires --model-type timesformer.")
    if args.shared_folding and args.run_mode == "full":
        raise RuntimeError(
            "--shared-folding requires --run-mode pruned or both."
        )
    return args


def parse_args(argv=None):
    args = validate_args(build_parser().parse_args(argv))
    if args.results_path is None:
        filename = (
            "inference_shared_folding_results.pt"
            if args.shared_folding
            else "inference_results.pt"
        )
        args.results_path = str(RESULTS_ROOT / filename)
    return args


def build_encoder_config(args):
    shared_folding = None
    if args.shared_folding:
        shared_folding = SharedFoldingOptions(
            mode=args.folding_mode,
            block_size=args.fold_block_size,
            slots_per_block=args.fold_slots_per_block,
            pooling=args.fold_pooling,
            min_slots_per_block=args.fold_min_slots_per_block,
            block_wise=args.fold_block_wise,
            global_k=None,
        )
    return VisionEncoderConfig(
        model_name=args.model_name,
        run_mode=args.run_mode,
        num_frames=args.num_frames,
        k_keep_rate=args.k_keep_rate,
        k_keep=args.k_keep,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        hevc_gop_size=args.hevc_gop_size,
        hevc_encode_scope=args.hevc_encode_scope,
        hevc_anchor_policy=args.hevc_anchor_policy,
        prune_mode=args.prune_mode,
        prune_stage=args.prune_stage,
        prune_layer=args.prune_layer,
        i_mode=args.i_mode,
        p_mode=args.p_mode,
        seed=args.seed,
        shared_folding=shared_folding,
    )
