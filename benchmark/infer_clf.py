"""Classification benchmark entry point."""

try:
    from . import bootstrap  # noqa: F401
except ImportError:  # Running the script by path.
    import bootstrap  # noqa: F401

import torch

from benchmark.classification.cli import (
    build_encoder_config,
    parse_args,
)
from benchmark.classification.datasets import build_dataset
from benchmark.classification.runner import run_classification_inference
from token_pruner.run_timesformer import load_timesformer
from token_pruner.run_vivit import load_vivit


def load_inference_session(args, device):
    loaders = {"timesformer": load_timesformer, "vivit": load_vivit}
    return loaders[str(args.model_type).lower()](build_encoder_config(args), device)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )
    device = torch.device("cuda")
    requested_model = args.model_name or f"default {args.model_type} checkpoint"
    print(
        f"Loading vision encoder: {args.model_type} -> {requested_model}...",
        flush=True,
    )
    session = load_inference_session(args, device)
    print(
        f"Loaded vision encoder: {session.model_type} -> "
        f"{session.model_name} on {session.device}.",
        flush=True,
    )
    print(
        f"Building dataset {args.dataset} with "
        f"random_samples={args.random_samples}...",
        flush=True,
    )
    dataset = build_dataset(
        args.dataset,
        model_config=session.model_config,
        random_samples=args.random_samples,
        seed=args.seed,
    )

    runner_options = {}
    if args.shared_folding:
        runner_options = {"summary_label": "Shared-folding"}
    return run_classification_inference(
        args,
        session,
        device,
        dataset=dataset,
        **runner_options,
    )


if __name__ == "__main__":
    main()
