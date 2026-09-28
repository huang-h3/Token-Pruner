from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from benchmark import infer_clf
from benchmark.classification.cli import build_encoder_config, parse_args
from token_pruner.classification_data import VisionEncoderConfig
from token_pruner.run_timesformer import (
    load_timesformer,
    resolve_timesformer_keep_budget,
)
from token_pruner.run_vivit import (
    VIVIT_LEGAL_MODES,
    load_vivit,
    resolve_vivit_keep_budget,
)
from token_pruner.tokens import SharedFoldingOptions


class StopBeforeModelLoad(RuntimeError):
    pass


def test_vivit_defaults_resolve_to_supported_modes_before_model_load():
    with (
        mock.patch(
            "token_pruner.run_vivit."
            "validate_pruning_modes"
        ) as validate,
        mock.patch(
            "token_pruner.run_vivit."
            "VivitConfig.from_pretrained",
            side_effect=StopBeforeModelLoad,
        ),
        pytest.raises(StopBeforeModelLoad),
    ):
        load_vivit(VisionEncoderConfig(), torch.device("cpu"))

    validate.assert_called_once_with(
        VIVIT_LEGAL_MODES,
        "vivit",
        "input",
        "global",
        "preserve",
        "hevc",
    )


def test_timesformer_shared_hevc_keeps_eight_model_frames():
    model_config = SimpleNamespace(
        num_frames=8,
        image_size=224,
        patch_size=16,
    )
    fake_model = SimpleNamespace(
        timesformer=SimpleNamespace(
            encoder=SimpleNamespace(layer=[object(), object()]),
        ),
    )
    fake_model.to = lambda _device: fake_model
    fake_model.eval = lambda: fake_model
    config = VisionEncoderConfig(
        num_frames=8,
        k_keep_rate=0.5,
        shared_folding=SharedFoldingOptions(mode="hevc-avg"),
        hevc_gop_size=16,
    )
    with (
        mock.patch(
            "token_pruner.run_timesformer."
            "TimesformerConfig.from_pretrained",
            return_value=model_config,
        ),
        mock.patch(
            "token_pruner.run_timesformer."
            "AutoImageProcessor.from_pretrained",
            return_value=object(),
        ),
        mock.patch(
            "token_pruner.run_timesformer."
            "TimesformerForVideoClassification.from_pretrained",
            return_value=fake_model,
        ),
    ):
        session = load_timesformer(config, torch.device("cpu"))

    assert session.num_frames == 8
    assert session.model_config.num_frames == 8
    assert session.selector.hevc_gop_size == 16


def test_classification_cli_routes_hevc_encode_scope():
    args = parse_args(["--hevc-encode-scope", "full-video"])
    config = build_encoder_config(args)

    assert args.hevc_encode_scope == "full-video"
    assert config.hevc_encode_scope == "full-video"


@pytest.mark.parametrize(
    "arguments",
    (
        ("--i-mode", "hevc"),
        ("--p-mode", "preserve"),
    ),
)
def test_classification_cli_rejects_modes_not_supported_by_src(arguments):
    with pytest.raises(SystemExit):
        parse_args(list(arguments))


def test_budget_resolution_is_terminal_silent(capsys):
    config = VisionEncoderConfig(k_keep_rate=0.5)

    assert resolve_timesformer_keep_budget(config, 5, 3) == (
        2,
        7,
        6,
        0.4,
    )
    assert resolve_vivit_keep_budget(config, 15, 6, 2) == (
        7,
        2,
        15,
        5,
    )
    assert capsys.readouterr().out == ""


def test_classification_entrypoint_owns_encoder_loading_output(capsys):
    args = SimpleNamespace(
        model_type="timesformer",
        model_name=None,
        dataset="k400",
        random_samples=4,
        seed=42,
        shared_folding=False,
    )
    session = SimpleNamespace(
        model_type="timesformer",
        model_name="facebook/timesformer-test",
        model_config=object(),
        device=torch.device("cuda"),
    )
    dataset = object()

    with (
        mock.patch.object(infer_clf, "parse_args", return_value=args),
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(
            infer_clf,
            "load_inference_session",
            return_value=session,
        ),
        mock.patch.object(infer_clf, "build_dataset", return_value=dataset),
        mock.patch.object(
            infer_clf,
            "run_classification_inference",
            return_value="result.pt",
        ) as run,
    ):
        assert infer_clf.main([]) == "result.pt"

    output = capsys.readouterr().out
    assert (
        "Loading vision encoder: timesformer -> default timesformer checkpoint..."
        in output
    )
    assert (
        "Loaded vision encoder: timesformer -> "
        "facebook/timesformer-test on cuda."
        in output
    )
    run.assert_called_once()
