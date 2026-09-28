from unittest import mock

from benchmark.infer_vlm import (
    build_parser,
    lmms_arguments,
    main,
    model_arguments,
)

from benchmark.paths import VLM_HEVC_DIR

BASE_ARGUMENTS = (
    "--tasks",
    "nextqa_mc_test",
    "--output-path",
    "/tmp/lmms-output",
    "--results-path",
    "/tmp/timing.pt",
)


def test_vlm_entrypoint_translates_explicit_model_arguments():
    args = build_parser().parse_args(
        (
            *BASE_ARGUMENTS,
            "--model-path",
            "LanguageBind/Video-LLaVA-7B-hf",
            "--run-mode",
            "pruned",
            "--prune-stage",
            "last",
            "--prune-layer",
            "7",
            "--prune-mode",
            "global",
            "--i-mode",
            "preserve",
            "--p-mode",
            "hevc",
            "--k-keep-rate",
            "0.25",
            "--batch-size",
            "2",
            "--log-samples",
            "--load-4bit",
        )
    )
    assert args.hevc_dir == VLM_HEVC_DIR
    assert args.num_frames is None
    assert args.hevc_encode_scope == "sampled-clip"

    model_args = model_arguments(args)
    assert "pretrained=LanguageBind/Video-LLaVA-7B-hf" in model_args
    assert "prune_layer=7" in model_args
    assert "k_keep_rate=0.25" in model_args
    assert "num_frames=" not in model_args
    assert "batch_size=" not in model_args
    assert "load_4bit=1" in model_args
    assert "hevc_encode_scope=sampled-clip" in model_args
    assert "results_path=/tmp/timing.pt" in model_args

    lmms_args = lmms_arguments(args)
    assert lmms_args[lmms_args.index("--model") + 1] == "video_llava_hf"
    assert lmms_args[lmms_args.index("--tasks") + 1] == "nextqa_mc_test"
    assert lmms_args[lmms_args.index("--output_path") + 1] == "/tmp/lmms-output"
    assert lmms_args[lmms_args.index("--batch_size") + 1] == "2"
    assert "--log_samples" in lmms_args


def test_vlm_main_resolves_plugin_and_calls_lmms():
    with (
        mock.patch("benchmark.infer_vlm.resolve_chat_model") as resolve,
        mock.patch("benchmark.infer_vlm.run_lmms") as run,
    ):
        assert main(BASE_ARGUMENTS) == 0

    resolve.assert_called_once_with("video_llava_hf")
    forwarded = run.call_args.args[0]
    assert forwarded[forwarded.index("--model") + 1] == "video_llava_hf"
    assert forwarded[forwarded.index("--tasks") + 1] == "nextqa_mc_test"


def test_vlm_main_forwards_batch_size_to_lmms():
    with (
        mock.patch("benchmark.infer_vlm.resolve_chat_model") as resolve,
        mock.patch("benchmark.infer_vlm.run_lmms") as run,
    ):
        assert main((*BASE_ARGUMENTS, "--batch-size", "2")) == 0

    resolve.assert_called_once_with("video_llava_hf")
    forwarded = run.call_args.args[0]
    assert forwarded[forwarded.index("--batch_size") + 1] == "2"
