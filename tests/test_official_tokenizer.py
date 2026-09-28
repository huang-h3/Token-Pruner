import json

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
import pytest
import sentencepiece as spm
import torch

from token_pruner.run_videollava_official import (
    IMAGE_TOKEN_INDEX,
    VideoLlavaOfficialBackend,
    tokenizer_image_token,
)


def test_prompt_chunks_use_the_checkpoint_sentencepiece_model(tmp_path, monkeypatch):
    corpus = [
        "USER: What happened in the video? ASSISTANT:",
        "Please count the number of repeated actions in the video.",
        "A. 7\nB. 2\nC. 1\nD. 4",
        "Answer with only the option letter.",
    ]
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus),
        model_prefix=str(tmp_path / "tokenizer"),
        model_type="bpe",
        vocab_size=128,
        hard_vocab_limit=False,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llava"}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "LlamaTokenizer",
        "legacy": False,
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    }))
    monkeypatch.setattr(
        "token_pruner.run_videollava_official.load_official_checkpoint",
        lambda *args, **kwargs: tuple(torch.nn.Identity() for _ in range(3)),
    )
    backend = VideoLlavaOfficialBackend(
        pretrained=str(tmp_path), device="cpu", local_files_only=True
    )
    processor = spm.SentencePieceProcessor(model_file=str(tmp_path / "tokenizer.model"))
    bos, image = [processor.bos_id()], [IMAGE_TOKEN_INDEX]
    question = "\nWhat happened? ASSISTANT:"
    cases = {
        "USER: " + "<image>" * 8 + question:
            bos + processor.encode("USER: ") + image * 8 + processor.encode(question),
        "USER: " + " ".join(["<image>"] * 8) + question:
            bos + processor.encode("USER: ")
            + (image + processor.encode(" ")) * 7 + image + processor.encode(question),
        " USER: " + "<image>" * 8 + " What happened? ASSISTANT:":
            bos + processor.encode(" USER: ") + image * 8
            + processor.encode(" What happened? ASSISTANT:"),
    }
    for prompt, expected in cases.items():
        assert tokenizer_image_token(prompt, backend.sentencepiece) == expected


GOLDEN_IDS = {
    "default": [
        1, 3148, 1001, 29901, 29871, -200, -200, -200, -200, -200, -200, -200, -200,
        29871, 13, 5618, 338, 10464, 297, 278, 4863, 29973, 319, 1799, 9047, 13566, 29901,
    ],
    "llava_v1": [
        1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082, 20255, 29889,
        450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278,
        5199, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 29871, -200, 259, -200, 259,
        -200, 259, -200, 259, -200, 259, -200, 259, -200, 259, -200, 29871, 13, 5618,
        338, 10464, 297, 278, 4863, 29973, 319, 1799, 9047, 13566, 29901,
    ],
}


@pytest.mark.parametrize("style", sorted(GOLDEN_IDS))
def test_prompts_match_the_official_slow_tokenizer(style, monkeypatch):
    try:
        path = hf_hub_download(
            "LanguageBind/Video-LLaVA-7B", "tokenizer.model",
            revision="aecae02b7dee5c249e096dcb0ce546eb6f811806", local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pytest.skip("LanguageBind/Video-LLaVA-7B tokenizer.model is not cached")
    monkeypatch.setenv("VIDEOLLAVA_OFFICIAL_PROMPT", style)
    prompt = VideoLlavaOfficialBackend._prompt_from_messages(
        [{"role": "user", "content": [{"type": "text", "text": "What is happening in the video?"}]}]
    )

    assert tokenizer_image_token(prompt, spm.SentencePieceProcessor(model_file=path)) == GOLDEN_IDS[style]
