# HEVC feature decoder

HEVC-signal pruning shells out to a patched OpenHEVC decoder that emits motion vectors and residuals per frame. That decoder is FFmpeg-derived and therefore LGPL-2.1-or-later, so it is not redistributed here. [OneVision-Encoder](https://github.com/EvolvingLMMs-Lab/OneVision-Encoder) publishes it as a prebuilt Linux x86-64 binary in `dataloader/decoder/bin/hevc`; commit `54434a5461e6881b851ad04bbd8ca82edcf88c60` is the revision this repository's reader follows. Download it into the default location from the project root:

```bash
mkdir -p third_party/hevc_feature_decoder/bin
curl -L -o third_party/hevc_feature_decoder/bin/hevc \
  https://raw.githubusercontent.com/EvolvingLMMs-Lab/OneVision-Encoder/54434a5461e6881b851ad04bbd8ca82edcf88c60/dataloader/decoder/bin/hevc
chmod +x third_party/hevc_feature_decoder/bin/hevc
export HEVC_FEAT_DECODER="$PWD/third_party/hevc_feature_decoder/bin/hevc"
```

The binary needs no build step. Its git blob hash is `5bdda1df0e5aa184beec482e9b738ae229539a4b` (`git hash-object third_party/hevc_feature_decoder/bin/hevc`).

`src/token_pruner/dataloading/hevc_feature_decoder_mv.py` reads `HEVC_FEAT_DECODER`, defaults to `third_party/hevc_feature_decoder/bin/hevc`, and fails with an explicit message when the binary is missing or not executable. Every non-HEVC selector (`folder`, `shared_folding`, `random`, `uniform`) and the full baseline run without it.

The decoder is invoked as a separate process, never linked, so its license does not propagate to this repository.
