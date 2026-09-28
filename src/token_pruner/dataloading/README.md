# OneVision-Encoder dataloading tools

This directory isolates the HEVC reader code adapted from `tools/tools_for_hevc/hevc_feature_decoder_mv.py` in [OneVision-Encoder](https://github.com/EvolvingLMMs-Lab/OneVision-Encoder), commit `54434a5461e6881b851ad04bbd8ca82edcf88c60`. The upstream code is licensed under Apache-2.0; see `LICENSE`.

| File | Origin status |
| --- | --- |
| `hevc_feature_decoder_mv.py` | Upstream derivative, trimmed to the feature reader used here and modified for this repository's decoder path |

The decoder binary is read from `HEVC_FEAT_DECODER` and defaults to `third_party/hevc_feature_decoder/bin/hevc` in a source checkout; see [third_party/README.md](../../../third_party/README.md).

The reader ships with the `token_pruner` package as `token_pruner.dataloading`. `token_pruner.hevc` imports it lazily, only when HEVC feature selection runs.
