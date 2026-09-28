# Token-Pruner

Token pruning for video vision transformers and vision-language models. VLM evaluation runs on [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and token merging uses the external [FOLDER](https://github.com/anakin-skywalker-Joseph/Folder) implementation.

## Models and pruning

| Model ID | Default checkpoint | Project default frames |
| --- | --- | ---: |
| `video_llava_hf` | `LanguageBind/Video-LLaVA-7B-hf` | 8 |
| `video_llava_official` | `LanguageBind/Video-LLaVA-7B` | 8 |
| `llava_next_video` | `llava-hf/LLaVA-NeXT-Video-7B-hf` | 8 |
| `qwen3_vl` | `Qwen/Qwen3-VL-8B-Instruct` | 64 |
| `qwen3_vl_thinking` | `Qwen/Qwen3-VL-8B-Thinking` | 64 |
| `internvl` | `OpenGVLab/InternVL3_5-8B-Instruct` | 8 |
| `llava_onevision2` | `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` | 32 |

`NUM_FRAMES` overrides the project defaults. InternVL supports variable frame counts; use `NUM_FRAMES=32` for a 32-frame experiment.

TimeSFormer and ViViT support video classification.

### Supported pruning configurations

All rows support `PRUNE_STAGE=input` and `PRUNE_STAGE=last`. `PRUNE_LAYER` selects an explicit encoder boundary.

| Model ID | preserve / hevc | folder / hevc | preserve / folder | folder / folder | Shared HEVC | Shared folding | Random / Uniform | Global FOLDER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `video_llava_hf` | local, global | local, global | local, global | local | shared | shared | local, global | global |
| `video_llava_official` | — | local | — | local | shared | shared | local | global |
| `llava_next_video` | local, global | local, global | local, global | local | shared | shared | local, global | global |
| `qwen3_vl` | local, global | — | — | — | shared | — | local, global | — |
| `qwen3_vl_thinking` | local, global | — | — | — | shared | — | local, global | — |
| `internvl` | local, global | local, global | local, global | local | shared | shared | local, global | global |
| `llava_onevision2` | local, global | — | — | — | shared | — | local, global | — |
| `timesformer` | — | local | — | local | — | shared | local | global |
| `vivit` | local, global | local, global | local, global | local | — | — | global | global |

Cells list supported scopes; `—` marks an unsupported configuration. The four paired columns specify `I_MODE / P_MODE`: the operation on anchor and other frame or tubelet groups. `preserve` keeps tokens, `hevc` selects by HEVC scores, and `folder` merges tokens with FOLDER.

For VLM launchers:

- Set `PRUNE_MODE` to a listed scope and `I_MODE` / `P_MODE` to the column pair.
- Global FOLDER: `PRUNE_MODE=global I_MODE=global_folder P_MODE=global_folder`.
- Shared HEVC: `PRUNE_MODE=shared I_MODE=shared_hevc P_MODE=shared_hevc`.
- Shared folding: `PRUNE_MODE=shared I_MODE=shared_folding P_MODE=shared_folding`; `FOLDING_MODE=temporal-diff` selects temporal-difference folding.
- Random or uniform: set both `I_MODE` and `P_MODE` to `random` or `uniform`.

`global_folder` runs FOLDER once per sample over all frames, using one total budget and zero anchor frames. CLS tokens supply the merge metric and remain outside the merge. InternVL and LLaVA-NeXT merge complete 2×2 cells. Output slots are distributed across temporal groups for downstream layers; merged tokens can contain features from any frame. TimeSFormer and official Video-LLaVA round the budget to equal group sizes. Set `FOLDER_ROOT` to the reference implementation; this mode uses no HEVC preprocessing.

Classification uses `--prune-stage`, `--prune-mode`, `--i-mode`, and `--p-mode`. Enable TimeSFormer shared folding with `--shared-folding`; its random and uniform baselines use local selection. ViViT baselines use global selection.

InternVL 3.5 uses complete 2×2 pixel-shuffle cells and the checkpoint's `chat` prompt, which is non-thinking by default; its `last` stage prunes before the final InternViT block. OneVision-2's `last` stage selects encoder output before the merger. Qwen3-VL selects a boundary compatible with its DeepStack features.

## Installation

Tested with Python 3.11, PyTorch 2.13.0, torchvision 0.28.0, Transformers 5.16.1, and lmms-eval commit `b485e66`. Run from the project root:

```bash
conda create -n token-pruner python=3.11
conda activate token-pruner
python -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
conda install -c conda-forge "ffmpeg=8.0.1=gpl*"

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git ../lmms-eval
git -C ../lmms-eval checkout b485e662a95aa9c62616e9b48bb085442422a928
python -m pip install -e ../lmms-eval
python -m pip install "transformers==5.16.1" "numpy==2.4.6" "accelerate==1.14.0" \
  "bitsandbytes==0.49.2" "decord==0.6.0" "einops==0.8.2" "pyarrow==25.0.0" \
  safetensors tqdm pytest
python -m pip install -e .
```

Choose PyTorch wheels compatible with your GPU driver. The editable project installation registers the lmms-eval model plugin.

The plugin registers every model ID in the table above. `qwen3_vl`, `internvl`, and `llava_onevision2` are also built-in lmms-eval models; with Token-Pruner installed, these IDs resolve to Token-Pruner's implementations instead of lmms-eval's own.

InternVL and OneVision-2 load checkpoint code with `trust_remote_code=True`.

### Resources and configuration

- HEVC encoding uses FFmpeg and ffprobe. `hevc_nvenc` requires an NVENC-capable GPU; `libx265` uses the CPU.
- HEVC selection and `FOLDING_MODE=hevc-avg` require the external [feature decoder](third_party/README.md). Set `HEVC_FEAT_DECODER` to its binary.
- FOLDER merging requires `$FOLDER_ROOT/folder.py`.
- Model weights and task datasets must be cached or downloadable. `LOCAL_FILES_ONLY=1` restricts model loading to local files.
- InternVL uses eager attention without flash-attn, as its checkpoint code does; an unpruned 32-frame pass does not fit in 48 GB, so run it with `BATCH_SIZE=1` on a larger GPU.

Set paths for your machine:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export LMMS_EVAL_DATASETS_CACHE="$HOME/.cache/lmms_eval_datasets"
export HEVC_PERMANENT_DIR="$PWD/benchmark/results/hevc_cache/store"
export HEVC_ENCODER=hevc_nvenc
```

Launchers use the active environment. Set `CONDA_ENV` to activate another environment and `CONDA_SH` to its `conda.sh` path when needed. SLURM submissions inherit exported variables; `SBATCH_PARTITION` selects the partition.

For tasks requiring NLTK data, download the resources before offline execution:

```bash
python -c "import nltk; [nltk.download(r) for r in ('wordnet', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng')]"
```

## Inference

This GPU smoke test evaluates two NExT-QA samples with InternVL:

```bash
MODEL_ID=internvl MODEL_PATH=OpenGVLab/InternVL3_5-8B-Instruct \
TASKS=nextqa_mc_test RUN_MODE=pruned \
PRUNE_STAGE=last PRUNE_MODE=global I_MODE=uniform P_MODE=uniform \
K_KEEP_RATE=0.5 NUM_FRAMES=2 BATCH_SIZE=1 LIMIT=2 \
TOKEN_PRUNER_MAX_NEW_TOKENS=16 \
OUTPUT_DIR="$PWD/benchmark/results/example" \
scripts/infer_vlm_hf.sh
```

Choose `MODEL_ID` and `MODEL_PATH` from the table, or use a local checkpoint path.

- `RUN_MODE=full` runs the baseline; `RUN_MODE=both` runs both arms.
- Space-separated `K_KEEP_RATE` values create a sweep.
- `SUITE=all_modes` expands supported pruning configurations.
- `LIMIT` applies to each leaf task in a task group.
- `RUN_ON_CLUSTER=1` submits through `sbatch`.

For classification, use `scripts/infer_clf.sh` or `python -m benchmark.infer_clf --help`. The scripts `k400val.sh` and `k400test.sh` download Kinetics-400 into `data/kinetics400`.

## HEVC preprocessing

```bash
TASKS=nextqa_mc_test LIMIT=16 \
HEVC_ENCODE_SCOPE=full-video HEVC_SAMPLED_GOP_SIZE=32 \
scripts/preencode_hevc.sh --dry-run
```

Remove `--dry-run` to encode missing videos into the shared store. Matching artifacts are reused. Full-video artifacts share an encoding across frame counts, pruning stages, and keep rates. Feature extraction uses the decoder during HEVC-based pruning.

Set `HEVC_ENCODE_SCOPE=full-video` for experiments using these artifacts. The inference default is `sampled-clip`.

`scripts/submit_vlm_last.sh` submits one preencoding job and four dependent arrays for Qwen3-VL, both Video-LLaVA variants, and LLaVA-NeXT-Video. Configure its task lists, model paths, and SLURM resources, then run:

```bash
bash scripts/submit_vlm_last.sh
```

See [benchmark commands](benchmark/README.md) for all evaluation options.

## Results

Each VLM arm writes lmms-eval metrics, timing measurements in `.pt`, and responses in `.responses.jsonl`. `SKIP_EXISTING=1` reuses completed arms whose configuration and output files match.

```bash
python -m benchmark.reporting.summary \
  --results-root benchmark/results/example \
  --output benchmark/results/example/summary.csv
```

`hevc_encode_ms` records the encoding cost attributed to a sample; `hevc_encode_paid_ms` records work performed during this run and is zero on a cache hit. CSV columns containing `__batch__` describe batch measurements.

## Development

```bash
python -m pytest -q
python -c "from lmms_eval.models import MODEL_REGISTRY_V2; print(MODEL_REGISTRY_V2.resolve('qwen3_vl').class_path)"
ffmpeg -hide_banner -encoders | grep -E 'libx265|hevc_nvenc'
```

Tests use synthetic inputs. GPU and FOLDER tests require their respective resources, and the official Video-LLaVA tokenizer test reads the cached `LanguageBind/Video-LLaVA-7B` tokenizer and is skipped without it. `LMMS_EVAL_ROOT` points tests and launchers to an lmms-eval checkout.

- `src/token_pruner/`: model execution, pruning, caches, measurements, and the HEVC feature reader
- `benchmark/`: inference, sweeps, and summaries
- `scripts/`: local and SLURM launchers
- `tools/`: cache inspection
- `tests/`: model and runtime tests

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE). Checkpoints, datasets, and external dependencies have their own licenses. HEVC reader attribution is in [src/token_pruner/dataloading](src/token_pruner/dataloading/README.md).
