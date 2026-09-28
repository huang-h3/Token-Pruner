# Benchmarks

Install the project in the environment used by the launchers, then run from the
project root.

- `python -m benchmark.infer_clf`: one classification configuration.
- `python -m benchmark.infer_vlm`: one VLM configuration through lmms-eval.
- `python -m benchmark.sweeps classification|vlm`: environment-driven experiment sweeps.
- `python -m benchmark.reporting.summary`: CSV metrics and measurements.

`reporting/io.py` reads response JSONL, native task metrics, and timing files.
Results are grouped by run, checkpoint, task, and configuration.
`results/` is the default output directory and is excluded from Git.

## Global FOLDER

Merge all frames once per sample:

```bash
MODEL_ID=internvl TASKS=nextqa_mc_test RUN_MODE=pruned \
PRUNE_STAGE=last PRUNE_MODE=global I_MODE=global_folder P_MODE=global_folder \
K_KEEP_RATE=0.5 NUM_FRAMES=32 BATCH_SIZE=1 LIMIT=16 \
FOLDER_ROOT=/path/to/Folder \
scripts/infer_vlm_hf.sh
```

Supported models: both Video-LLaVA variants, LLaVA-NeXT-Video, InternVL,
TimeSFormer, and ViViT. `SUITE=all_modes` includes this configuration for
supported VLMs. Classification uses
`--prune-mode global --i-mode global_folder --p-mode global_folder`.

## Evaluation options

These environment variables change what a VLM run generates or scores, and are
recorded with every response so a resumed run never mixes settings.

| Variable | Effect |
| --- | --- |
| `TOKEN_PRUNER_MAX_NEW_TOKENS` | Generation budget in place of the task's own |
| `TOKEN_PRUNER_GREEDY_DECODING=1` | Greedy decoding for every model |
| `TOKEN_PRUNER_STOP_STRINGS=1` | Stop generation at the task's `until` strings |
| `TOKEN_PRUNER_UNTRIMMED_SCORING=1` | Score the generation before `until` trimming |
| `TOKEN_PRUNER_REASONING_SEGMENT=answer` | Score only the text after `</think>` |
| `TOKEN_PRUNER_ANSWER_FORMAT_SUFFIX` | Text appended to multiple-choice prompts |
| `QWEN3_VL_DECODING_POLICY` | `task`, `greedy`, or `checkpoint` sampling for Qwen3-VL |
| `VIDEOLLAVA_OFFICIAL_PROMPT` | `default` or `llava_v1` prompt for the official checkpoint |
| `VIDEO_ABLATION` | `shuffle`, `static`, `noise`, or `black` input frames |

`TOKEN_PRUNER_RESUME_RESPONSES=0` disables replay of completed responses;
`SKIP_EXISTING=0` sets it for sweeps.

## Preencode full-video HEVC artifacts

`scripts/preencode_hevc.sh` loads lmms-eval task definitions and datasets, but no
VLM weights or external HEVC feature decoder. It uses the same task-group
expansion, per-leaf-task `LIMIT`, chat video paths and `HevcArtifactStore` as
inference. Video paths are deduplicated; identical content at different paths
also reuses the store's content-addressed artifacts. Existing hits are skipped,
missing artifacts are encoded, and failures produce a nonzero exit code.

```bash
TASKS='nextqa motionbench mvbench video_mmmu vitatecs' \
LIMIT=1024 HEVC_ENCODE_SCOPE=full-video HEVC_SAMPLED_GOP_SIZE=32 \
HEVC_ENCODER=hevc_nvenc \
scripts/preencode_hevc.sh --dry-run
```

Remove `--dry-run` to encode on a GPU node. `--report PATH` overrides the default
`STORE/preencode_summary.json`; a dry run only lists source videos, not cache
hit estimates. Re-run the same command after interruption to resume from the
artifact store. This command supports `full-video` with no video ablation;
`sampled-clip` needs model-specific preprocessing and is rejected.
Keep the encoder, GOP, CRF and other HEVC environment settings identical between
preencoding and inference. `NUM_FRAMES`, keep rate and prune stage do not affect
full-video artifact keys.

For the four-checkpoint last-stage matrix, submit from the project root on the
cluster login node:

```bash
bash scripts/submit_vlm_last.sh
```

This submits one shared GPU preencode job for the union of the four task lists,
then four inference arrays with `--dependency=afterok:<preencode-job-id>`.
Default settings: 64/8/8/16 frames, four keep rates, `LIMIT=1024`, batch size 8,
and up to four concurrent elements per array. `PROJECT_ROOT`, `CONDA_ENV`,
`HEVC_PERMANENT_DIR`, `LIMIT`, `CONCURRENT`, `OFFICIAL_MODEL_PATH`, and the four
`*_TASKS` variables may be overridden; `sbatch` reads `SBATCH_PARTITION` and the
other `SBATCH_*` variables itself. For NextQA multiple-choice only, use
`nextqa_mc_test` in each task list; `nextqa` also includes open-ended subtasks.

Preencoding prepares video artifacts, not `patch_scores/`. A patch-score miss
runs the feature decoder during inference. `shared_hevc` scores use
`anchor_policy=none`, so they are cached separately from selectors that use
`first`.
