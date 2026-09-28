#!/usr/bin/env bash
#SBATCH --job-name=clf-matrix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Classification pruning matrix, one case per array element:
#   sbatch --array=0-17 scripts/clf_matrix.sh
set -euo pipefail

# idx: MODEL_TYPE STAGE KIND FOLDING_MODE SLOTS RATE
CASES=(
  "vivit        input all_modes   -             -  -"
  "vivit        last  all_modes   -             -  -"
  "timesformer  input all_modes   -             -  -"
  "timesformer  last  all_modes   -             -  -"
  # Shared folding with temporal-diff scores, 2x2 blocks: rate = slots / 4.
  "timesformer  input fold_block  temporal-diff 1  0.25"
  "timesformer  input fold_block  temporal-diff 2  0.5"
  "timesformer  input fold_block  temporal-diff 3  0.75"
  "timesformer  last  fold_block  temporal-diff 1  0.25"
  "timesformer  last  fold_block  temporal-diff 2  0.5"
  "timesformer  last  fold_block  temporal-diff 3  0.75"
  # Global shared folding needs an integer 196 * rate, so no 0.125.
  "timesformer  input fold_global temporal-diff -  -"
  "timesformer  last  fold_global temporal-diff -  -"
  # Shared folding with HEVC scores, 2x2 blocks.
  "timesformer  input fold_block  hevc-avg      1  0.25"
  "timesformer  input fold_block  hevc-avg      2  0.5"
  "timesformer  input fold_block  hevc-avg      3  0.75"
  "timesformer  last  fold_block  hevc-avg      1  0.25"
  "timesformer  last  fold_block  hevc-avg      2  0.5"
  "timesformer  last  fold_block  hevc-avg      3  0.75"
)
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Submit as an array job: sbatch --array=0-$((${#CASES[@]} - 1)) $0" >&2
  exit 2
fi
read -r MT STAGE KIND FMODE SLOTS RATE <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-"$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"}}
source "$PROJECT_ROOT/scripts/lib/benchmark.sh"
benchmark_bootstrap "$0" "clf-matrix"
export HEVC_ENCODER=${HEVC_ENCODER:-hevc_nvenc}

RUN_TAG=${RUN_TAG:-clf-matrix}
export MODEL_TYPE="$MT" PRUNE_STAGE="$STAGE" RUN_MODE=both
export DATASET=k400 SEED=42 SKIP_EXISTING=1
export RANDOM_SAMPLES="${RANDOM_SAMPLES:-}"          # empty = whole split
export BATCH_SIZE="${BATCH_SIZE:-16}"
export OUTPUT_ROOT="$PROJECT_ROOT/benchmark/results/$RUN_TAG/$MT"
export OUTPUT_DIR="$OUTPUT_ROOT"

case "$KIND" in
  all_modes)
    export SUITE=all_modes K_KEEP_RATE="0.125 0.25 0.5 0.75" ;;
  fold_block)
    export SUITE="" P_MODE=shared_folding FOLDING_MODE="$FMODE" \
           FOLD_BLOCK_WISE=1 FOLD_BLOCK_SIZE=2 FOLD_POOLING=coverage-hard \
           FOLD_SLOTS_PER_BLOCK="$SLOTS" FOLD_MIN_SLOTS_PER_BLOCK=1 \
           K_KEEP_RATE="$RATE" ;;
  fold_global)
    export SUITE="" P_MODE=shared_folding FOLDING_MODE="$FMODE" \
           FOLD_BLOCK_WISE=0 K_KEEP_RATE="0.25 0.5 0.75" ;;
esac

exec python -m benchmark.sweeps classification
