#!/usr/bin/env bash
#SBATCH --job-name=vlm_hf_lmms
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
  SCRIPT_DIR="$PROJECT_ROOT/scripts"
else
  SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
fi
source "$SCRIPT_DIR/lib/benchmark.sh"
benchmark_bootstrap "$0" "vlm_hf_lmms"

# One array task per TASK_LIST entry; falls back to TASKS outside an array.
if [ -n "${TASK_LIST:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  # shellcheck disable=SC2086
  set -- $TASK_LIST
  if [ "$SLURM_ARRAY_TASK_ID" -gt "$#" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID exceeds TASK_LIST ($# entries)" >&2
    exit 2
  fi
  eval "export TASKS=\${$SLURM_ARRAY_TASK_ID}"
  echo "array task $SLURM_ARRAY_TASK_ID -> TASKS=$TASKS"
fi

export HEVC_ENCODER=${HEVC_ENCODER:-hevc_nvenc}
exec python -m benchmark.sweeps vlm
