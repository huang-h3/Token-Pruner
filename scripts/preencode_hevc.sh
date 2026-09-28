#!/usr/bin/env bash
#SBATCH --job-name=hevc_preencode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
  SCRIPT_DIR="$PROJECT_ROOT/scripts"
else
  SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
fi
source "$SCRIPT_DIR/lib/benchmark.sh"
# Submit with sbatch directly so CLI arguments are not discarded.
RUN_ON_CLUSTER=0
benchmark_bootstrap "$0" hevc_preencode
export HEVC_ENCODER=${HEVC_ENCODER:-hevc_nvenc}
exec python -m benchmark.preencode_hevc "$@"
