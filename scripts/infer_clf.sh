#!/usr/bin/env bash
#SBATCH --job-name=clf_benchmark
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
benchmark_bootstrap "$0" "clf_benchmark"

export HEVC_ENCODER=${HEVC_ENCODER:-hevc_nvenc}
exec python -m benchmark.sweeps classification
