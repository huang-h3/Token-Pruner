#!/usr/bin/env bash

benchmark_bootstrap() {
  local script_path job_name conda_sh log_dir
  script_path="$(readlink -f "$1")"
  job_name=$2

  PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-"$(cd "$(dirname "$script_path")/.." && pwd)"}}
  PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
  export PROJECT_ROOT
  if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "Project root is invalid: $PROJECT_ROOT" >&2
    exit 2
  fi

  # RUN_ON_CLUSTER=1 resubmits this script with sbatch; SBATCH_PARTITION and
  # other SBATCH_* variables are read by sbatch itself.
  log_dir=${SLURM_LOG_DIR:-"$PROJECT_ROOT/benchmark/results/slurm"}
  if [ "${RUN_ON_CLUSTER:-0}" = "1" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$log_dir"
    sbatch \
      --job-name "${SBATCH_JOB_NAME:-$job_name}" \
      --output "${SBATCH_OUTPUT:-$log_dir/%x_%j.out}" \
      --error "${SBATCH_ERROR:-$log_dir/%x_%j.err}" \
      --gres "${SBATCH_GRES:-gpu:1}" \
      --cpus-per-task "${SBATCH_CPUS_PER_TASK:-8}" \
      --export=ALL,RUN_ON_CLUSTER=0,PROJECT_ROOT="$PROJECT_ROOT" \
      "$script_path"
    exit 0
  fi

  # Activate CONDA_ENV when it is set; otherwise run in the current environment.
  if [ -n "${CONDA_ENV:-}" ]; then
    conda_sh=${CONDA_SH:-}
    if [ -z "$conda_sh" ] && command -v conda >/dev/null 2>&1; then
      conda_sh="$(conda info --base)/etc/profile.d/conda.sh"
    fi
    if [ ! -f "$conda_sh" ]; then
      echo "CONDA_ENV=$CONDA_ENV is set but conda.sh was not found; set CONDA_SH." >&2
      exit 2
    fi
    source "$conda_sh"
    conda activate "$CONDA_ENV"
  fi
  if [ -n "${LMMS_EVAL_ROOT:-}" ]; then
    export PYTHONPATH="$LMMS_EVAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  fi
  cd "$PROJECT_ROOT"
  export PYTHONUNBUFFERED=1
}
