"""Filesystem layout for benchmark entry points."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = BENCHMARK_ROOT / "results"

CLF_RESULTS_ROOT = RESULTS_ROOT / "classification"
VLM_RESULTS_ROOT = RESULTS_ROOT / "vlm"

CLF_HEVC_DIR = DATA_ROOT / "inference_hevc"
VLM_HEVC_DIR = RESULTS_ROOT / "vlm" / "hevc_tmp"
#: Content-addressed HEVC artifacts shared by every task, model and sweep.
HEVC_STORE_DIR = RESULTS_ROOT / "hevc_cache" / "store"
