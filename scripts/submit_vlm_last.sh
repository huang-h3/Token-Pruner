#!/usr/bin/env bash
# Last-stage pruning matrix for four checkpoints, after one shared HEVC preencode job.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"}
cd "$PROJECT_ROOT"
unset SLURM_JOB_ID PRUNE_LAYER TASK_LIST TOKEN_PRUNER_STOP_STRINGS
unset TOKEN_PRUNER_ANSWER_FORMAT_SUFFIX TOKEN_PRUNER_MAX_NEW_TOKENS

QWEN_TASKS=${QWEN_TASKS:-"nextqa motionbench mvbench vitatecs"}
HF_TASKS=${HF_TASKS:-"nextqa motionbench mvbench vitatecs"}
OFF_TASKS=${OFF_TASKS:-"nextqa motionbench mvbench vitatecs"}
LNV_TASKS=${LNV_TASKS:-"nextqa mvbench vitatecs"}
OFFICIAL_MODEL_PATH=${OFFICIAL_MODEL_PATH:-LanguageBind/Video-LLaVA-7B}
export LAST_RATES=${LAST_RATES:-"0.125 0.25 0.5 0.75"}
export LIMIT=${LIMIT:-1024}
CONCURRENT=${CONCURRENT:-4}
mkdir -p "$PROJECT_ROOT/benchmark/results/slurm"

export LOG_SAMPLES=1 SUITE='' FOLDING_MODE=temporal-diff FOLD_BLOCK_SIZE=2 FOLD_POOLING=coverage-hard
export DEVICE=cuda:0 LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0} SKIP_EXISTING=1 SEED=42 VERBOSITY=INFO
export LOAD_4BIT=${LOAD_4BIT:-0} LOAD_8BIT=${LOAD_8BIT:-0}
export HEVC_ENCODER=${HEVC_ENCODER:-hevc_nvenc} HEVC_ENCODE_SCOPE=full-video
export HEVC_SAMPLED_GOP_SIZE=32 HEVC_ANCHOR_POLICY=first HEVC_N_PARALLEL=4

CELL_LAST='global:preserve:hevc global:folder:hevc shared:shared_hevc:shared_hevc local:folder:folder shared:shared_folding:shared_folding global:random:random global:uniform:uniform'
OFF_LAST='local:folder:hevc shared:shared_hevc:shared_hevc local:folder:folder shared:shared_folding:shared_folding local:random:random local:uniform:uniform'
QWEN_LAST='global:preserve:hevc shared:shared_hevc:shared_hevc global:random:random global:uniform:uniform'

# The task union is deduplicated by preencode_hevc. No VLM weights are loaded.
HEVC_JOB=$(TASKS="$QWEN_TASKS $HF_TASKS $OFF_TASKS $LNV_TASKS" \
    sbatch --parsable --job-name=hevc-preencode \
    --gres=gpu:1 --mem=64G --cpus-per-task=8 --time=24:00:00 \
    --output="$PROJECT_ROOT/benchmark/results/slurm/%x_%j.out" \
    --error="$PROJECT_ROOT/benchmark/results/slurm/%x_%j.err" \
    --export=ALL "$PROJECT_ROOT/scripts/preencode_hevc.sh")
HEVC_JOB=${HEVC_JOB%%;*}
echo "HEVC preencode job: $HEVC_JOB"

WRAP='
set -- $TASK_LIST_IN
if [ "$SLURM_ARRAY_TASK_ID" -lt 1 ] || [ "$SLURM_ARRAY_TASK_ID" -gt "$#" ]; then
    echo "invalid array index: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
eval TASKS=\${$SLURM_ARRAY_TASK_ID}
export TASKS OUTPUT_DIR="$OUTPUT_DIR_BASE"
unset TASK_LIST
STATUS=0
for M in $LAST_MODES; do
    SC=${M%%:*}; REST=${M#*:}; IM=${REST%%:*}; PM=${REST##*:}
    echo "=== $TASKS / last / $SC-$IM-$PM / rates=$LAST_RATES ==="
    if env RUN_MODE=pruned PRUNE_STAGE=last PRUNE_MODE="$SC" I_MODE="$IM" P_MODE="$PM" \
        K_KEEP_RATE="$LAST_RATES" "$PROJECT_ROOT/scripts/infer_vlm_hf.sh"; then
        :
    else
        RC=$?
        echo "!! $TASKS / $M failed with exit code $RC" >&2
        STATUS=1
    fi
done
exit "$STATUS"
'

submit() {
    local run="$5-f$4-t$7-l${LIMIT}" count
    count=$(set -- $6; echo $#)
    echo "Submitting $run: $count array tasks after HEVC job $HEVC_JOB"
    RUN_ON_CLUSTER=1 MODEL_ID="$1" MODEL_PATH="$2" CKPT_NAME="$3" NUM_FRAMES="$4" \
    TASK_LIST_IN="$6" TOKEN_PRUNER_MAX_NEW_TOKENS="$7" BATCH_SIZE="$8" LAST_MODES="$9" \
    OUTPUT_ROOT="$PROJECT_ROOT/benchmark/results/$run" \
    OUTPUT_DIR_BASE="$PROJECT_ROOT/benchmark/results/$run/$3" \
    HEVC_DIR="$PROJECT_ROOT/benchmark/results/$run/hevc_tmp" \
    sbatch --job-name="$run-last" --dependency="afterok:$HEVC_JOB" \
        --array="1-${count}%${CONCURRENT}" \
        --gres=gpu:1 --mem=64G --cpus-per-task=8 --time=24:00:00 \
        --output="$PROJECT_ROOT/benchmark/results/slurm/%x_%A_%a.out" \
        --error="$PROJECT_ROOT/benchmark/results/slurm/%x_%A_%a.err" \
        --export=ALL --wrap="$WRAP"
}

submit qwen3_vl Qwen/Qwen3-VL-8B-Instruct Qwen3-VL-8B-Instruct 64 qwen3 "$QWEN_TASKS" 2048 8 "$QWEN_LAST"
submit video_llava_hf LanguageBind/Video-LLaVA-7B-hf Video-LLaVA-7B-hf 8 hf "$HF_TASKS" 1024 8 "$CELL_LAST"
submit video_llava_official "$OFFICIAL_MODEL_PATH" Video-LLaVA-7B 8 off "$OFF_TASKS" 1024 8 "$OFF_LAST"
submit llava_next_video llava-hf/LLaVA-NeXT-Video-7B-hf LLaVA-NeXT-Video-7B-hf 16 lnv "$LNV_TASKS" 1024 8 "$CELL_LAST"
