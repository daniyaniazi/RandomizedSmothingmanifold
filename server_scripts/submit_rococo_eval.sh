#!/usr/bin/env bash
# =============================================================================
# RoCOCO CLIP evaluation — sigma sweep × 3 modes × 5 annotations
#
# Task order: all modes/anns for sigma_0 first, then sigma_1, etc.
# e.g. 6 sigmas × 3 modes × 5 anns = 90 tasks, max 9 concurrent
#
# Usage:
#   sbatch server_scripts/submit_rococo_eval.sh          # all tasks
#   sbatch server_scripts/submit_rococo_eval.sh --dry-run
# =============================================================================
#SBATCH -p gpu20
#SBATCH -t 12:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-eval-%A_%a.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-eval-%A_%a.err
#SBATCH -J rococo-eval

set -euo pipefail
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

# ── Task matrix ───────────────────────────────────────────────────────────────
# Baseline runs ONCE per annotation (sigma has no effect on baseline).
# Isotropic and Manifold run once per (sigma, annotation).
#
# Task layout (flat index):
#   1  ..  5   = baseline  × 5 anns          (sigma ignored, uses sigma=0)
#   6  .. 35   = isotropic × 6 sigmas × 5 anns
#   36 .. 65   = manifold  × 6 sigmas × 5 anns

ANN_FILES=(coco_karpathy_test.json danger.json same_concept.json diff_concept.json rand_voca.json)
SIGMAS=(0.05 0.10 0.25 0.50 0.75 1.00)
N_ANNS=${#ANN_FILES[@]}    # 5
N_SIGMAS=${#SIGMAS[@]}     # 6

BASELINE_TASKS=$((N_ANNS))                  #  5
SMOOTH_TASKS=$((N_SIGMAS * N_ANNS))         # 30
TOTAL=$((BASELINE_TASKS + 2 * SMOOTH_TASKS)) # 65

# If running as array job
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IDX=$((SLURM_ARRAY_TASK_ID - 1))

    if [[ $IDX -lt $BASELINE_TASKS ]]; then
        # ── Baseline (no sigma) ────────────────────────────────────────────────
        ANN_IDX=$IDX
        MODE="baseline"
        ANN="${ANN_FILES[$ANN_IDX]}"
        ANN_STEM="${ANN%.json}"
        SIGMA="0.0"
        CFG="src/configs/experiments/rococo_clip_baseline.yaml"
        OUT="output/rococo/baseline/sigma_0_00/${ANN_STEM}_metrics.json"
    else
        SMOOTH_IDX=$((IDX - BASELINE_TASKS))
        if [[ $SMOOTH_IDX -lt $SMOOTH_TASKS ]]; then
            MODE="isotropic"
            OFFSET=$SMOOTH_IDX
        else
            MODE="manifold"
            OFFSET=$((SMOOTH_IDX - SMOOTH_TASKS))
        fi
        SIGMA_IDX=$((OFFSET / N_ANNS))
        ANN_IDX=$((OFFSET % N_ANNS))
        SIGMA="${SIGMAS[$SIGMA_IDX]}"
        ANN="${ANN_FILES[$ANN_IDX]}"
        ANN_STEM="${ANN%.json}"
        SIGMA_TAG="sigma_${SIGMA//./_}"
        CFG="src/configs/experiments/rococo_clip_${MODE}.yaml"
        OUT="output/rococo/${MODE}/${SIGMA_TAG}/${ANN_STEM}_metrics.json"
    fi

    echo "Task $SLURM_ARRAY_TASK_ID: mode=$MODE  sigma=$SIGMA  ann=${ANN%.json}  host=$(hostname)"

    if [[ -f "$OUT" ]]; then
        echo "SKIP — $OUT already exists"
        exit 0
    fi

    python -m src.experiments.eval.rococo_clip_eval \
        --config "$CFG" \
        --ann-file "$ANN" \
        --sigma  "$SIGMA" \
        --n-eval 100
    exit 0
fi

# ── Submit as array job (called without SLURM_ARRAY_TASK_ID) ─────────────────
DRY_RUN=false
MAX_CONCURRENT=9

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=true;          shift ;;
        --max-concurrent)  MAX_CONCURRENT="$2";    shift 2 ;;
        *) shift ;;
    esac
done

echo "Total tasks:    $TOTAL  (5 baseline + 30 isotropic + 30 manifold)"
echo "Max concurrent: $MAX_CONCURRENT"

if $DRY_RUN; then
    for a_idx in "${!ANN_FILES[@]}"; do
        echo "  Task $((a_idx+1)): mode=baseline  ann=${ANN_FILES[$a_idx]%.json}"
    done
    for s_idx in "${!SIGMAS[@]}"; do
        for a_idx in "${!ANN_FILES[@]}"; do
            t=$((BASELINE_TASKS + s_idx*N_ANNS + a_idx + 1))
            echo "  Task $t: mode=isotropic  sigma=${SIGMAS[$s_idx]}  ann=${ANN_FILES[$a_idx]%.json}"
        done
    done
    for s_idx in "${!SIGMAS[@]}"; do
        for a_idx in "${!ANN_FILES[@]}"; do
            t=$((BASELINE_TASKS + SMOOTH_TASKS + s_idx*N_ANNS + a_idx + 1))
            echo "  Task $t: mode=manifold   sigma=${SIGMAS[$s_idx]}  ann=${ANN_FILES[$a_idx]%.json}"
        done
    done
else
    sbatch \
        --array="1-${TOTAL}%${MAX_CONCURRENT}" \
        --partition=gpu20 \
        --time=12:00:00 \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem-per-cpu=8G \
        --job-name=rococo-eval \
        --output="$PROJECT_ROOT/output/slurm/rococo-eval-%A_%a.out" \
        --error="$PROJECT_ROOT/output/slurm/rococo-eval-%A_%a.err" \
        "$0"
    echo "Submitted $TOTAL tasks (max $MAX_CONCURRENT concurrent)"
fi
