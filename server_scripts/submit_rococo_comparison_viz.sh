#!/usr/bin/env bash
# =============================================================================
# RoCOCO comparison viz — CLIP vs ISO vs Manifold, one job per (sigma, ann)
#
# Task layout:
#   N_SIGMAS × N_ANNS tasks, one figure set per (sigma, ann)
#
# Usage:
#   sbatch server_scripts/submit_rococo_comparison_viz.sh
#   sbatch server_scripts/submit_rococo_comparison_viz.sh --dry-run
#   sbatch server_scripts/submit_rococo_comparison_viz.sh --sigma 0.05 0.10
# =============================================================================
#SBATCH -p gpu20
#SBATCH -t 02:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-compviz-%A_%a.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-compviz-%A_%a.err
#SBATCH -J rococo-compviz

set -euo pipefail
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

ANN_FILES=(coco_karpathy_test.json danger.json same_concept.json diff_concept.json rand_voca.json)
SIGMAS=(0.02 0.05 0.07 0.10 0.20 0.50 0.70 1.00)
N_ANNS=${#ANN_FILES[@]}
N_SIGMAS=${#SIGMAS[@]}
TOTAL=$((N_SIGMAS * N_ANNS))   # 40

# ── Array task body ───────────────────────────────────────────────────────────
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    SIGMA_IDX=$((IDX / N_ANNS))
    ANN_IDX=$((IDX % N_ANNS))
    SIGMA="${SIGMAS[$SIGMA_IDX]}"
    ANN="${ANN_FILES[$ANN_IDX]}"
    ANN_STEM="${ANN%.json}"
    SIGMA_TAG="sigma_${SIGMA//./_}"

    echo "Task $SLURM_ARRAY_TASK_ID: sigma=$SIGMA  ann=$ANN_STEM  host=$(hostname)"

    # Skip if all sample files already exist
    OUT_DIR="output/rococo/comparison/${ANN_STEM}/${SIGMA_TAG}"
    if [[ -f "${OUT_DIR}/sample_05.png" ]]; then
        echo "SKIP — $OUT_DIR already complete"
        exit 0
    fi

    python -m src.experiments.eval.rococo_comparison_viz \
        --config src/configs/experiments/rococo_clip_manifold.yaml \
        --sigma  "$SIGMA" \
        --ann    "$ANN_STEM" \
        --n-samples 5
    exit 0
fi

# ── Submit ────────────────────────────────────────────────────────────────────
DRY_RUN=false
MAX_CONCURRENT=8
OVERRIDE_SIGMAS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=true;       shift ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --sigma)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                OVERRIDE_SIGMAS+=("$1"); shift
            done ;;
        *) shift ;;
    esac
done

# If specific sigmas requested, remap SIGMAS and recompute TOTAL
if [[ ${#OVERRIDE_SIGMAS[@]} -gt 0 ]]; then
    SIGMAS=("${OVERRIDE_SIGMAS[@]}")
    N_SIGMAS=${#SIGMAS[@]}
    TOTAL=$((N_SIGMAS * N_ANNS))
fi

echo "Tasks: $TOTAL  (${N_SIGMAS} sigmas × ${N_ANNS} anns)"
echo "Sigmas: ${SIGMAS[*]}"
echo "Max concurrent: $MAX_CONCURRENT"

if $DRY_RUN; then
    for s_idx in "${!SIGMAS[@]}"; do
        for a_idx in "${!ANN_FILES[@]}"; do
            t=$((s_idx * N_ANNS + a_idx + 1))
            echo "  Task $t: sigma=${SIGMAS[$s_idx]}  ann=${ANN_FILES[$a_idx]%.json}"
        done
    done
else
    sbatch \
        --array="1-${TOTAL}%${MAX_CONCURRENT}" \
        --partition=gpu20 \
        --time=02:00:00 \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem-per-cpu=8G \
        --job-name=rococo-compviz \
        --output="$PROJECT_ROOT/output/slurm/rococo-compviz-%A_%a.out" \
        --error="$PROJECT_ROOT/output/slurm/rococo-compviz-%A_%a.err" \
        "$0"
    echo "Submitted $TOTAL tasks (max $MAX_CONCURRENT concurrent)"
fi
