#!/usr/bin/env bash
# =============================================================================
# RoCOCO visualization — array job matching eval structure
# Same task matrix as submit_rococo_eval.sh:
#   Tasks 1–5:   baseline × 5 anns
#   Tasks 6–35:  isotropic × 6 sigmas × 5 anns
#   Tasks 36–65: manifold  × 6 sigmas × 5 anns
#   Total: 65 tasks, max 9 concurrent
#
# Requires: embeddings + eval metrics already exist.
#
# Usage:
#   sbatch server_scripts/submit_rococo_viz.sh
#   sbatch server_scripts/submit_rococo_viz.sh --max-concurrent 5
#   sbatch server_scripts/submit_rococo_viz.sh --dry-run
# =============================================================================
#SBATCH -p gpu-rtx8000
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-viz-%A_%a.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-viz-%A_%a.err
#SBATCH -J rococo-viz

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
BASELINE_TASKS=$((N_ANNS))
SMOOTH_TASKS=$((N_SIGMAS * N_ANNS))
TOTAL=$((BASELINE_TASKS + 2 * SMOOTH_TASKS))

# ── Array task runner ─────────────────────────────────────────────────────────
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IDX=$((SLURM_ARRAY_TASK_ID - 1))

    if [[ $IDX -lt $BASELINE_TASKS ]]; then
        MODE="baseline"; SIGMA="0.0"; ANN="${ANN_FILES[$IDX]}"
    else
        SMOOTH_IDX=$((IDX - BASELINE_TASKS))
        [[ $SMOOTH_IDX -lt $SMOOTH_TASKS ]] && MODE="isotropic" || MODE="manifold"
        OFFSET=$(( SMOOTH_IDX < SMOOTH_TASKS ? SMOOTH_IDX : SMOOTH_IDX - SMOOTH_TASKS ))
        SIGMA="${SIGMAS[$((OFFSET / N_ANNS))]}"
        ANN="${ANN_FILES[$((OFFSET % N_ANNS))]}"
    fi

    SIGMA_TAG="sigma_${SIGMA//./_}"
    ANN_STEM="${ANN%.json}"
    CFG="src/configs/experiments/rococo_clip_${MODE}.yaml"
    VIZ_OUT="output/rococo/${MODE}/${SIGMA_TAG}/visualizations/${ANN_STEM}_viz.png"

    echo "Task $SLURM_ARRAY_TASK_ID: mode=$MODE  sigma=$SIGMA  ann=$ANN_STEM  host=$(hostname)"

    if [[ -f "$VIZ_OUT" ]]; then
        echo "SKIP — $VIZ_OUT already exists"
        exit 0
    fi

    python -m src.experiments.eval.rococo_viz \
        --config "$CFG" \
        --ann-file "$ANN" \
        --sigma "$SIGMA"
    exit 0
fi

# ── Submit ────────────────────────────────────────────────────────────────────
DRY_RUN=false
MAX_CONCURRENT=9
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=true;       shift ;;
        --max-concurrent)  MAX_CONCURRENT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "Total tasks: $TOTAL  (5 baseline + 30 iso + 30 manifold)"
echo "Max concurrent: $MAX_CONCURRENT"

if $DRY_RUN; then
    echo "[dry-run] Would submit $TOTAL tasks"
else
    sbatch \
        --array="1-${TOTAL}%${MAX_CONCURRENT}" \
        --partition=gpu-rtx8000 \
        --time=24:00:00 \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem-per-cpu=8G \
        --job-name=rococo-viz \
        --output="$PROJECT_ROOT/output/slurm/rococo-viz-%A_%a.out" \
        --error="$PROJECT_ROOT/output/slurm/rococo-viz-%A_%a.err" \
        "$0"
    echo "Submitted $TOTAL viz tasks (max $MAX_CONCURRENT concurrent)"
fi
