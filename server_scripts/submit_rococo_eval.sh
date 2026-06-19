#!/usr/bin/env bash
# =============================================================================
# RoCOCO CLIP evaluation — 3 modes × 5 annotation files = 15 array tasks
# Each task: one (mode, annotation) pair
#
# Usage:
#   sbatch server_scripts/submit_rococo_eval.sh            # all 15 tasks
#   sbatch server_scripts/submit_rococo_eval.sh --dry-run  # print tasks
#   sbatch --array=1-5 server_scripts/submit_rococo_eval.sh  # baseline only
# =============================================================================
#SBATCH -p gpu20
#SBATCH -t 8:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-15%5
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

# Task matrix: 3 modes × 5 annotation files = 15 tasks
MODES=(baseline isotropic manifold)
ANN_FILES=(coco_karpathy_test.json danger.json same_concept.json diff_concept.json rand_voca.json)

TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))
MODE_IDX=$((TASK_IDX / 5))
ANN_IDX=$((TASK_IDX % 5))

MODE="${MODES[$MODE_IDX]}"
ANN="${ANN_FILES[$ANN_IDX]}"
ANN_STEM="${ANN%.json}"
CFG="src/configs/experiments/rococo_clip_${MODE}.yaml"

echo "Task $SLURM_ARRAY_TASK_ID: mode=$MODE  ann=$ANN_STEM  host=$(hostname)"

# Skip if already done
OUT="output/rococo/${MODE}/${ANN_STEM}_metrics.json"
if [[ -f "$OUT" ]]; then
    echo "SKIP — $OUT already exists"
    exit 0
fi

python -m src.experiments.eval.rococo_clip_eval \
    --config "$CFG" \
    --ann-file "$ANN"
