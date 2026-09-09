#!/usr/bin/env bash
# =============================================================================
# RoCOCO retrieval-instability diagnostic: isotropic vs manifold smoothing
#
# Saves score-change, margin-change, rank-change, and flip-rate summaries.
#
# Usage:
#   sbatch server_scripts/submit_rococo_retrieval_instability.sh
#   sbatch server_scripts/submit_rococo_retrieval_instability.sh danger global 0.10 1000 32
#   sbatch server_scripts/submit_rococo_retrieval_instability.sh danger paired 0.10 5000 32
# =============================================================================
#SBATCH -p cpu20
#SBATCH -t 24:00:00
#SBATCH --gres gpu:0
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-instability-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-instability-%j.err
#SBATCH -J rococo-instability

set -euo pipefail
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

ANN_STEM="${1:-danger}"
ADV_SCOPE="${2:-global}"
SIGMA="${3:-0.10}"
N_IMAGES="${4:-1000}"
N_SAMPLES="${5:-32}"
KNN_K="${6:-500}"

echo "Running RoCOCO retrieval instability"
echo "  ann_stem:  $ANN_STEM"
echo "  adv_scope: $ADV_SCOPE"
echo "  sigma:     $SIGMA"
echo "  n_images:  $N_IMAGES"
echo "  n_samples: $N_SAMPLES"
echo "  knn_k:     $KNN_K"

python -m src.experiments.analysis.rococo_retrieval_instability \
    --config src/configs/experiments/rococo_clip_manifold.yaml \
    --ann-stem "$ANN_STEM" \
    --adv-scope "$ADV_SCOPE" \
    --sigma "$SIGMA" \
    --n-images "$N_IMAGES" \
    --n-samples "$N_SAMPLES" \
    --knn-k "$KNN_K"

echo "Done."
