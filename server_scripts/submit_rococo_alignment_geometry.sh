#!/usr/bin/env bash
# =============================================================================
# RoCOCO GT-vs-adversarial alignment geometry diagnostic
#
# Computes Q_sep = d_sep^T Sigma_loc d_sep and compares it to random
# directions, then saves CSV/JSON summaries and example geometry figures.
#
# Output:
#   output/rococo/alignment_geometry/{ann_stem}/
#
# Usage:
#   sbatch server_scripts/submit_rococo_alignment_geometry.sh
#   sbatch server_scripts/submit_rococo_alignment_geometry.sh danger paired 1000
#   sbatch server_scripts/submit_rococo_alignment_geometry.sh danger global 5000
#   sbatch server_scripts/submit_rococo_alignment_geometry.sh same_concept paired 5000
# =============================================================================
#SBATCH -p cpu20
#SBATCH -t 24:00:00
#SBATCH --gres gpu:0
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-align-geometry-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-align-geometry-%j.err
#SBATCH -J rococo-align-geometry

set -euo pipefail
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

ANN_STEM="${1:-danger}"
ADV_SCOPE="${2:-paired}"
N_IMAGES="${3:-1000}"
KNN_K="${4:-500}"
N_RANDOM="${5:-128}"
N_VIZ="${6:-12}"

echo "Running RoCOCO alignment geometry"
echo "  ann_stem:  $ANN_STEM"
echo "  adv_scope: $ADV_SCOPE"
echo "  n_images:  $N_IMAGES"
echo "  knn_k:     $KNN_K"
echo "  n_random:  $N_RANDOM"
echo "  n_viz:     $N_VIZ"
echo "Output -> $PROJECT_ROOT/output/rococo/alignment_geometry/$ANN_STEM/"

python -m src.experiments.analysis.rococo_alignment_geometry \
    --config src/configs/experiments/rococo_clip_manifold.yaml \
    --ann-stem "$ANN_STEM" \
    --adv-scope "$ADV_SCOPE" \
    --n-images "$N_IMAGES" \
    --knn-k "$KNN_K" \
    --n-random "$N_RANDOM" \
    --n-viz "$N_VIZ"

echo "Done."
