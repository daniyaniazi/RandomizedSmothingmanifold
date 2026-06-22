#!/usr/bin/env bash
# =============================================================================
# RoCOCO neighbourhood geometry analysis
# Runs on a single CPU node — no GPU needed.
#
# Saves to: output/rococo/geometry/
#   geometry_anchor{i}_sigma{s}.png   — 4-panel figure per anchor × sigma
#   alpha_amplification.png           — noise amplification curve
#   geometry_summary.json             — lambda_max / alpha per anchor
#
# Usage:
#   sbatch server_scripts/submit_rococo_geometry.sh
# =============================================================================
#SBATCH -p cpu20
#SBATCH -t 01:00:00
#SBATCH --gres gpu:0
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-geometry-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-geometry-%j.err
#SBATCH -J rococo-geometry

set -euo pipefail
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "Running RoCOCO neighbourhood geometry analysis"
echo "Output → $PROJECT_ROOT/output/rococo/geometry/"

python -m src.experiments.analysis.rococo_neighborhood_geometry \
    --config src/configs/experiments/rococo_clip_manifold.yaml \
    --sigmas 0.02 0.05 0.07 0.10 0.20 \
    --n-anchors 10 \
    --knn-k 500 \
    --n-mc 0

echo "Done."
