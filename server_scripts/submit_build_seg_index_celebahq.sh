#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=32G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/seg-index-celebahq-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/seg-index-celebahq-%j.err
#SBATCH -J seg-index-celebahq

# =============================================================================
# Build pixel-space kNN index for CelebAMask-HQ segmentation certification.
# Indexes train split (IDs 0-27999) in raw [0,1] pixel space.
#
# Usage:
#   sbatch server_scripts/submit_build_seg_index_celebahq.sh
#   sbatch server_scripts/submit_build_seg_index_celebahq.sh --metric angular
#   sbatch server_scripts/submit_build_seg_index_celebahq.sh --rebuild
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "Running on: $(hostname)"

python -m src.experiments.indexing.celebahq_seg_images \
    --config src/configs/experiments/certify_celebahq_seg_manifold.yaml \
    "$@"
