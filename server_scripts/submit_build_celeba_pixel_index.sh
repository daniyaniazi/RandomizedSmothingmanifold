#!/usr/bin/env bash
# =============================================================================
# Build CelebA pixel index ONLY (no certification)
# Submit this FIRST, then submit sigma sweeps with --dependency
#
# Usage:
#   JOB_ID=$(sbatch --parsable submit_build_celeba_pixel_index.sh)
#   sbatch --dependency=afterok:$JOB_ID submit_sigma_sweep.sh src/configs/experiments/certify_celeba_pixel.yaml
# =============================================================================

#SBATCH -p gpu-rtx8000
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=32G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/build-celeba-pixel-index-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/build-celeba-pixel-index-%j.err
#SBATCH -J build-celeba-pixel-idx

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]; then
    . "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "================================================"
echo "Building CelebA Pixel Index (224x224)"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

python -m src.experiments.indexing.celeba_images \
    --config src/configs/experiments/certify_celeba_pixel.yaml \
    --space pixel \
    --backend annoy

echo "================================================"
echo "Done: $(date)"
echo "================================================"
