#!/usr/bin/env bash
# =============================================================================
# Build CelebA-HQ latent index (1024-dim VAE)
# Requires trained VAE at output/pretrained_model/vae_celebahq_512/best.pt
# Submit AFTER VAE training, BEFORE latent certification sweeps
#
# Usage:
#   sbatch server_scripts/submit_build_celebahq_latent_index.sh
# =============================================================================

#SBATCH -p gpu-rtx8000
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/build-celebahq-latent-index-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/build-celebahq-latent-index-%j.err
#SBATCH -J build-hq-latent-idx

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
echo "Building CelebA-HQ Latent Index (1024-dim)"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

python -m src.experiments.indexing.celeba_images \
    --config src/configs/experiments/certify_celebahq_latent.yaml \
    --space latent \
    --backend annoy

echo "================================================"
echo "Done: $(date)"
echo "================================================"
