#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/certify-celebahq-iso-latent-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/certify-celebahq-iso-latent-%j.err
#SBATCH -J hq-iso-lat

# CelebA-HQ Certification - Isotropic (Gaussian) Smoothing in Latent Space
# Baseline experiment: Gaussian noise in VAE latent space (no manifold)

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "================================================"
echo "CelebA-HQ Isotropic Latent Certification"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

python -m src.experiments.certify.celeba \
    --config src/configs/experiments/certify_celebahq_isotropic_latent_128.yaml

echo "Done: $(date)"
