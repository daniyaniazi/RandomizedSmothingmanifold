#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/certify-celebahq-latent-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/certify-celebahq-latent-%j.err
#SBATCH -J certify-celebahq-latent

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
echo "CelebA-HQ Latent Manifold Certification"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

python -m src.experiments.certify.celeba \
    --config src/configs/experiments/certify_celebahq_latent.yaml

echo "Done: $(date)"
