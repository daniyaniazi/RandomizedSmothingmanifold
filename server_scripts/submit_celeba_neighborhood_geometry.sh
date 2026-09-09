#!/usr/bin/env bash
#SBATCH -p gpu-rtx8000
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/celeba-neigh-geo-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/celeba-neigh-geo-%j.err
#SBATCH -J celeba-neigh-geo

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"

mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"

python -m src.experiments.analysis.celeba_neighborhood_geometry \
    --pca-backend gpu \
    --sigmas 0.10 0.20 0.50 1.00 \
    --knn-k 500 \
    --n-samples 5 \
    --pca-components 20 \
    "$@"
