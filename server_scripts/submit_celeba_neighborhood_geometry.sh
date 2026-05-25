#!/bin/bash
#SBATCH --job-name=celeba-neigh-geo
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p output/slurm

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate smoothing || true
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python -m src.experiments.analysis.celeba_neighborhood_geometry --pca-backend gpu "$@"
