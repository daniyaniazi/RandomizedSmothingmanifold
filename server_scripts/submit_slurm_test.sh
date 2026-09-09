#!/usr/bin/env bash
# MPI-INF Slurm smoke-test job for test.py
#
# Usage:
#   sbatch src/experiments/training/resnet_smile/submit_slurm_test.sh

#SBATCH -p gpu-rtx8000
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/smoke-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/smoke-%j.err
#SBATCH -J smile-smoke

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"

cd "$PROJECT_ROOT"
mkdir -p output/slurm

# Activate conda env (miniforge3)
if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python: $(command -v python)"

python test.py