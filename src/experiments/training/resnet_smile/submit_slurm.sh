#!/usr/bin/env bash
# MPI-INF Slurm GPU job — submit from slurm-submit.mpi-inf.mpg.de
#
# Usage:
#   sbatch src/experiments/training/resnet_smile/submit_slurm.sh
#   sbatch src/experiments/training/resnet_smile/submit_slurm.sh src/configs/training/smile_resnet_celebahq.yaml

#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/smile-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/smile-%j.err
#SBATCH -J smile-resnet

set -euo pipefail

CONFIG_PATH="${1:-src/configs/training/smile_resnet_celeba.yaml}"
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"

cd "$PROJECT_ROOT"
mkdir -p output/slurm

# Activate conda env (miniforge3)
if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "Running on: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG_PATH"

python -m src.experiments.training.resnet_smile.main --config "$CONFIG_PATH"