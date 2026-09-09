#!/usr/bin/env bash
#SBATCH -p gpu-rtx8000
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/vae-celebahq128-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/vae-celebahq128-%j.err
#SBATCH -J vae-celebahq128

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"

python -m src.experiments.training.vae.main \
    --config src/configs/training/vae_celebahq_128.yaml
