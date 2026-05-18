#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 72:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=32G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/vae-celebahq512-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/vae-celebahq512-%j.err
#SBATCH -J vae-celebahq512

# VAE CelebA-HQ at 512×512 — needs more time & memory than 128
# Auto-resumes from latest.pt checkpoint if killed

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
nvidia-smi || true

python -m src.experiments.training.vae.main \
    --config src/configs/training/vae_celebahq_512.yaml
