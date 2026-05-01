#!/usr/bin/env bash
# MPI-INF Slurm GPU job — submit from slurm-submit.mpi-inf.mpg.de
#
# Usage:
#   sbatch src/experiments/training/ner/submit_slurm.sh
#   sbatch src/experiments/training/ner/submit_slurm.sh src/configs/experiments/ner_conll2003_bert.yaml

#SBATCH -p gpu20
#SBATCH -t 12:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/ner-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/ner-%j.err
#SBATCH -J ner-finetune

set -euo pipefail

CONFIG_PATH="${1:-src/configs/experiments/ner_conll2003_distilbert.yaml}"
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

python -m src.models.transformer.ner.train --config "$CONFIG_PATH"
