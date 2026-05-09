#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=16G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/ner-bert-isotropic-certify-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/ner-bert-isotropic-certify-%j.err
#SBATCH -J ner-bert-iso

# NER Certification - Isotropic (Gaussian) Smoothing
# Baseline: standard Gaussian noise on hidden states

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate smoothing
fi

# Load W&B credentials from project .env only.
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    . "$PROJECT_ROOT/.env"
    set +a
fi

echo "================================================"
echo "NER BERT Isotropic Certification"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

CHECKPOINT="${PROJECT_ROOT}/output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt"
CONFIG="${PROJECT_ROOT}/src/configs/experiments/ner_conll2003_bert_isotropic_certify.yaml"

python -m src.experiments.certify.ner \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --resume \
    --save-every-batches 2

echo "Done: $(date)"
