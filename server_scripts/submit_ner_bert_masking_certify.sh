#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=16G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/ner-bert-masking-certify-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/ner-bert-masking-certify-%j.err
#SBATCH -J ner-bert-mask-cert

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "Running on: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

CHECKPOINT="${PROJECT_ROOT}/output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt"
CONFIG="${PROJECT_ROOT}/src/configs/experiments/ner_conll2003_bert_masking_certify.yaml"

python -m src.experiments.certify.ner \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --resume \
    --save-every-batches 2 \
    --log-every-batches 1

echo "Done: $(date)"
