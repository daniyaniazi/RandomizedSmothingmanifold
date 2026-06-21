#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 4:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-embeddings-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-embeddings-%j.err
#SBATCH -J rococo-embeddings

# =============================================================================
# Generate CLIP embeddings for all COCO images and RoCOCO captions.
# Run ONCE before any evaluation or index building.
# =============================================================================

set -euo pipefail
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "Running on: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"

FORCE=""
[[ "${1:-}" == "--force" ]] && FORCE="--force"

python -m src.experiments.clip_embedding.generate_embeddings \
    --config src/configs/experiments/rococo_clip_baseline.yaml \
    $FORCE
