#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 4:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/seg-baseline-celebahq-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/seg-baseline-celebahq-%j.err
#SBATCH -J seg-baseline-celebahq

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "Running on: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Parse optional --subset arg from sbatch command line
SUBSET=""
if [[ "${1:-}" == "--subset" ]]; then
    SUBSET="--subset $2"
fi

python -m src.experiments.inference.segmentation.celebahq_segmentation \
    --checkpoint output/pretrained_model/bisenet_celebahq/79999_iter.pth \
    --data-root  /BS/dniazi_thesis/static00/CelebAMask-HQ/CelebAMask-HQ \
    --output-dir output/segmentation/celebahq/baseline \
    --num-viz    10 \
    $SUBSET
