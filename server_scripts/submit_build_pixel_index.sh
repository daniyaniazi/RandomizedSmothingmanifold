#!/bin/sh
#SBATCH --job-name=build_pixel_index
#SBATCH --output=logs/build_pixel_index_%j.out
#SBATCH --error=logs/build_pixel_index_%j.err
#SBATCH --partition=gpu20
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH -c 8
#SBATCH --time=12:00:00

set -eu

# ── Environment ──────────────────────────────────────────────────────────────
. /BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh
conda activate randomized_smoothing

cd /BS/dniazi_thesis/work/RandomizedSmothingmanifold
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p logs

echo "=== Building pixel-space kNN index ==="
echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python: $(which python)"

python -m src.indexing.build_ner_token_artifacts \
    --mode pixel \
    --dataset celeba \
    --config src/configs/smoothing/manifold_knn.yaml

echo "=== Done ==="
