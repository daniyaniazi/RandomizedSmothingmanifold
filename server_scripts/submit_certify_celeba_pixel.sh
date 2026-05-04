#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/certify-celeba-pixel-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/certify-celeba-pixel-%j.err
#SBATCH -J certify-celeba-pixel

# Checkpoint support: if job gets killed and resubmitted, it will resume from last checkpoint.
# To force resume, set RESUME=true when submitting: sbatch --export=RESUME=true submit_certify_celeba_pixel.sh

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting CelebA Pixel Certification"

# Check if resume flag is set (useful for resubmitting after timeout)
CONFIG_PATH="src/configs/experiments/certify_celeba_pixel_128.yaml"
RESUME_FLAG="${RESUME:-false}"

if [[ "$RESUME_FLAG" == "true" ]]; then
    echo "Resume mode enabled - will continue from last checkpoint"
    # Create a temp config with resume enabled
    TEMP_CONFIG=$(mktemp /tmp/certify_config_XXXXXX.yaml)
    sed 's/resume: false/resume: true/' "$CONFIG_PATH" > "$TEMP_CONFIG"
    python -m src.certify.celeba_workflow --config "$TEMP_CONFIG"
    rm -f "$TEMP_CONFIG"
else
    python -m src.certify.celeba_workflow --config "$CONFIG_PATH"
fi
