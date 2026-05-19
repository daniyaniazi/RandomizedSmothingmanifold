#!/usr/bin/env bash
# =============================================================================
# PRE-BUILD INDEXES using src.experiments.indexing.celeba_images
# Usage:
#   ./submit_build_indexes.sh all               # all 4 indexes
#   ./submit_build_indexes.sh celeba             # CelebA pixel + latent
#   ./submit_build_indexes.sh celebahq           # CelebaHQ pixel + latent
#   ./submit_build_indexes.sh celeba-pixel       # CelebA pixel only
#   ./submit_build_indexes.sh celeba-latent      # CelebA latent only
#   ./submit_build_indexes.sh celebahq-pixel     # CelebaHQ pixel only
#   ./submit_build_indexes.sh celebahq-latent    # CelebaHQ latent only
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
PARTITION="gpu20"
TIME="12:00:00"
GPUS=1
CPUS=8
MEM_PER_CPU="8G"
CONFIGS_DIR="${PROJECT_ROOT}/src/configs/experiments"

WRAP_PREFIX="cd $PROJECT_ROOT && export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
. /BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh && conda activate smoothing"

mkdir -p "$PROJECT_ROOT/output/slurm"

submit_index() {
    local config=$1
    local space=$2
    local job_name=$3
    local mem=${4:-"8G"}  # default 8G, override for large jobs

    sbatch --partition=$PARTITION --time=$TIME --gres=gpu:$GPUS \
        --cpus-per-task=$CPUS --mem-per-cpu=$mem \
        --job-name="$job_name" \
        --output=$PROJECT_ROOT/output/slurm/${job_name}-%j.out \
        --error=$PROJECT_ROOT/output/slurm/${job_name}-%j.err \
        --wrap="$WRAP_PREFIX && python -m src.experiments.indexing.celeba_images --config $config --space $space"

    echo "  ✅ $job_name (mem-per-cpu=$mem)"
}

TARGET=${1:-"all"}

echo "Submitting index build jobs: $TARGET"
echo ""

case $TARGET in
    celeba-pixel)
        submit_index "$CONFIGS_DIR/certify_celeba_isotropic_pixel.yaml" pixel "idx-celeba-pixel" "32G"
        ;;
    celeba-latent)
        submit_index "$CONFIGS_DIR/certify_celeba_isotropic_latent_128.yaml" latent "idx-celeba-latent"
        ;;
    celeba)
        submit_index "$CONFIGS_DIR/certify_celeba_isotropic_pixel.yaml" pixel "idx-celeba-pixel" "32G"
        submit_index "$CONFIGS_DIR/certify_celeba_isotropic_latent_128.yaml" latent "idx-celeba-latent"
        ;;
    celebahq-pixel)
        submit_index "$CONFIGS_DIR/certify_celebahq_isotropic_pixel.yaml" pixel "idx-celebahq-pixel" "32G"
        ;;
    celebahq-latent)
        submit_index "$CONFIGS_DIR/certify_celebahq_isotropic_latent.yaml" latent "idx-celebahq-latent"
        ;;
    celebahq)
        submit_index "$CONFIGS_DIR/certify_celebahq_isotropic_pixel.yaml" pixel "idx-celebahq-pixel" "32G"
        submit_index "$CONFIGS_DIR/certify_celebahq_isotropic_latent.yaml" latent "idx-celebahq-latent"
        ;;
    all)
        submit_index "$CONFIGS_DIR/certify_celeba_isotropic_pixel.yaml" pixel "idx-celeba-pixel" "32G"
        submit_index "$CONFIGS_DIR/certify_celeba_isotropic_latent_128.yaml" latent "idx-celeba-latent"
        submit_index "$CONFIGS_DIR/certify_celebahq_isotropic_pixel.yaml" pixel "idx-celebahq-pixel" "32G"
        submit_index "$CONFIGS_DIR/certify_celebahq_isotropic_latent.yaml" latent "idx-celebahq-latent"
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: ./submit_build_indexes.sh {all|celeba|celebahq|celeba-pixel|celeba-latent|celebahq-pixel|celebahq-latent}"
        exit 1
        ;;
esac

echo ""
echo "══════════════════════════════════════════"
echo "4 index-build jobs submitted."
echo "Wait for completion, then run:"
echo "  ./submit_chained.sh celeba"
echo "  ./submit_chained.sh celebahq"
echo "══════════════════════════════════════════"
