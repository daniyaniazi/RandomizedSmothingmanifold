#!/usr/bin/env bash
# =============================================================================
# Pre-compute local PCA cache for every CelebA training image.
# Runs kNN lookup + SVD once per image, saves mean/evals/evecs to .npz.
# Must be run BEFORE submit_smooth_classifiers_celeba.sh (manifold mode).
# Re-runnable: already-cached images are skipped automatically.
#
# Usage:
#   ./server_scripts/submit_precompute_pca_cache_celeba.sh [--knn-k K] [--dry-run]
#
# Output:
#   output/pca_cache/celeba/knn500_img_align_celeba_pca_cache.npz
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
PARTITION="gpu20"
TIME="48:00:00"
GPUS=1
CPUS=8
MEM_PER_CPU="8G"

KNN_K=500
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --knn-k)    KNN_K="$2";   shift 2 ;;
        --dry-run)  DRY_RUN=true; shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

IMAGE_DIR="/BS/databases08/CelebA/img_align_celeba"
INDEX_PATH="${PROJECT_ROOT}/output/smile_classification/celeba/index/pixel/annoy/euclidean/index.ann"
OUTPUT_DIR="${PROJECT_ROOT}/output/pca_cache/celeba"
CACHE_FILE="${OUTPUT_DIR}/knn${KNN_K}_img_align_celeba_pca_cache.npz"

WRAP_PREFIX="cd $PROJECT_ROOT && export PYTHONPATH=$PROJECT_ROOT:\${PYTHONPATH:-} && \
. /BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh && conda activate smoothing"

mkdir -p "${PROJECT_ROOT}/output/slurm"

if [[ -f "$CACHE_FILE" ]]; then
    echo "Cache already exists: $CACHE_FILE"
    echo "Re-running will skip already-cached images (safe to re-submit)."
fi

JOB_NAME="pca-cache-celeba"

SBATCH_CMD="sbatch \
    --partition=${PARTITION} \
    --time=${TIME} \
    --gres=gpu:${GPUS} \
    --cpus-per-task=${CPUS} \
    --mem-per-cpu=${MEM_PER_CPU} \
    --job-name=${JOB_NAME} \
    --output=${PROJECT_ROOT}/output/slurm/${JOB_NAME}-%j.out \
    --error=${PROJECT_ROOT}/output/slurm/${JOB_NAME}-%j.err \
    --wrap=\"${WRAP_PREFIX} && \
        python -m src.experiments.training.resnet_smile.precompute_pca_cache \
            --image_dir ${IMAGE_DIR} \
            --index_path ${INDEX_PATH} \
            --image_size 224 \
            --knn_k ${KNN_K} \
            --output_dir ${OUTPUT_DIR} \
            --workers ${CPUS}\""

echo "Submitting PCA cache precompute: CelebA  knn_k=${KNN_K}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] $SBATCH_CMD"
else
    eval "$SBATCH_CMD"
    echo "  OK ${JOB_NAME} (knn_k=${KNN_K})"
    echo ""
    echo "Next step after this completes:"
    echo "  ./server_scripts/submit_smooth_classifiers_celeba.sh"
    echo "  Set in your YAML: smoothing_aug.pca_cache_path: ${CACHE_FILE}"
fi