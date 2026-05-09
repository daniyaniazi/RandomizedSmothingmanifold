#!/usr/bin/env bash
# =============================================================================
# SUBMIT ALL CERTIFICATION SIGMA SWEEPS
# =============================================================================
# Submits all 10 experiment types × 4 sigmas = 40 jobs total
#
# CelebA Experiments:
#   1. CelebA Pixel Isotropic (Gaussian)
#   2. CelebA Pixel Manifold
#   3. CelebA Latent Isotropic (Gaussian)
#   4. CelebA Latent Manifold
#
# NER Experiments:
#   5. NER Isotropic (Gaussian)
#   6. NER Manifold
#   7. NER + Context Masking Isotropic
#   8. NER + Context Masking Manifold
#
# Usage:
#   ./submit_all_sigma_sweeps.sh [--dry-run]
#   ./submit_all_sigma_sweeps.sh [--dry-run] [--celeba-only | --ner-only]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=""
CELEBA_ONLY=false
NER_ONLY=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN="--dry-run"
            echo "=============================================="
            echo "DRY RUN MODE - No jobs will be submitted"
            echo "=============================================="
            ;;
        --celeba-only)
            CELEBA_ONLY=true
            ;;
        --ner-only)
            NER_ONLY=true
            ;;
    esac
done

echo ""
echo "=============================================="
echo "SUBMITTING ALL CERTIFICATION SIGMA SWEEPS"
echo "=============================================="
echo "10 experiments × 4 sigmas = 40 jobs total"
echo "Sigmas: [0.25, 0.50, 0.75, 1.00]"
echo "=============================================="
echo ""

# Track job count
TOTAL_JOBS=0

submit_sweep() {
    local config=$1
    local desc=$2
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 $desc"
    echo "   Config: $config"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ./submit_sigma_sweep.sh "$config" $DRY_RUN
    TOTAL_JOBS=$((TOTAL_JOBS + 4))
    echo ""
}

# =============================================================================
# CELEBA EXPERIMENTS
# =============================================================================
if [ "$NER_ONLY" = false ]; then

echo "=============================================="
echo "CELEBA EXPERIMENTS (Pixel + Latent)"
echo "=============================================="

# 1. CelebA Pixel Isotropic (Gaussian)
submit_sweep \
    "src/configs/experiments/certify_celeba_isotropic_pixel_128.yaml" \
    "CelebA Pixel ISOTROPIC (Gaussian baseline)"

# 2. CelebA Pixel Manifold
submit_sweep \
    "src/configs/experiments/certify_celeba_pixel_128.yaml" \
    "CelebA Pixel MANIFOLD"

# 3. CelebA Latent Isotropic (Gaussian)
submit_sweep \
    "src/configs/experiments/certify_celeba_isotropic_latent_128.yaml" \
    "CelebA Latent ISOTROPIC (Gaussian baseline)"

# 4. CelebA Latent Manifold
submit_sweep \
    "src/configs/experiments/certify_celeba_latent_128.yaml" \
    "CelebA Latent MANIFOLD"

fi

# =============================================================================
# NER EXPERIMENTS
# =============================================================================
if [ "$CELEBA_ONLY" = false ]; then

echo "=============================================="
echo "NER EXPERIMENTS"
echo "=============================================="

# 5. NER Isotropic (Gaussian)
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_isotropic_certify.yaml" \
    "NER ISOTROPIC (Gaussian baseline)"

# 6. NER Manifold
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_certify.yaml" \
    "NER MANIFOLD"

# 7. NER + Context Masking Isotropic
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_isotropic_masking_certify.yaml" \
    "NER + Context Masking ISOTROPIC"

# 8. NER + Context Masking Manifold
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_masking_certify.yaml" \
    "NER + Context Masking MANIFOLD"

fi

echo "=============================================="
echo "SUBMISSION COMPLETE"
echo "=============================================="
echo "Total jobs: $TOTAL_JOBS"
echo ""
if [[ -z "$DRY_RUN" ]]; then
    echo "Monitor with: squeue -u \$USER"
    echo ""
    echo "Results will be saved to:"
    echo "  - output/smile_classification/celeba/certify/..."
    echo "  - output/ner_conll2003_bert/certify/..."
else
    echo "This was a DRY RUN - no jobs were submitted"
    echo "Remove --dry-run to actually submit"
fi
echo "=============================================="
