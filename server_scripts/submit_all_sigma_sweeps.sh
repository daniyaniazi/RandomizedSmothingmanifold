#!/usr/bin/env bash
# =============================================================================
# SUBMIT ALL CERTIFICATION SIGMA SWEEPS
# =============================================================================
# Submits all 12 experiment types × 4 sigmas = 48 jobs total
#
# CelebA Experiments:
#   1. CelebA Pixel Isotropic (Gaussian)
#   2. CelebA Pixel Manifold
#   3. CelebA Latent Isotropic (Gaussian)
#   4. CelebA Latent Manifold
#
# CelebA-HQ Experiments:
#   5. CelebA-HQ Pixel Isotropic (Gaussian)
#   6. CelebA-HQ Pixel Manifold
#   7. CelebA-HQ Latent Isotropic (Gaussian)
#   8. CelebA-HQ Latent Manifold
#
# NER Experiments:
#   9.  NER Isotropic (Gaussian)
#   10. NER Manifold
#   11. NER + Context Masking Isotropic
#   12. NER + Context Masking Manifold
#
# Usage:
#   ./submit_all_sigma_sweeps.sh [--dry-run]
#   ./submit_all_sigma_sweeps.sh [--dry-run] [--celeba-only | --celebahq-only | --ner-only]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=""
CELEBA_ONLY=false
CELEBAHQ_ONLY=false
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
        --celebahq-only)
            CELEBAHQ_ONLY=true
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
echo "12 experiments × 4 sigmas = 48 jobs total"
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
if [ "$NER_ONLY" = false ] && [ "$CELEBAHQ_ONLY" = false ]; then

echo "=============================================="
echo "CELEBA EXPERIMENTS (Pixel + Latent)"
echo "=============================================="

# 1. CelebA Pixel Isotropic (Gaussian)
submit_sweep \
    "src/configs/experiments/certify_celeba_isotropic_pixel.yaml" \
    "CelebA Pixel ISOTROPIC (Gaussian baseline)"

# 2. CelebA Pixel Manifold
submit_sweep \
    "src/configs/experiments/certify_celeba_pixel.yaml" \
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
# CELEBAHQ EXPERIMENTS
# =============================================================================
if [ "$NER_ONLY" = false ] && [ "$CELEBA_ONLY" = false ]; then

echo "=============================================="
echo "CELEBAHQ EXPERIMENTS (Pixel + Latent)"
echo "=============================================="

# 5. CelebA-HQ Pixel Isotropic (Gaussian)
submit_sweep \
    "src/configs/experiments/certify_celebahq_isotropic_pixel.yaml" \
    "CelebA-HQ Pixel ISOTROPIC (Gaussian baseline)"

# 6. CelebA-HQ Pixel Manifold
submit_sweep \
    "src/configs/experiments/certify_celebahq_pixel.yaml" \
    "CelebA-HQ Pixel MANIFOLD"

# 7. CelebA-HQ Latent Isotropic (Gaussian)
submit_sweep \
    "src/configs/experiments/certify_celebahq_isotropic_latent.yaml" \
    "CelebA-HQ Latent ISOTROPIC (Gaussian baseline)"

# 8. CelebA-HQ Latent Manifold
submit_sweep \
    "src/configs/experiments/certify_celebahq_latent.yaml" \
    "CelebA-HQ Latent MANIFOLD"

fi

# =============================================================================
# NER EXPERIMENTS
# =============================================================================
if [ "$CELEBA_ONLY" = false ] && [ "$CELEBAHQ_ONLY" = false ]; then

echo "=============================================="
echo "NER EXPERIMENTS"
echo "=============================================="

# 9. NER Isotropic (Gaussian)
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_isotropic_certify.yaml" \
    "NER ISOTROPIC (Gaussian baseline)"

# 10. NER Manifold
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_certify.yaml" \
    "NER MANIFOLD"

# 11. NER + Context Masking Isotropic
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_isotropic_masking_certify.yaml" \
    "NER + Context Masking ISOTROPIC"

# 12. NER + Context Masking Manifold
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
    echo "  - output/smile_classification/celebahq/certify/..."
    echo "  - output/ner_conll2003_bert/certify/..."
else
    echo "This was a DRY RUN - no jobs were submitted"
    echo "Remove --dry-run to actually submit"
fi
echo "=============================================="
