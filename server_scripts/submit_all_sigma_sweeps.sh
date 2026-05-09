#!/usr/bin/env bash
# =============================================================================
# SUBMIT ALL CERTIFICATION SIGMA SWEEPS
# =============================================================================
# Submits all 6 experiment types × 4 sigmas = 24 jobs total
#
# Experiments:
#   1. CelebA Latent Isotropic (Gaussian)
#   2. CelebA Latent Manifold
#   3. NER Isotropic (Gaussian)
#   4. NER Manifold
#   5. NER + Context Masking Isotropic
#   6. NER + Context Masking Manifold
#
# Usage:
#   ./submit_all_sigma_sweeps.sh [--dry-run]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=============================================="
    echo "DRY RUN MODE - No jobs will be submitted"
    echo "=============================================="
fi

echo ""
echo "=============================================="
echo "SUBMITTING ALL CERTIFICATION SIGMA SWEEPS"
echo "=============================================="
echo "6 experiments × 4 sigmas = 24 jobs"
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
# 1. CelebA Latent Isotropic (Gaussian)
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celeba_isotropic_latent_128.yaml" \
    "CelebA Latent ISOTROPIC (Gaussian baseline)"

# =============================================================================
# 2. CelebA Latent Manifold
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celeba_latent_128.yaml" \
    "CelebA Latent MANIFOLD"

# =============================================================================
# 3. NER Isotropic (Gaussian)
# =============================================================================
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_isotropic_certify.yaml" \
    "NER ISOTROPIC (Gaussian baseline)"

# =============================================================================
# 4. NER Manifold
# =============================================================================
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_certify.yaml" \
    "NER MANIFOLD"

# =============================================================================
# 5. NER + Context Masking Isotropic
# =============================================================================
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_isotropic_masking_certify.yaml" \
    "NER + Context Masking ISOTROPIC"

# =============================================================================
# 6. NER + Context Masking Manifold
# =============================================================================
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_masking_certify.yaml" \
    "NER + Context Masking MANIFOLD"

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
