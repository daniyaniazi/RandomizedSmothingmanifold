#!/usr/bin/env bash
# =============================================================================
# SUBMIT NER CERTIFICATION SIGMA SWEEPS ONLY
# =============================================================================
# Submits 4 NER experiment types × 4 sigmas = 16 jobs total
#
# Experiments:
#   1. NER Isotropic (Gaussian)
#   2. NER Manifold
#   3. NER + Context Masking Isotropic
#   4. NER + Context Masking Manifold
#
# Usage:
#   ./submit_ner_sigma_sweeps.sh [--dry-run]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
    echo "=============================================="
    echo "DRY RUN MODE - No jobs will be submitted"
    echo "=============================================="
fi

echo ""
echo "=============================================="
echo "SUBMITTING NER CERTIFICATION SIGMA SWEEPS"
echo "=============================================="
echo "4 experiments × 4 sigmas = 16 jobs"
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
# 1. NER Isotropic (Gaussian)
# =============================================================================
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_isotropic_certify.yaml" \
    "NER ISOTROPIC (Gaussian baseline)"

# =============================================================================
# 2. NER Manifold
# =============================================================================
submit_sweep \
    "src/configs/experiments/ner_conll2003_bert_certify.yaml" \
    "NER MANIFOLD"

# =============================================================================
# # 3. NER + Context Masking Isotropic
# # =============================================================================
# submit_sweep \
#     "src/configs/experiments/ner_conll2003_bert_isotropic_masking_certify.yaml" \
#     "NER + MASKING ISOTROPIC (Gaussian baseline)"

# # =============================================================================
# # 4. NER + Context Masking Manifold
# # =============================================================================
# submit_sweep \
#     "src/configs/experiments/ner_conll2003_bert_masking_certify.yaml" \
#     "NER + MASKING MANIFOLD"

# =============================================================================
# SUMMARY
# =============================================================================
echo "=============================================="
echo "SUBMISSION COMPLETE"
echo "=============================================="
echo "Total jobs submitted: $TOTAL_JOBS"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "=============================================="
