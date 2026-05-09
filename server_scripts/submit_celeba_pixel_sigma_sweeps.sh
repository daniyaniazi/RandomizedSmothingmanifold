#!/usr/bin/env bash
# =============================================================================
# SUBMIT CELEBA PIXEL CERTIFICATION SIGMA SWEEPS ONLY
# =============================================================================
# Submits 2 pixel experiment types × 4 sigmas = 8 jobs total
#
# Experiments:
#   1. CelebA Pixel Isotropic (Gaussian)
#   2. CelebA Pixel Manifold
#
# Usage:
#   ./submit_celeba_pixel_sigma_sweeps.sh [--dry-run]
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
echo "SUBMITTING CELEBA PIXEL SIGMA SWEEPS"
echo "=============================================="
echo "2 experiments × 4 sigmas = 8 jobs"
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
# 1. CelebA Pixel Isotropic (Gaussian)
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celeba_isotropic_pixel_128.yaml" \
    "CelebA Pixel ISOTROPIC (Gaussian baseline)"

# =============================================================================
# 2. CelebA Pixel Manifold
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celeba_pixel_128.yaml" \
    "CelebA Pixel MANIFOLD"

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
