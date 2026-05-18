#!/usr/bin/env bash
# =============================================================================
# SUBMIT CELEBAHQ ALL CERTIFICATION SIGMA SWEEPS
# =============================================================================
# Submits 4 experiment types × 4 sigmas = 16 jobs total
#
# Experiments:
#   1. CelebA-HQ Pixel Isotropic (Gaussian)
#   2. CelebA-HQ Pixel Manifold
#   3. CelebA-HQ Latent Isotropic (Gaussian)
#   4. CelebA-HQ Latent Manifold
#
# Usage:
#   ./submit_celebahq_all_sigma_sweeps.sh [--dry-run]
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
echo "SUBMITTING CELEBAHQ ALL SIGMA SWEEPS"
echo "=============================================="
echo "4 experiments × 4 sigmas = 16 jobs"
echo "Sigmas: [0.25, 0.50, 0.75, 1.00]"
echo "=============================================="
echo ""

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
# 1. CelebA-HQ Pixel Isotropic (Gaussian)
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celebahq_isotropic_pixel.yaml" \
    "CelebA-HQ Pixel ISOTROPIC (Gaussian baseline)"

# =============================================================================
# 2. CelebA-HQ Pixel Manifold
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celebahq_pixel.yaml" \
    "CelebA-HQ Pixel MANIFOLD"

# =============================================================================
# 3. CelebA-HQ Latent Isotropic (Gaussian)
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celebahq_isotropic_latent.yaml" \
    "CelebA-HQ Latent ISOTROPIC (Gaussian baseline)"

# =============================================================================
# 4. CelebA-HQ Latent Manifold
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celebahq_latent.yaml" \
    "CelebA-HQ Latent MANIFOLD"

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
