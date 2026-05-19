#!/usr/bin/env bash
# =============================================================================
# SUBMIT CELEBA ALL CERTIFICATION SIGMA SWEEPS
# =============================================================================
# Submits 4 experiment types × 4 sigmas = 16 jobs total
#
# Experiments:
#   1. CelebA Pixel Isotropic (Gaussian)
#   2. CelebA Pixel Manifold
#   3. CelebA Latent Isotropic (Gaussian)
#   4. CelebA Latent Manifold
#
# Usage:
#   ./submit_celeba_all_sigma_sweeps.sh [--dry-run]
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
echo "SUBMITTING CELEBA ALL SIGMA SWEEPS"
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
# 1. CelebA Pixel Isotropic (Gaussian)
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celeba_isotropic_pixel.yaml" \
    "CelebA Pixel ISOTROPIC (Gaussian baseline)"

# =============================================================================
# 2. CelebA Pixel Manifold
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celeba_pixel.yaml" \
    "CelebA Pixel MANIFOLD"

# =============================================================================
# 3. CelebA Latent Isotropic (Gaussian)
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celeba_isotropic_latent_128.yaml" \
    "CelebA Latent ISOTROPIC (Gaussian baseline)"

# =============================================================================
# 4. CelebA Latent Manifold
# =============================================================================
submit_sweep \
    "src/configs/experiments/certify_celeba_latent_128.yaml" \
    "CelebA Latent MANIFOLD"

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
