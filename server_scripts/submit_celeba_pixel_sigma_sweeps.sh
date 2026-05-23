#!/usr/bin/env bash
# =============================================================================
# SUBMIT CELEBA PIXEL CERTIFICATION SIGMA SWEEPS ONLY
# =============================================================================
# Submits 2 pixel experiment types × N sigmas (from config)
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

count_sigmas() {
    local config=$1
    python3 -c "
import yaml
with open('$1', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
vals = cfg.get('smoothing', {}).get('sigma_values', [])
print(len(vals))
"
}

list_sigmas() {
    local config=$1
    python3 -c "
import yaml
with open('$1', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
vals = cfg.get('smoothing', {}).get('sigma_values', [])
print(', '.join(str(v) for v in vals))
"
}

ISO_CFG="src/configs/experiments/certify_celeba_isotropic_pixel.yaml"
MANI_CFG="src/configs/experiments/certify_celeba_pixel.yaml"
SIGMA_COUNT=$(count_sigmas "$ISO_CFG")
SIGMA_LIST=$(list_sigmas "$ISO_CFG")
TOTAL_EXPECTED=$((SIGMA_COUNT * 2))

echo "2 experiments × ${SIGMA_COUNT} sigmas = ${TOTAL_EXPECTED} jobs"
echo "Sigmas: [${SIGMA_LIST}]"
echo "=============================================="
echo ""

# Track job count
TOTAL_JOBS=0

submit_sweep() {
    local config=$1
    local desc=$2
    local sigma_count
    sigma_count=$(count_sigmas "$config")
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 $desc"
    echo "   Config: $config"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ./submit_sigma_sweep.sh "$config" $DRY_RUN
    TOTAL_JOBS=$((TOTAL_JOBS + sigma_count))
    echo ""
}

# =============================================================================
# 1. CelebA Pixel Isotropic (Gaussian)
# =============================================================================
submit_sweep \
    "$ISO_CFG" \
    "CelebA Pixel ISOTROPIC (Gaussian baseline)"

# =============================================================================
# 2. CelebA Pixel Manifold
# =============================================================================
submit_sweep \
    "$MANI_CFG" \
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
