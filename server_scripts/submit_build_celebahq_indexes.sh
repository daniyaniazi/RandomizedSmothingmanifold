#!/usr/bin/env bash
# =============================================================================
# Build ALL CelebA-HQ indexes (pixel + latent) in one go
# =============================================================================
# Submits both index build jobs. Run AFTER ResNet + VAE training complete.
#
# Usage:
#   ./submit_build_celebahq_indexes.sh [--dry-run] [--pixel-only] [--latent-only]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
PIXEL=true
LATENT=true

for arg in "$@"; do
    case $arg in
        --dry-run)      DRY_RUN=true ;;
        --pixel-only)   LATENT=false ;;
        --latent-only)  PIXEL=false ;;
    esac
done

echo ""
echo "=============================================="
echo "CelebA-HQ INDEX BUILD"
echo "  Pixel: $PIXEL | Latent: $LATENT"
echo "=============================================="
echo ""

TOTAL=0

if [ "$PIXEL" = true ]; then
    if $DRY_RUN; then
        echo "🔹 [DRY-RUN] sbatch submit_build_celebahq_pixel_index.sh"
    else
        echo "🚀 Submitting pixel index (512×512, streaming)..."
        sbatch submit_build_celebahq_pixel_index.sh
    fi
    TOTAL=$((TOTAL + 1))
fi

if [ "$LATENT" = true ]; then
    if $DRY_RUN; then
        echo "🔹 [DRY-RUN] sbatch submit_build_celebahq_latent_index.sh"
    else
        echo "🚀 Submitting latent index (1024-dim VAE)..."
        sbatch submit_build_celebahq_latent_index.sh
    fi
    TOTAL=$((TOTAL + 1))
fi

echo ""
echo "=============================================="
echo "Submitted: $TOTAL jobs"
echo "Monitor: squeue -u \$USER"
echo "=============================================="
