#!/usr/bin/env bash
# =============================================================================
# Aggregate CelebA manifold ablation results into CSVs and plots.
#
# This does not run certification. Run it after:
#   server_scripts/submit_celeba_manifold_ablation.sh
#
# Usage:
#   bash server_scripts/aggregate_celeba_ablation_results.sh
# =============================================================================

set -euo pipefail
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

python -m src.experiments.analysis.celeba_ablation_results "$@"
