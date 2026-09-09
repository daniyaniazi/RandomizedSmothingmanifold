#!/usr/bin/env bash
# =============================================================================
# Aggregate CelebA manifold ablation results into CSVs and plots.
#
# This does not run certification. Run it after:
#   server_scripts/submit_celeba_manifold_ablation.sh
#
# Usage:
#   bash server_scripts/aggregate_celeba_ablation_results.sh
#   sbatch server_scripts/aggregate_celeba_ablation_results.sh
# =============================================================================
#SBATCH -p cpu20
#SBATCH -t 24:00:00
#SBATCH --gres gpu:0
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/celeba-ablation-aggregate-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/celeba-ablation-aggregate-%j.err
#SBATCH -J celeba-ablation-aggregate

set -euo pipefail
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

python -m src.experiments.analysis.celeba_ablation_results "$@"
