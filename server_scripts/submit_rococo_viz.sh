#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 2:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-viz-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/rococo-viz-%j.err
#SBATCH -J rococo-viz

set -euo pipefail
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

echo "Running on: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"

MODE="${1:-baseline}"
CFG="src/configs/experiments/rococo_clip_${MODE}.yaml"
echo "Generating visualizations for mode=$MODE …"

for ANN in coco_karpathy_test.json danger.json same_concept.json diff_concept.json rand_voca.json; do
    python -m src.experiments.eval.rococo_viz \
        --config "$CFG" \
        --ann-file "$ANN" \
        --n-show 10
done

echo "Done."
