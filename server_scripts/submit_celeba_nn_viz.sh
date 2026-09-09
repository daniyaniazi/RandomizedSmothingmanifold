#!/usr/bin/env bash
# Submit pixel + latent NN visualization job
PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"

sbatch \
  --partition=gpu-rtx8000 \
  --time=24:00:00 \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem-per-cpu=8G \
  --job-name=celeba-nn-viz \
  --output=$PROJECT_ROOT/output/slurm/celeba-nn-viz-%j.out \
  --error=$PROJECT_ROOT/output/slurm/celeba-nn-viz-%j.err \
  --wrap="cd $PROJECT_ROOT && \
    export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
    . /BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh && \
    conda activate smoothing && \
    python -m src.experiments.analysis.celeba_nn_viz \
      --output-dir output/analysis/nn_viz \
      --n-attrs 40 \
      --k 5"

echo "Submitted. Monitor with: squeue -u \$USER"
echo "Results will be saved to: $PROJECT_ROOT/output/analysis/nn_viz/"
