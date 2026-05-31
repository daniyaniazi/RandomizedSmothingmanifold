#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train 44 smoothed classifiers for CelebA  (re-submittable, max 48 h)
#   22 sigmas × 2 modes (isotropic + manifold)
#
# Re-submit safely at any time — already-finished sigma runs are skipped
# (detected by presence of best.pt).  In-progress runs resume via latest.pt.
#
# Submit:
#   sbatch server_scripts/submit_smooth_classifiers_celeba.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=16G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/smooth-classifiers-celeba-%j.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/smooth-classifiers-celeba-%j.err
#SBATCH -J smooth-clf-celeba

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ── Environment ───────────────────────────────────────────────────────────────
if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "Host : $(hostname)"
echo "GPU  : ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Start: $(date)"

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_CFG="src/configs/training/smile_resnet_celeba.yaml"
BASE_CKPT="output/pretrained_model"
PIXEL_INDEX="output/smile_classification/celeba/index/pixel/annoy/euclidean/index.ann"

# SIGMAS=(0.01 0.03 0.05 0.07 0.10 0.12 0.15 0.17 0.20 0.22 0.25 0.27 0.30 0.32 0.35 0.37 0.40 0.42 0.45 0.47 0.50 0.55)
SIGMAS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
MODES=(isotropic manifold)

skipped=0; trained=0; failed=0

for MODE in "${MODES[@]}"; do
    echo ""
    echo "════════  MODE: $MODE  ════════  $(date)"

    if [[ "$MODE" == "manifold" ]] && [[ ! -f "$PIXEL_INDEX" ]]; then
        echo "SKIP manifold — pixel index not found at $PIXEL_INDEX"
        echo "Build it first (run any certify job with build_if_missing: true)."
        continue
    fi

    for SIGMA in "${SIGMAS[@]}"; do
        STAG="sigma_${SIGMA//./_}"
        CKPT_DIR="$BASE_CKPT/$MODE/celeba/smile_resnet18_celeba_$STAG"
        BEST_PT="$CKPT_DIR/best.pt"

        if [[ -f "$BEST_PT" ]]; then
            echo "  SKIP  $MODE  σ=$SIGMA  (best.pt exists)"
            ((skipped++)) || true
            continue
        fi

        echo "  TRAIN $MODE  σ=$SIGMA  → $CKPT_DIR"

        EXTRA_ARGS=""
        if [[ "$MODE" == "manifold" ]]; then
            EXTRA_ARGS="--index_path $PIXEL_INDEX"
        fi

        set +e
        python -m src.experiments.training.resnet_smile.main \
            --config "$TRAIN_CFG" \
            --sigma "$SIGMA" \
            --aug_mode "$MODE" \
            --ckpt_dir "$CKPT_DIR" \
            --output_dir "output/smoothed_training/$MODE/celeba/$STAG" \
            $EXTRA_ARGS
        EXIT_CODE=$?
        set -e

        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "  WARN  $MODE  σ=$SIGMA  exited with code $EXIT_CODE"
            ((failed++)) || true
        else
            ((trained++)) || true
        fi
    done
done

echo ""
echo "Done: $(date)"
echo "  trained=$trained  skipped=$skipped  failed=$failed"
echo "  Re-submit the same job to continue any remaining/failed runs."
  echo "  Checkpoints: $BASE_CKPT/{isotropic,manifold}/celeba/smile_resnet18_celeba_sigma_*/best.pt"