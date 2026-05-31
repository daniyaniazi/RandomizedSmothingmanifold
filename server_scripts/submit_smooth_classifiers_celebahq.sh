#!/usr/bin/env bash
# =============================================================================
# Submit smoothed-classifier training array for CelebA-HQ
#   20 tasks: 10 sigmas x 2 modes (isotropic + manifold)
#   Each task gets its own GPU; already-done tasks exit immediately.
#
# Usage:
#   ./server_scripts/submit_smooth_classifiers_celebahq.sh [OPTIONS]
#
# Options:
#   --max-concurrent N   How many tasks run in parallel (default: 4)
#   --time TIME          Wall time per task (default: 48:00:00)
#   --partition PART     SLURM partition (default: gpu20)
#   --dry-run            Print sbatch command without submitting
#   --help               Show this help
#
# Re-submit any time; tasks whose best.pt exists skip instantly.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MAX_CONCURRENT=4
TIME="48:00:00"
PARTITION="gpu20"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --time)           TIME="$2";           shift 2 ;;
        --partition)      PARTITION="$2";      shift 2 ;;
        --dry-run)        DRY_RUN=true;        shift   ;;
        --help|-h)
            sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$PROJECT_ROOT/output/slurm"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
JOB_SCRIPT="$PROJECT_ROOT/output/slurm/smooth_celebahq_worker_${RUN_ID}.sh"

cat > "$JOB_SCRIPT" << 'WORKER'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="__PROJECT_ROOT__"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

. /BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh
conda activate smoothing

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# SIGMAS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
SIGMAS=(0.10 0.20 0.30 0.40 0.50)
N_SIGMA=${#SIGMAS[@]}

if [[ ${SLURM_ARRAY_TASK_ID} -lt ${N_SIGMA} ]]; then
    MODE="isotropic"
    SIGMA_IDX=${SLURM_ARRAY_TASK_ID}
else
    MODE="manifold"
    SIGMA_IDX=$(( SLURM_ARRAY_TASK_ID - N_SIGMA ))
fi
SIGMA=${SIGMAS[$SIGMA_IDX]}

TRAIN_CFG="src/configs/training/smile_resnet_celebahq.yaml"
BASE_CKPT="output/pretrained_model"
PIXEL_INDEX="output/smile_classification/celebahq/index/pixel/annoy/euclidean/index.ann"
STAG="sigma_${SIGMA//./_}"
CKPT_DIR="$BASE_CKPT/$MODE/celebahq/smile_resnet18_celebahq_$STAG"

echo "[Task ${SLURM_ARRAY_TASK_ID}] mode=$MODE  sigma=$SIGMA"
echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}  Start: $(date)"

if [[ -f "$CKPT_DIR/best.pt" ]]; then
    echo "SKIP -- best.pt exists"; exit 0
fi

if [[ "$MODE" == "manifold" ]] && [[ ! -f "$PIXEL_INDEX" ]]; then
    echo "SKIP manifold -- pixel index missing: $PIXEL_INDEX"; exit 0
fi

EXTRA_ARGS=""
[[ "$MODE" == "manifold" ]] && EXTRA_ARGS="--index_path $PIXEL_INDEX"

python -m src.experiments.training.resnet_smile.main \
    --config "$TRAIN_CFG" \
    --sigma "$SIGMA" \
    --aug_mode "$MODE" \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "output/smoothed_training/$MODE/celebahq/$STAG" \
    $EXTRA_ARGS

echo "Done: $(date)  ->  $CKPT_DIR"
WORKER

sed -i "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

N_TASKS=20
ARRAY_SPEC="0-$(( N_TASKS - 1 ))%${MAX_CONCURRENT}"

SBATCH_CMD="sbatch \
  --partition=$PARTITION \
  --time=$TIME \
  --array=$ARRAY_SPEC \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem-per-cpu=32G \
  --job-name=smooth-celebahq \
  --output=$PROJECT_ROOT/output/slurm/smooth-celebahq-%A_%a.out \
  --error=$PROJECT_ROOT/output/slurm/smooth-celebahq-%A_%a.err \
  $JOB_SCRIPT"

echo "============================================"
echo " CelebA-HQ smoothed-classifier training array"
echo "============================================"
echo "  Tasks      : $N_TASKS  (10 sigmas x 2 modes)"
echo "  Concurrent : $MAX_CONCURRENT"
echo "  Array spec : $ARRAY_SPEC"
echo "  Partition  : $PARTITION"
echo "  Time       : $TIME"
echo "  Worker     : $JOB_SCRIPT"
echo "============================================"

if $DRY_RUN; then
    echo "[dry-run] $SBATCH_CMD"
else
    eval "$SBATCH_CMD"
    echo ""
    echo "Submitted. Monitor with:  squeue -u \$USER"
    echo "Re-submit anytime -- completed tasks auto-skip via best.pt check."
fi
