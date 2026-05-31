#!/usr/bin/env bash
# =============================================================================
# Submit smoothed-classifier training array for CelebA
#   20 tasks: 10 sigmas × 2 modes (isotropic + manifold)
#   Each task gets its own GPU; already-done tasks exit immediately.
#
# Usage:
#   ./server_scripts/submit_smooth_classifiers_celeba.sh [OPTIONS]
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

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_CONCURRENT=4
TIME="48:00:00"
PARTITION="gpu20"
DRY_RUN=false

# ── Parse args ────────────────────────────────────────────────────────────────
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

# ── Generate the per-task worker script ───────────────────────────────────────
JOB_SCRIPT="$PROJECT_ROOT/output/slurm/smooth_celeba_worker_${RUN_ID}.sh"
cat > "$JOB_SCRIPT" << 'WORKER'
#!/usr/bin/env bash
# Auto-generated worker — do not edit directly
set -euo pipefail

PROJECT_ROOT="__PROJECT_ROOT__"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# Map SLURM_ARRAY_TASK_ID → (mode, sigma)
SIGMAS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
N_SIGMA=${#SIGMAS[@]}

if [[ ${SLURM_ARRAY_TASK_ID} -lt ${N_SIGMA} ]]; then
    MODE="isotropic"
    SIGMA_IDX=${SLURM_ARRAY_TASK_ID}
else
    MODE="manifold"
    SIGMA_IDX=$(( SLURM_ARRAY_TASK_ID - N_SIGMA ))
fi
SIGMA=${SIGMAS[$SIGMA_IDX]}

TRAIN_CFG="src/configs/training/smile_resnet_celeba.yaml"
BASE_CKPT="output/pretrained_model"
PIXEL_INDEX="output/smile_classification/celeba/index/pixel/annoy/euclidean/index.ann"
STAG="sigma_${SIGMA//./_}"
CKPT_DIR="$BASE_CKPT/$MODE/celeba/smile_resnet18_celeba_$STAG"

echo "[Task ${SLURM_ARRAY_TASK_ID}] mode=$MODE  sigma=$SIGMA"
echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}  Start: $(date)"

if [[ -f "$CKPT_DIR/best.pt" ]]; then
    echo "SKIP — best.pt already exists"; exit 0
fi

if [[ "$MODE" == "manifold" ]] && [[ ! -f "$PIXEL_INDEX" ]]; then
    echo "SKIP manifold — pixel index missing: $PIXEL_INDEX"; exit 0
fi

EXTRA_ARGS=""
[[ "$MODE" == "manifold" ]] && EXTRA_ARGS="--index_path $PIXEL_INDEX"

python -m src.experiments.training.resnet_smile.main \
    --config "$TRAIN_CFG" \
    --sigma "$SIGMA" \
    --aug_mode "$MODE" \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "output/smoothed_training/$MODE/celeba/$STAG" \
    $EXTRA_ARGS

echo "Done: $(date)  →  $CKPT_DIR"
WORKER

sed -i "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

# ── Submit ────────────────────────────────────────────────────────────────────
N_TASKS=20   # 10 sigmas × 2 modes
ARRAY_SPEC="0-$(( N_TASKS - 1 ))%${MAX_CONCURRENT}"

SBATCH_CMD="sbatch \
  --partition=$PARTITION \
  --time=$TIME \
  --array=$ARRAY_SPEC \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem-per-cpu=16G \
  --job-name=smooth-celeba \
  --output=$PROJECT_ROOT/output/slurm/smooth-celeba-%A_%a.out \
  --error=$PROJECT_ROOT/output/slurm/smooth-celeba-%A_%a.err \
  $JOB_SCRIPT"

echo "============================================"
echo " CelebA smoothed-classifier training array"
echo "============================================"
echo "  Tasks      : $N_TASKS  (10 sigmas × 2 modes)"
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
    echo "Re-submit anytime — completed tasks auto-skip via best.pt check."
fi

#   10 sigmas × 2 modes (isotropic + manifold)
#   Each array task = one (mode, sigma) pair on its own GPU
#
# Array indices:
#    0–9  → isotropic, sigmas[0–9]
#   10–19 → manifold,  sigmas[0–9]  (skipped if pixel index missing)
#
# Submit:
#   sbatch server_scripts/submit_smooth_classifiers_celeba.sh
#
# Re-submit any time — tasks whose best.pt already exists exit immediately.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH --array=0-19
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=16G
#SBATCH -o /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/smooth-celeba-%A_%a.out
#SBATCH -e /BS/dniazi_thesis/work/RandomizedSmothingmanifold/output/slurm/smooth-celeba-%A_%a.err
#SBATCH -J smooth-celeba

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

# ── Map array index → (mode, sigma) ──────────────────────────────────────────
SIGMAS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
N_SIGMA=${#SIGMAS[@]}   # 10

if [[ ${SLURM_ARRAY_TASK_ID} -lt ${N_SIGMA} ]]; then
    MODE="isotropic"
    SIGMA_IDX=${SLURM_ARRAY_TASK_ID}
else
    MODE="manifold"
    SIGMA_IDX=$(( SLURM_ARRAY_TASK_ID - N_SIGMA ))
fi
SIGMA=${SIGMAS[$SIGMA_IDX]}

TRAIN_CFG="src/configs/training/smile_resnet_celeba.yaml"
BASE_CKPT="output/pretrained_model"
PIXEL_INDEX="output/smile_classification/celeba/index/pixel/annoy/euclidean/index.ann"

STAG="sigma_${SIGMA//./_}"
CKPT_DIR="$BASE_CKPT/$MODE/celeba/smile_resnet18_celeba_$STAG"

echo "Task ${SLURM_ARRAY_TASK_ID}: mode=$MODE  sigma=$SIGMA"
echo "Host : $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-<not set>}  Start: $(date)"

# ── Skip if already done ──────────────────────────────────────────────────────
if [[ -f "$CKPT_DIR/best.pt" ]]; then
    echo "SKIP — best.pt already exists at $CKPT_DIR"
    exit 0
fi

# ── Manifold requires pixel index ─────────────────────────────────────────────
if [[ "$MODE" == "manifold" ]] && [[ ! -f "$PIXEL_INDEX" ]]; then
    echo "SKIP manifold — pixel index not found at $PIXEL_INDEX"
    echo "Build it first, then re-submit."
    exit 0
fi

# ── Train ─────────────────────────────────────────────────────────────────────
EXTRA_ARGS=""
if [[ "$MODE" == "manifold" ]]; then
    EXTRA_ARGS="--index_path $PIXEL_INDEX"
fi

python -m src.experiments.training.resnet_smile.main \
    --config "$TRAIN_CFG" \
    --sigma "$SIGMA" \
    --aug_mode "$MODE" \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "output/smoothed_training/$MODE/celeba/$STAG" \
    $EXTRA_ARGS

echo "Done: $(date)  →  $CKPT_DIR"


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