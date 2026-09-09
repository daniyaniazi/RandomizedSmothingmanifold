#!/usr/bin/env bash
# =============================================================================
# Submit one Slurm job per OOD attribute for CelebA OOD smile classifier training.
# Each job trains a ResNet on the non-OOD population (ood_exclude_attribute=ATTR).
#
# Usage:
#   ./submit_smile_celeba_ood_classifiers.sh [--dry-run] [--attrs "Attr1 Attr2 ..."]
#
# Output per attribute:
#   output/smile_resnet_celeba_ood_<attr>/metrics.json
#   output/pretrained_model/smile_resnet_celeba_ood_<attr>/best.pt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

PARTITION="gpu-rtx8000"
TIME="24:00:00"
GPUS=1
CPUS=8
MEM_PER_CPU="4G"
DRY_RUN=false

# Default OOD attribute list — edit or override with --attrs
OOD_ATTRS=(
    "Wearing_Hat"
    "Eyeglasses"
    "Male"
    "Wearing_Lipstick"
    "Mouth_Slightly_Open"
    "High_Cheekbones"
    "Big_Lips"
    "Narrow_Eyes"
    "Mustache"
    "No_Beard"
    "Rosy_Cheeks"
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --attrs)
            IFS=' ' read -r -a OOD_ATTRS <<< "$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"; shift 2 ;;
        --time)
            TIME="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--attrs 'Attr1 Attr2 ...'] [--partition PART] [--time TIME]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "OOD Smile Classifier Training — CelebA"
echo "Attributes: ${OOD_ATTRS[*]}"
echo "Dry-run: $DRY_RUN"
echo "=============================================="

SUBMITTED=0
SKIPPED=0

for ATTR in "${OOD_ATTRS[@]}"; do
    ATTR_LOWER="${ATTR,,}"   # lowercase
    CKPT_DIR="$PROJECT_ROOT/output/pretrained_model/smile_resnet_celeba_ood_${ATTR_LOWER}"
    BEST_PT="$CKPT_DIR/best.pt"

    if [[ -f "$BEST_PT" ]]; then
        echo "  SKIP $ATTR — best.pt already exists: $BEST_PT"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    JOB_SCRIPT="$PROJECT_ROOT/output/slurm/smile_celeba_ood_${ATTR_LOWER}.sh"
    cat > "$JOB_SCRIPT" << JOBEOF
#!/usr/bin/env bash
#SBATCH -p ${PARTITION}
#SBATCH -t 24:00:00
#SBATCH --gres gpu:${GPUS}
#SBATCH -c ${CPUS}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH -o ${PROJECT_ROOT}/output/slurm/smile-celeba-ood-${ATTR_LOWER}-%j.out
#SBATCH -e ${PROJECT_ROOT}/output/slurm/smile-celeba-ood-${ATTR_LOWER}-%j.err
#SBATCH -J smile-ood-${ATTR_LOWER}

set -euo pipefail
cd "${PROJECT_ROOT}"

if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

echo "OOD attribute: ${ATTR}"
echo "Running on: \$(hostname)"
echo "GPU: \$CUDA_VISIBLE_DEVICES"

python -m src.experiments.training.resnet_smile.main \\
    --config src/configs/training/smile_resnet_celeba.yaml \\
    --ood_attr "${ATTR}"
JOBEOF

    chmod +x "$JOB_SCRIPT"

    if $DRY_RUN; then
        echo "  [DRY-RUN] Would submit: $JOB_SCRIPT  (attr=$ATTR)"
    else
        JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
        echo "  Submitted $ATTR → job $JOB_ID"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

echo "=============================================="
echo "Submitted: $SUBMITTED  |  Skipped (done): $SKIPPED"
echo "Monitor: squeue -u \$USER"
echo "=============================================="
