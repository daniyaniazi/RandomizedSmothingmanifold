#!/usr/bin/env bash
# =============================================================================
# NER LAYER SWEEP — σ=0.50 fixed, layers 0, 3, 6, 9 + last
# =============================================================================
# Submits one SLURM job per layer×mode. No sigma sweep — σ is fixed at 0.50
# so that embedding extraction and kNN index building per layer stay clean.
#
# Total: 5 layers × 2 modes = 10 jobs
#
# Usage:
#   ./submit_ner_layer_sweeps.sh [--dry-run] [--manifold-only | --isotropic-only]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
CHECKPOINT="${PROJECT_ROOT}/output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt"
CONFIGS_DIR="${PROJECT_ROOT}/src/configs/experiments"

DRY_RUN=false
MANIFOLD=true
ISOTROPIC=true

for arg in "$@"; do
    case $arg in
        --dry-run)    DRY_RUN=true ;;
        --manifold-only)  ISOTROPIC=false ;;
        --isotropic-only) MANIFOLD=false ;;
    esac
done

if $DRY_RUN; then
    echo "=============================================="
    echo "DRY RUN MODE"
    echo "=============================================="
fi

TOTAL=0

submit() {
    local config=$1
    local desc=$2
    local job_name=$3

    if $DRY_RUN; then
        echo "🔹 [DRY-RUN] $desc"
        echo "   config: $config"
    else
        echo "🚀 $desc"
        sbatch <<EOF
#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=16G
#SBATCH -o ${PROJECT_ROOT}/output/slurm/${job_name}-%j.out
#SBATCH -e ${PROJECT_ROOT}/output/slurm/${job_name}-%j.err
#SBATCH -J ${job_name}

set -euo pipefail
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:\${PYTHONPATH:-}"

if [ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]; then
    . "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; . "$PROJECT_ROOT/.env"; set +a
fi

echo "Running: $desc"
echo "Host: \$(hostname) | GPU: \$CUDA_VISIBLE_DEVICES | Start: \$(date)"

python -m src.experiments.certify.ner \\
    --config   "$config" \\
    --checkpoint "$CHECKPOINT" \\
    --split    test \\
    --resume \\
    --save-every-batches 2

echo "Done: \$(date)"
EOF
    fi
    TOTAL=$((TOTAL + 1))
}

LAYERS=(0 3 6 9)

echo ""
echo "=============================================="
echo "NER LAYER SWEEP — σ=0.50"
echo "Layers: ${LAYERS[*]} + last"
echo "=============================================="
echo ""

# ── Manifold ──────────────────────────────────────
if [ "$MANIFOLD" = true ]; then
    echo "── MANIFOLD ──"
    for L in "${LAYERS[@]}"; do
        submit "${CONFIGS_DIR}/ner_conll2003_bert_certify_layer_${L}.yaml" \
               "NER Manifold — Layer ${L} (σ=0.50)" \
               "ner-manifold-L${L}"
    done
    submit "${CONFIGS_DIR}/ner_conll2003_bert_certify.yaml" \
           "NER Manifold — Last layer (σ=0.50)" \
           "ner-manifold-Llast"
    echo ""
fi

# ── Isotropic ─────────────────────────────────────
if [ "$ISOTROPIC" = true ]; then
    echo "── ISOTROPIC ──"
    for L in "${LAYERS[@]}"; do
        submit "${CONFIGS_DIR}/ner_conll2003_bert_isotropic_certify_layer_${L}.yaml" \
               "NER Isotropic — Layer ${L} (σ=0.50)" \
               "ner-iso-L${L}"
    done
    submit "${CONFIGS_DIR}/ner_conll2003_bert_isotropic_certify.yaml" \
           "NER Isotropic — Last layer (σ=0.50)" \
           "ner-iso-Llast"
    echo ""
fi

echo "=============================================="
echo "Submitted: $TOTAL jobs"
if ! $DRY_RUN; then
    echo "Monitor: squeue -u \$USER"
    echo ""
    echo "Outputs by layer:"
    for L in "${LAYERS[@]}"; do
        echo "  .../token_layer_embeddings/train/layer_${L}/"
    done
    echo "  .../token_layer_embeddings/train/last/"
fi
echo "=============================================="
