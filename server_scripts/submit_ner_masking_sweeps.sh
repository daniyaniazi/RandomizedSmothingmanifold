#!/usr/bin/env bash
# =============================================================================
# NER MASKING SWEEP — σ=0.50, masking modes: context, entity, hybrid
# =============================================================================
# Each masking mode × {manifold, isotropic} = 6 jobs
#
# Masking modes:
#   context  — mask random context tokens (already have results)
#   entity   — mask entity tokens
#   hybrid   — mask both context and entity tokens
#
# Usage:
#   ./submit_ner_masking_sweeps.sh [--dry-run] [--manifold-only | --isotropic-only]
#   ./submit_ner_masking_sweeps.sh --entity-only
#   ./submit_ner_masking_sweeps.sh --skip-context   # skip context (already done)
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
DO_CONTEXT=true
DO_ENTITY=true
DO_HYBRID=true

for arg in "$@"; do
    case $arg in
        --dry-run)         DRY_RUN=true ;;
        --manifold-only)   ISOTROPIC=false ;;
        --isotropic-only)  MANIFOLD=false ;;
        --entity-only)     DO_CONTEXT=false; DO_HYBRID=false ;;
        --hybrid-only)     DO_CONTEXT=false; DO_ENTITY=false ;;
        --skip-context)    DO_CONTEXT=false ;;
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

echo ""
echo "=============================================="
echo "NER MASKING SWEEP — σ=0.50"
echo "Modes: context, entity, hybrid"
echo "=============================================="
echo ""

# ── Context masking ───────────────────────────────
if [ "$DO_CONTEXT" = true ]; then
    echo "── CONTEXT MASKING ──"
    if [ "$MANIFOLD" = true ]; then
        submit "${CONFIGS_DIR}/ner_conll2003_bert_masking_certify.yaml" \
               "NER Manifold + Context Masking (σ=0.50)" \
               "ner-mask-ctx-mani"
    fi
    if [ "$ISOTROPIC" = true ]; then
        submit "${CONFIGS_DIR}/ner_conll2003_bert_isotropic_masking_certify.yaml" \
               "NER Isotropic + Context Masking (σ=0.50)" \
               "ner-mask-ctx-iso"
    fi
    echo ""
fi

# ── Entity masking ────────────────────────────────
if [ "$DO_ENTITY" = true ]; then
    echo "── ENTITY MASKING ──"
    if [ "$MANIFOLD" = true ]; then
        submit "${CONFIGS_DIR}/ner_conll2003_bert_masking_entity_certify.yaml" \
               "NER Manifold + Entity Masking (σ=0.50)" \
               "ner-mask-ent-mani"
    fi
    if [ "$ISOTROPIC" = true ]; then
        submit "${CONFIGS_DIR}/ner_conll2003_bert_isotropic_masking_entity_certify.yaml" \
               "NER Isotropic + Entity Masking (σ=0.50)" \
               "ner-mask-ent-iso"
    fi
    echo ""
fi

# ── Hybrid masking ────────────────────────────────
if [ "$DO_HYBRID" = true ]; then
    echo "── HYBRID MASKING ──"
    if [ "$MANIFOLD" = true ]; then
        submit "${CONFIGS_DIR}/ner_conll2003_bert_masking_hybrid_certify.yaml" \
               "NER Manifold + Hybrid Masking (σ=0.50)" \
               "ner-mask-hyb-mani"
    fi
    if [ "$ISOTROPIC" = true ]; then
        submit "${CONFIGS_DIR}/ner_conll2003_bert_isotropic_masking_hybrid_certify.yaml" \
               "NER Isotropic + Hybrid Masking (σ=0.50)" \
               "ner-mask-hyb-iso"
    fi
    echo ""
fi

echo "=============================================="
echo "Submitted: $TOTAL jobs"
if ! $DRY_RUN; then
    echo "Monitor: squeue -u \$USER"
fi
echo "=============================================="
