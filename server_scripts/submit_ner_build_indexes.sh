#!/usr/bin/env bash
# =============================================================================
# NER TOKEN INDEX BUILD — layers 0, 3, 6, 9 + last (with dedup)
# =============================================================================
# Builds deduplicated token embedding indexes for each layer.
# Run this ONCE before certification sweeps to pre-build all indexes.
#
# Total: 5 jobs (one per layer)
#
# Usage:
#   ./submit_ner_build_indexes.sh [--dry-run] [--no-dedup]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
CHECKPOINT="${PROJECT_ROOT}/output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt"
CONFIG="${PROJECT_ROOT}/src/configs/experiments/ner_conll2003_bert_certify.yaml"
OUT_BASE="${PROJECT_ROOT}/output/ner_conll2003_bert/token_layer_embeddings/train"

DRY_RUN=false
DEDUP_FLAG=""

for arg in "$@"; do
    case $arg in
        --dry-run)    DRY_RUN=true ;;
        --no-dedup)   DEDUP_FLAG="--no-dedup" ;;
    esac
done

if $DRY_RUN; then
    echo "=============================================="
    echo "DRY RUN MODE"
    echo "=============================================="
fi

TOTAL=0

submit() {
    local layer_index=$1  # integer or "none" for last
    local layer_tag=$2    # e.g. "layer_0" or "last"

    local layer_arg=""
    if [ "$layer_index" != "none" ]; then
        layer_arg="--layer-index $layer_index"
    fi

    local out_dir="${OUT_BASE}/${layer_tag}"
    local job_name="ner-idx-${layer_tag}"

    if $DRY_RUN; then
        echo "🔹 [DRY-RUN] Build index — ${layer_tag}"
        echo "   out: ${out_dir}"
        echo "   dedup: ${DEDUP_FLAG:-yes}"
    else
        echo "🚀 Build index — ${layer_tag}"
        sbatch <<EOF
#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 4:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
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

echo "Building NER token index: ${layer_tag}"
echo "Host: \$(hostname) | GPU: \$CUDA_VISIBLE_DEVICES | Start: \$(date)"

python -m src.experiments.indexing.ner_tokens \\
    --config   "$CONFIG" \\
    --checkpoint "$CHECKPOINT" \\
    --out-dir  "$out_dir" \\
    --split    train \\
    --backend  annoy \\
    --metric   euclidean \\
    --n-trees  50 \\
    ${layer_arg} \\
    ${DEDUP_FLAG}

echo "Done: \$(date)"
EOF
    fi
    TOTAL=$((TOTAL + 1))
}

LAYERS=(0 3 6 9)

echo ""
echo "=============================================="
echo "NER TOKEN INDEX BUILD (dedup=${DEDUP_FLAG:-on})"
echo "Layers: ${LAYERS[*]} + last"
echo "=============================================="
echo ""

for L in "${LAYERS[@]}"; do
    submit "$L" "layer_${L}"
done
submit "none" "last"

echo ""
echo "=============================================="
echo "Submitted: $TOTAL jobs"
if ! $DRY_RUN; then
    echo "Monitor: squeue -u \$USER"
    echo ""
    echo "Outputs:"
    for L in "${LAYERS[@]}"; do
        echo "  ${OUT_BASE}/layer_${L}/"
    done
    echo "  ${OUT_BASE}/last/"
fi
echo "=============================================="
