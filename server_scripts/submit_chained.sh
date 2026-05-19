#!/usr/bin/env bash
# =============================================================================
# CHAINED SUBMISSION: Isotropic → Manifold (with SLURM dependency)
# =============================================================================
# Submits isotropic jobs first, then manifold jobs with --dependency=afterok
# so manifold only starts after ALL isotropic jobs finish successfully.
#
# This ensures manifold runs can load companion iso_radii for Qty 1,2,3.
#
# Usage:
#   ./submit_chained.sh ner-sigma   [--dry-run]   # Chain 1: NER sigma sweep (last layer)
#   ./submit_chained.sh ner-layer   [--dry-run]   # Chain 2: NER layer sweep (L=0,3,6,9)
#   ./submit_chained.sh celeba      [--dry-run]   # Chain 3: CelebA pixel+latent
#   ./submit_chained.sh celebahq    [--dry-run]   # Chain 4: CelebaHQ pixel+latent
#   ./submit_chained.sh all         [--dry-run]   # All 4 chains (independent/parallel)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ROOT="/BS/dniazi_thesis/work/RandomizedSmothingmanifold"
CONFIGS_DIR="${PROJECT_ROOT}/src/configs/experiments"
CHECKPOINT="${PROJECT_ROOT}/output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt"

# ── Defaults ──
PARTITION="gpu20"
TIME="48:00:00"
GPUS=1
CPUS=8
MEM_PER_CPU="8G"
DRY_RUN=false
TARGET=""

for arg in "$@"; do
    case $arg in
        --dry-run)  DRY_RUN=true ;;
        ner-sigma|ner-layer|celeba|celebahq|all) TARGET=$arg ;;
    esac
done

if [ -z "$TARGET" ]; then
    echo "Usage: ./submit_chained.sh {ner-sigma|ner-layer|celeba|celebahq|all} [--dry-run]"
    exit 1
fi

SIGMAS=(0.25 0.50 0.75 1.00)
SWEEP_CONFIG_DIR="${PROJECT_ROOT}/output/sweep_configs"
mkdir -p "$SWEEP_CONFIG_DIR" "${PROJECT_ROOT}/output/slurm"

# ── Helper: submit one job, echo its SLURM job ID ──
submit_one() {
    local config=$1
    local job_name=$2
    local dep=$3  # empty string or "--dependency=afterok:id1:id2:..."
    local is_ner=$4

    if $is_ner; then
        local run_cmd="python -m src.experiments.certify.ner --config $config --checkpoint $CHECKPOINT --split test --resume --save-every-batches 5"
    else
        local run_cmd="python -m src.experiments.certify.celeba --config $config"
    fi

    local sbatch_args="--partition=$PARTITION --time=$TIME --gres=gpu:$GPUS \
        --cpus-per-task=$CPUS --mem-per-cpu=$MEM_PER_CPU \
        --job-name=$job_name \
        --output=$PROJECT_ROOT/output/slurm/${job_name}-%j.out \
        --error=$PROJECT_ROOT/output/slurm/${job_name}-%j.err"

    if [ -n "$dep" ]; then
        sbatch_args="$sbatch_args $dep"
    fi

    local wrap="cd $PROJECT_ROOT && export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
. /BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh && \
conda activate smoothing && $run_cmd"

    if $DRY_RUN; then
        echo "DRY-RUN: $job_name ${dep:+(dep: $dep)}"
        echo "DUMMY_$(echo $job_name | md5sum | head -c 6)"  # fake ID for dry-run chaining
    else
        local output
        output=$(sbatch $sbatch_args --wrap="$wrap" 2>&1)
        local job_id=$(echo "$output" | grep -oP '\d+$')
        echo "$job_id"
    fi
}

# ── Helper: create per-sigma config ──
make_sigma_config() {
    local base_config=$1
    local sigma=$2
    local exp_name=$3
    local sigma_tag=$(echo "$sigma" | tr '.' '_')
    local out="${SWEEP_CONFIG_DIR}/${exp_name}_sigma_${sigma_tag}.yaml"

    python3 -c "
import yaml
with open('$base_config') as f:
    cfg = yaml.safe_load(f)
cfg['smoothing']['sigma'] = $sigma
cfg['experiment_name'] = '${exp_name}_sigma_${sigma_tag}'
if 'checkpoint' not in cfg:
    cfg['checkpoint'] = {}
cfg['checkpoint']['resume'] = True
with open('$out', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
    echo "$out"
}

# ── Submit a sigma sweep: iso first, then manifold with dependency ──
submit_sigma_sweep_chained() {
    local iso_config=$1
    local mani_config=$2
    local iso_prefix=$3
    local mani_prefix=$4
    local is_ner=$5

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 CHAINED: $iso_prefix → $mani_prefix"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Stage 1: Isotropic
    local iso_job_ids=()
    for sigma in "${SIGMAS[@]}"; do
        local sigma_tag=$(echo "$sigma" | tr '.' '_')
        local cfg=$(make_sigma_config "$iso_config" "$sigma" "$iso_prefix")
        local job_name="${iso_prefix}_s${sigma_tag}"
        local jid=$(submit_one "$cfg" "$job_name" "" "$is_ner")
        iso_job_ids+=("$jid")
        echo "  ✅ ISO  σ=$sigma  →  job $jid"
    done

    # Build dependency string
    local dep_str=$(IFS=:; echo "${iso_job_ids[*]}")
    local dep_flag="--dependency=afterok:${dep_str}"

    # Stage 2: Manifold (depends on ALL isotropic)
    for sigma in "${SIGMAS[@]}"; do
        local sigma_tag=$(echo "$sigma" | tr '.' '_')
        local cfg=$(make_sigma_config "$mani_config" "$sigma" "$mani_prefix")
        local job_name="${mani_prefix}_s${sigma_tag}"
        local jid=$(submit_one "$cfg" "$job_name" "$dep_flag" "$is_ner")
        echo "  ⏳ MANI σ=$sigma  →  job $jid  (after iso)"
    done
}

# ═════════════════════════════════════════════════════════════════════════════
# CHAIN 1: NER SIGMA SWEEP (last layer, 4 sigmas)
# ═════════════════════════════════════════════════════════════════════════════
submit_ner_sigma() {
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║   CHAIN 1: NER SIGMA SWEEP (last layer) ║"
    echo "╚══════════════════════════════════════════╝"

    submit_sigma_sweep_chained \
        "${CONFIGS_DIR}/ner_conll2003_bert_isotropic_certify.yaml" \
        "${CONFIGS_DIR}/ner_conll2003_bert_certify.yaml" \
        "ner-iso-last" \
        "ner-mani-last" \
        true
}

# ═════════════════════════════════════════════════════════════════════════════
# CHAIN 2: NER LAYER SWEEP (L=0,3,6,9 at σ=0.50)
# ═════════════════════════════════════════════════════════════════════════════
submit_ner_layer() {
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║   CHAIN 2: NER LAYER SWEEP (σ=0.50)     ║"
    echo "╚══════════════════════════════════════════╝"

    local layer_iso_ids=()
    for L in 0 3 6 9; do
        local cfg="${CONFIGS_DIR}/ner_conll2003_bert_isotropic_certify_layer_${L}.yaml"
        local job_name="ner-iso-L${L}"
        local jid=$(submit_one "$cfg" "$job_name" "" true)
        layer_iso_ids+=("$jid")
        echo "  ✅ ISO  Layer $L  →  job $jid"
    done

    local dep_str=$(IFS=:; echo "${layer_iso_ids[*]}")
    local dep_flag="--dependency=afterok:${dep_str}"

    for L in 0 3 6 9; do
        local cfg="${CONFIGS_DIR}/ner_conll2003_bert_certify_layer_${L}.yaml"
        local job_name="ner-mani-L${L}"
        local jid=$(submit_one "$cfg" "$job_name" "$dep_flag" true)
        echo "  ⏳ MANI Layer $L  →  job $jid  (after iso)"
    done
}

# ═════════════════════════════════════════════════════════════════════════════
# CHAIN 3: CELEBA (pixel + latent)
# ═════════════════════════════════════════════════════════════════════════════
submit_celeba() {
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║   CHAIN 3: CELEBA (pixel + latent)       ║"
    echo "╚══════════════════════════════════════════╝"

    # Pixel
    submit_sigma_sweep_chained \
        "${CONFIGS_DIR}/certify_celeba_isotropic_pixel.yaml" \
        "${CONFIGS_DIR}/certify_celeba_pixel.yaml" \
        "celeba-pix-iso" \
        "celeba-pix-mani" \
        false

    # Latent
    submit_sigma_sweep_chained \
        "${CONFIGS_DIR}/certify_celeba_isotropic_latent_128.yaml" \
        "${CONFIGS_DIR}/certify_celeba_latent_128.yaml" \
        "celeba-lat-iso" \
        "celeba-lat-mani" \
        false
}

# ═════════════════════════════════════════════════════════════════════════════
# CHAIN 4: CELEBAHQ (pixel + latent)
# ═════════════════════════════════════════════════════════════════════════════
submit_celebahq() {
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║   CHAIN 4: CELEBAHQ (pixel + latent)     ║"
    echo "╚══════════════════════════════════════════╝"

    # Pixel
    submit_sigma_sweep_chained \
        "${CONFIGS_DIR}/certify_celebahq_isotropic_pixel.yaml" \
        "${CONFIGS_DIR}/certify_celebahq_pixel.yaml" \
        "celebahq-pix-iso" \
        "celebahq-pix-mani" \
        false

    # Latent
    submit_sigma_sweep_chained \
        "${CONFIGS_DIR}/certify_celebahq_isotropic_latent.yaml" \
        "${CONFIGS_DIR}/certify_celebahq_latent.yaml" \
        "celebahq-lat-iso" \
        "celebahq-lat-mani" \
        false
}

# ═════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═════════════════════════════════════════════════════════════════════════════

if $DRY_RUN; then
    echo "=============================================="
    echo "DRY RUN MODE - No jobs will be submitted"
    echo "=============================================="
fi

case $TARGET in
    ner-sigma) submit_ner_sigma ;;
    ner-layer) submit_ner_layer ;;
    celeba)    submit_celeba ;;
    celebahq)  submit_celebahq ;;
    all)       submit_ner_sigma; submit_ner_layer; submit_celeba; submit_celebahq ;;
esac

echo ""
echo "=============================================="
echo "CHAINED SUBMISSION COMPLETE"
echo "=============================================="
echo "Isotropic jobs run first."
echo "Manifold jobs queued with --dependency=afterok."
echo "Monitor: squeue -u \$USER"
echo "=============================================="
