#!/usr/bin/env bash
# =============================================================================
# SIGMA SWEEP - Submit one job per sigma value from config
# =============================================================================
# Reads sigma_values array from config and submits separate SLURM jobs.
# Each job runs one sigma → results saved to separate directories.
#
# Usage:
#   ./submit_sigma_sweep.sh CONFIG_FILE [OPTIONS]
#
# Examples:
#   ./submit_sigma_sweep.sh src/configs/experiments/certify_celeba_latent_128.yaml
#   ./submit_sigma_sweep.sh src/configs/experiments/ner_conll2003_bert_certify.yaml --dry-run
#
# Output Structure (per sigma):
#   output/smile_classification/celeba/certify/latent_manifold/sigma_0_25/metrics.json
#   output/smile_classification/celeba/certify/latent_manifold/sigma_0_50/metrics.json
#   output/smile_classification/celeba/certify/latent_manifold/sigma_0_75/metrics.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# SLURM defaults
PARTITION="gpu20"
TIME="48:00:00"
GPUS=1
CPUS=4
MEM="16G"

DRY_RUN=false
CONFIG_FILE=""

print_help() {
    cat << 'EOF'
SIGMA SWEEP - Submit one job per sigma value

Usage: ./submit_sigma_sweep.sh CONFIG_FILE [OPTIONS]

Arguments:
  CONFIG_FILE          Path to experiment config with sigma_values array

Options:
  --partition PART     SLURM partition (default: gpu20)
  --time TIME          Time limit (default: 12:00:00)
  --cpus N             CPUs per task (default: 4)
  --mem MEM            Memory (default: 16G)
  --dry-run            Print commands without submitting
  --help               Show this help

Examples:
  # CelebA latent manifold sweep
  ./submit_sigma_sweep.sh src/configs/experiments/certify_celeba_latent_128.yaml

  # NER sweep with custom time limit
  ./submit_sigma_sweep.sh src/configs/experiments/ner_conll2003_bert_certify.yaml --time 24:00:00

  # Dry run to preview jobs
  ./submit_sigma_sweep.sh src/configs/experiments/certify_celeba_pixel_128.yaml --dry-run
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
        *)
            CONFIG_FILE="$1"
            shift
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is required"
    print_help
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract sigma_values from config using Python
SIGMAS=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
sigmas = cfg.get('smoothing', {}).get('sigma_values', [cfg.get('smoothing', {}).get('sigma', 0.25)])
print(' '.join(str(s) for s in sigmas))
")

# Detect experiment type (celeba vs ner)
IS_NER=false
if grep -q "ner_conll" "$CONFIG_FILE" || grep -q "target: hidden_states" "$CONFIG_FILE"; then
    IS_NER=true
fi

# Extract experiment base name
EXP_NAME=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('experiment_name', 'experiment'))
")

echo "=============================================="
echo "SIGMA SWEEP SUBMISSION"
echo "=============================================="
echo "Config:     $CONFIG_FILE"
echo "Experiment: $EXP_NAME"
echo "Type:       $(if $IS_NER; then echo 'NER'; else echo 'CelebA/Image'; fi)"
echo "Sigmas:     $SIGMAS"
echo "Partition:  $PARTITION"
echo "Time:       $TIME"
echo "=============================================="
echo ""

# Submit one job per sigma
for SIGMA in $SIGMAS; do
    SIGMA_TAG=$(echo "$SIGMA" | tr '.' '_')
    JOB_NAME="${EXP_NAME}_sigma_${SIGMA_TAG}"
    
    # Create temp config with this sigma
    TEMP_CONFIG="/tmp/${JOB_NAME}.yaml"
    
    # Copy config and override sigma
    python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
cfg['smoothing']['sigma'] = $SIGMA
cfg['experiment_name'] = '$JOB_NAME'
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
    
    echo "----------------------------------------"
    echo "Sigma: $SIGMA"
    echo "Job:   $JOB_NAME"
    echo "Config: $TEMP_CONFIG"
    
    # Build SLURM command
    if $IS_NER; then
        # NER experiment
        CHECKPOINT="${PROJECT_ROOT}/output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt"
        RUN_CMD="python -m src.eval.run_ner_experiment --config $TEMP_CONFIG --checkpoint $CHECKPOINT --split test --resume --save-every-batches 5"
    else
        # CelebA/Image experiment
        RUN_CMD="python -m src.experiments.certify.celeba --config $TEMP_CONFIG"
    fi
    
    SBATCH_CMD="sbatch \
        --partition=$PARTITION \
        --time=$TIME \
        --gres=gpu:$GPUS \
        --cpus-per-task=$CPUS \
        --mem=$MEM \
        --job-name=$JOB_NAME \
        --output=$PROJECT_ROOT/output/slurm/${JOB_NAME}-%j.out \
        --error=$PROJECT_ROOT/output/slurm/${JOB_NAME}-%j.err \
        --wrap=\"cd $PROJECT_ROOT && source ~/miniforge3/etc/profile.d/conda.sh && conda activate smoothing && $RUN_CMD\""
    
    if $DRY_RUN; then
        echo "[DRY-RUN] Would run:"
        echo "  $SBATCH_CMD"
    else
        eval "$SBATCH_CMD"
        echo "✓ Submitted"
    fi
done

echo ""
echo "=============================================="
if $DRY_RUN; then
    echo "DRY RUN COMPLETE - No jobs submitted"
else
    echo "ALL SIGMA JOBS SUBMITTED"
    echo ""
    echo "Monitor:    squeue -u \$USER"
    echo ""
    echo "Results will be saved to separate directories:"
    if $IS_NER; then
        echo "  output/ner_conll2003_bert/certify/sigma_X_XX/"
    else
        echo "  output/smile_classification/{dataset}/certify/{mode}/sigma_X_XX/"
    fi
fi
echo "=============================================="
