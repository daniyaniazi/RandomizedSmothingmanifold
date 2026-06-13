#!/usr/bin/env bash
# =============================================================================
# SUBMIT SEGMENTATION SIGMA SWEEP — iso and manifold, multi-sigma per job
# =============================================================================
# Strategy (same as submit_sigma_quad_array.sh):
#   isotropic  → one Slurm ARRAY task per sigma  (parallel jobs)
#   manifold   → one Slurm job for ALL sigmas     (PCA computed once per sample)
#
# Usage:
#   ./submit_seg_sigma_quad.sh [--dry-run] [--strategy multi|array|auto]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
mkdir -p output/slurm

PARTITION="gpu20"
TIME="48:00:00"
GPUS=1
CPUS=8
MEM_PER_CPU="8G"
DRY_RUN=false
STRATEGY=""   # auto | multi | array

ISO_CFG="src/configs/experiments/certify_celebahq_seg_isotropic.yaml"
MANI_CFG="src/configs/experiments/certify_celebahq_seg_manifold.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true;    shift ;;
        --strategy)   STRATEGY="$2";   shift 2 ;;
        --partition)  PARTITION="$2";  shift 2 ;;
        --time)       TIME="$2";       shift 2 ;;
        *)
            echo "Usage: $0 [--dry-run] [--strategy multi|array|auto] [--partition PART] [--time TIME]"
            exit 1 ;;
    esac
done

# ── Read sigma_values from yaml ───────────────────────────────────────────────
read_sigmas() {
    python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$1'))
vals = cfg.get('smoothing', {}).get('sigma_values')
if not vals:
    vals = [cfg.get('smoothing', {}).get('sigma', 0.25)]
print(' '.join(str(v) for v in vals))
"
}

ISO_SIGMAS=$(read_sigmas "$ISO_CFG")
MANI_SIGMAS=$(read_sigmas "$MANI_CFG")

echo "=============================================="
echo "SEGMENTATION SIGMA SWEEP"
echo "  ISO  config: $ISO_CFG"
echo "  MANI config: $MANI_CFG"
echo "  ISO  sigmas: $ISO_SIGMAS"
echo "  MANI sigmas: $MANI_SIGMAS"
echo "  Strategy:    ${STRATEGY:-auto}"
echo "  Dry-run:     $DRY_RUN"
echo "=============================================="

# ── Helper: submit one job ────────────────────────────────────────────────────
submit_job() {
    local name="$1" cfg="$2" sigmas="$3"
    local job_script="$PROJECT_ROOT/output/slurm/${name}.sh"

    cat > "$job_script" << JOBEOF
#!/usr/bin/env bash
#SBATCH -p ${PARTITION}
#SBATCH -t ${TIME}
#SBATCH --gres gpu:${GPUS}
#SBATCH -c ${CPUS}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH -o ${PROJECT_ROOT}/output/slurm/${name}-%j.out
#SBATCH -e ${PROJECT_ROOT}/output/slurm/${name}-%j.err
#SBATCH -J ${name}

set -euo pipefail
cd "${PROJECT_ROOT}"
if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi
echo "Running on: \$(hostname)  GPU: \$CUDA_VISIBLE_DEVICES"
python -m src.experiments.certify.celebahq_segmentation \\
    --config ${cfg} \\
    --sigmas ${sigmas}
JOBEOF
    chmod +x "$job_script"

    if $DRY_RUN; then
        echo "  [DRY-RUN] $name → $job_script"
    else
        local jid
        jid=$(sbatch --parsable "$job_script")
        echo "  Submitted $name → job $jid"
    fi
}

# ── Helper: submit array (one task per sigma) ─────────────────────────────────
submit_array() {
    local name="$1" cfg="$2"
    shift 2
    local sigmas=("$@")
    local n="${#sigmas[@]}"
    local task_file="$PROJECT_ROOT/output/slurm/${name}_tasks.txt"

    # Write one sigma per line
    printf '%s\n' "${sigmas[@]}" > "$task_file"

    local runner="$PROJECT_ROOT/output/slurm/${name}_runner.sh"
    cat > "$runner" << JOBEOF
#!/usr/bin/env bash
#SBATCH -p ${PARTITION}
#SBATCH -t ${TIME}
#SBATCH --gres gpu:${GPUS}
#SBATCH -c ${CPUS}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH -o ${PROJECT_ROOT}/output/slurm/${name}-%A_%a.out
#SBATCH -e ${PROJECT_ROOT}/output/slurm/${name}-%A_%a.err
#SBATCH -J ${name}
#SBATCH --array=1-${n}

set -euo pipefail
cd "${PROJECT_ROOT}"
if [[ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]]; then
    source "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate smoothing
fi
SIGMA=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${task_file}")
echo "Task \${SLURM_ARRAY_TASK_ID}: sigma=\${SIGMA}  host=\$(hostname)  GPU=\${CUDA_VISIBLE_DEVICES}"
python -m src.experiments.certify.celebahq_segmentation \\
    --config ${cfg} \\
    --sigmas \${SIGMA}
JOBEOF
    chmod +x "$runner"

    if $DRY_RUN; then
        echo "  [DRY-RUN] $name array (${n} tasks) → $runner"
    else
        local jid
        jid=$(sbatch --parsable "$runner")
        echo "  Submitted $name array (${n} tasks) → job $jid"
    fi
}

# ── Determine strategy and submit ────────────────────────────────────────────
# Isotropic
if [[ "$STRATEGY" == "multi" ]]; then
    submit_job "seg-iso-multi"  "$ISO_CFG"  "$ISO_SIGMAS"
elif [[ "$STRATEGY" == "array" || "$STRATEGY" == "" || "$STRATEGY" == "auto" ]]; then
    # Default for iso: array (parallel per sigma)
    read -ra ISO_ARR <<< "$ISO_SIGMAS"
    submit_array "seg-iso" "$ISO_CFG" "${ISO_ARR[@]}"
fi

# Manifold
if [[ "$STRATEGY" == "array" ]]; then
    read -ra MANI_ARR <<< "$MANI_SIGMAS"
    submit_array "seg-mani" "$MANI_CFG" "${MANI_ARR[@]}"
else
    # Default for manifold: multi (PCA once)
    submit_job "seg-mani-multi" "$MANI_CFG" "$MANI_SIGMAS"
fi

echo "=============================================="
echo "Monitor: squeue -u \$USER"
echo "Outputs: output/segmentation/celebahq/certify/"
echo "=============================================="
