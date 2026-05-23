#!/usr/bin/env bash
# =============================================================================
# SUBMIT SIGMA SWEEP AS ONE SLURM ARRAY (mixed sigma lists per experiment)
# =============================================================================
# Runs all four experiment configs in one Slurm array, each with its own
# sigma_values list. The array queues unfinished tasks in file order.
# Total tasks = sum of per-config sigma_values lengths
# Concurrency is capped with Slurm array limit: --array=1-N%MAX_CONCURRENT
#
# Usage:
#   ./submit_sigma_quad_array.sh [--dry-run] [--max-concurrent 4]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Slurm defaults
PARTITION="gpu20"
TIME="48:00:00"
GPUS=1
CPUS=8
MEM_PER_CPU="8G"
MAX_CONCURRENT=4
DRY_RUN=false

# Fixed config set (four experiment configs)
NER_ISO_CFG="src/configs/experiments/ner_conll2003_bert_isotropic_certify.yaml"
NER_MANI_CFG="src/configs/experiments/ner_conll2003_bert_certify.yaml"
CELEBA_ISO_CFG="src/configs/experiments/certify_celeba_isotropic_pixel.yaml"
CELEBA_MANI_CFG="src/configs/experiments/certify_celeba_pixel.yaml"

print_help() {
    cat << 'EOF'
Usage: ./submit_sigma_quad_array.sh [OPTIONS]

Options:
  --partition PART        Slurm partition (default: gpu20)
  --time TIME             Time limit (default: 48:00:00)
  --cpus N                CPUs per task (default: 8)
  --mem-per-cpu MEM       Memory per CPU (default: 8G)
  --max-concurrent N      Array concurrency cap (default: 4)
  --dry-run               Print generated array command only
  --help                  Show this help

Behavior:
    - Reads sigma_values from each config independently.
    - Skips tasks whose `metrics.json` already exists.
    - Creates one Slurm array over all unfinished tasks.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --mem-per-cpu)
            MEM_PER_CPU="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
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
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

if ! [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [[ "$MAX_CONCURRENT" -lt 1 ]]; then
    echo "Error: --max-concurrent must be a positive integer"
    exit 1
fi

for cfg in "$NER_ISO_CFG" "$NER_MANI_CFG" "$CELEBA_ISO_CFG" "$CELEBA_MANI_CFG"; do
    if [[ ! -f "$cfg" ]]; then
        echo "Error: config not found: $cfg"
        exit 1
    fi
done

mkdir -p "$PROJECT_ROOT/output/slurm"
SWEEP_CONFIG_DIR="$PROJECT_ROOT/output/sweep_configs"
mkdir -p "$SWEEP_CONFIG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
TASK_FILE="$SWEEP_CONFIG_DIR/sigma_quad_tasks_${RUN_ID}.tsv"

python3 - <<PY
import yaml
from pathlib import Path

project_root = Path(r"$PROJECT_ROOT")
sweep_dir = Path(r"$SWEEP_CONFIG_DIR")
task_file = Path(r"$TASK_FILE")

entries = [
    ("ner_iso", Path(r"$NER_ISO_CFG"), "ner"),
    ("ner_mani", Path(r"$NER_MANI_CFG"), "ner"),
    ("celeb_iso", Path(r"$CELEBA_ISO_CFG"), "celeba"),
    ("celeb_mani", Path(r"$CELEBA_MANI_CFG"), "celeba"),
]

def sigma_list(cfg_path: Path):
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vals = cfg.get("smoothing", {}).get("sigma_values")
    if vals is None:
        sigma = cfg.get("smoothing", {}).get("sigma")
        vals = [sigma] if sigma is not None else []
    return [float(v) for v in vals]

def sigma_tag(sigma: float) -> str:
    return f"sigma_{sigma:.2f}".replace(".", "_")

def celeb_output_dir(cfg: dict, sigma: float) -> Path:
    dataset_name = str(cfg.get("dataset", {}).get("name", "celeba")).lower().replace("-", "").replace("_", "")
    base_dir = Path(cfg.get("output", {}).get("output_dir", "output")) / "smile_classification" / dataset_name
    mode = str(cfg.get("smoothing", {}).get("mode", "pixel")).strip().lower()
    return base_dir / "certify" / mode / sigma_tag(sigma)

def _layer_tag(layer_index):
    return "last" if layer_index is None else f"layer_{int(layer_index)}"

def ner_output_dir(cfg: dict, sigma: float, resolved_index_path: str | None = None) -> Path:
    smoothing = cfg.get("smoothing", {})
    cfg_root = Path(cfg.get("output_dir", "output"))
    smoothing_mode = str(smoothing.get("mode", "manifold")).strip().lower()
    backend = str(smoothing.get("index_backend", "torch")).strip().lower()
    metric = str(smoothing.get("index_metric", "euclidean")).strip().lower()
    layer = _layer_tag(smoothing.get("layer_index", None))
    index_name = Path(resolved_index_path).name if resolved_index_path else "in_memory_index"
    masking = cfg.get("masking", {})
    masking_enabled = bool(masking.get("enabled", False))
    masking_mode = str(masking.get("mode", "none")).strip().lower() if masking_enabled else "none"

    if smoothing_mode == "isotropic":
        if masking_enabled and masking_mode and masking_mode != "none":
            return cfg_root / "isotropic_masked_certify" / layer / masking_mode / sigma_tag(sigma)
        return cfg_root / "isotropic_certify" / layer / sigma_tag(sigma)

    if masking_enabled and masking_mode and masking_mode != "none":
        return cfg_root / "masked_certify" / layer / metric / backend / index_name / masking_mode / sigma_tag(sigma)
    return cfg_root / "certify" / layer / metric / backend / index_name / sigma_tag(sigma)

lines = []
skipped = []
config_sigma_counts = {}
for short_name, cfg_path, kind in entries:
    sigma_values = sigma_list(cfg_path)
    if not sigma_values:
        raise SystemExit(f"No sigma_values found in {cfg_path}")
    config_sigma_counts[short_name] = len(sigma_values)

    for sigma in sigma_values:
        sig_tag = sigma_tag(sigma)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        base_exp = cfg.get("experiment_name", short_name)
        job_exp_name = f"{base_exp}_sigma_{sig_tag}"
        cfg.setdefault("smoothing", {})["sigma"] = float(sigma)
        cfg["experiment_name"] = job_exp_name
        cfg.setdefault("checkpoint", {})["resume"] = True

        if kind == "ner":
            resolved_index_path = cfg.get("smoothing", {}).get("index_path")
            out_dir = ner_output_dir(cfg, sigma, resolved_index_path)
        else:
            out_dir = celeb_output_dir(cfg, sigma)

        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            skipped.append(f"{short_name} sigma={sigma} -> {metrics_path}")
            continue

        out_cfg = sweep_dir / f"{job_exp_name}_{short_name}_{sig_tag}.yaml"
        with out_cfg.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        lines.append(f"{job_exp_name}|{out_cfg}|{kind}|{sigma}")

if skipped:
    print("SKIPPED_EXISTING=")
    for item in skipped:
        print("  - " + item)

if not lines:
    print("All sigma jobs already have metrics.json; nothing to submit.")
    raise SystemExit(0)

with task_file.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

for name, count in config_sigma_counts.items():
    print(f"SIGMA_COUNT_{name.upper()}={count}")
print(f"TASK_COUNT={len(lines)}")
print("TASKS_PER_CONFIG=" + ", ".join(f"{name}:{count}" for name, count in config_sigma_counts.items()))
PY

TASK_COUNT=$(wc -l < "$TASK_FILE" | awk '{print $1}')
ARRAY_SPEC="1-${TASK_COUNT}%${MAX_CONCURRENT}"

echo "=============================================="
echo "SUBMITTING SIGMA QUAD ARRAY"
echo "=============================================="
echo "Task file:       $TASK_FILE"
echo "Total tasks:     $TASK_COUNT"
echo "Concurrency cap: $MAX_CONCURRENT"
echo "Array spec:      $ARRAY_SPEC"
echo "Partition:       $PARTITION"
echo "Time:            $TIME"
echo "=============================================="

WRAP_CMD="
set -euo pipefail
TASK_LINE=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" '$TASK_FILE')
IFS='|' read -r TASK_NAME TASK_CFG TASK_KIND TASK_SIGMA <<< \"\$TASK_LINE\"
echo \"[Task \${SLURM_ARRAY_TASK_ID}] \$TASK_NAME (kind=\$TASK_KIND, sigma=\$TASK_SIGMA)\"
cd '$PROJECT_ROOT'
export PYTHONPATH='$PROJECT_ROOT':\$PYTHONPATH
. /BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh
conda activate smoothing
if [ \"\$TASK_KIND\" = \"ner\" ]; then
  CHECKPOINT='$PROJECT_ROOT/output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt'
  python -m src.experiments.certify.ner --config \"\$TASK_CFG\" --checkpoint \"\$CHECKPOINT\" --split test --resume --save-every-batches 5
else
  python -m src.experiments.certify.celeba --config \"\$TASK_CFG\"
fi
"

SBATCH_CMD="sbatch \
  --partition=$PARTITION \
  --time=$TIME \
  --gres=gpu:$GPUS \
  --cpus-per-task=$CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --array=$ARRAY_SPEC \
  --job-name=sigma-quad \
  --output=$PROJECT_ROOT/output/slurm/sigma-quad-%A_%a.out \
  --error=$PROJECT_ROOT/output/slurm/sigma-quad-%A_%a.err \
  --wrap '$WRAP_CMD'"

if $DRY_RUN; then
    echo ""
    echo "[DRY-RUN] Command:"
    echo "$SBATCH_CMD"
else
    eval "$SBATCH_CMD"
    echo ""
    echo "Submitted array. Monitor with:"
    echo "  squeue -u \$USER"
fi
