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
#   ./submit_sigma_quad_array.sh [--dry-run] [--max-concurrent 4] [--celeba-only|--ner-only]
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
DATASET_FILTER=""   # celeba | celebahq | ner | "" (all)
SPACE_FILTER=""     # pixel | latent | "" (all)

# Fixed config set (four experiment configs)
NER_ISO_CFG="src/configs/experiments/ner_conll2003_bert_isotropic_certify.yaml"
NER_MANI_CFG="src/configs/experiments/ner_conll2003_bert_certify.yaml"
CELEBA_ISO_CFG="src/configs/experiments/certify_celeba_isotropic_pixel.yaml"
CELEBA_MANI_CFG="src/configs/experiments/certify_celeba_pixel.yaml"
CELEBA_ISO_LATENT_CFG="src/configs/experiments/certify_celeba_isotropic_latent_128.yaml"
CELEBA_MANI_LATENT_CFG="src/configs/experiments/certify_celeba_latent_128.yaml"
CELEBAHQ_ISO_CFG="src/configs/experiments/certify_celebahq_isotropic_pixel.yaml"
CELEBAHQ_MANI_CFG="src/configs/experiments/certify_celebahq_pixel.yaml"

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
  --dataset DATASET       Filter by dataset: celeba | celebahq | ner  (default: all)
  --space SPACE           Filter by space:   pixel | latent           (default: all)
  --help                  Show this help

Examples:
  --dataset celeba  --space pixel       CelebA pixel ISO+manifold
  --dataset celeba  --space latent      CelebA latent ISO+manifold
  --dataset celebahq --space pixel      CelebA-HQ pixel ISO+manifold
  --dataset ner                         NER ISO+manifold
  (no flags)                            All configs

Behavior:
    - Reads sigma_values from each config independently.
    - Skips tasks whose metrics.json already exists.
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
        --dataset)
            DATASET_FILTER="$2"
            shift 2
            ;;
        --space)
            SPACE_FILTER="$2"
            shift 2
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

if [[ -n "$DATASET_FILTER" ]] && ! [[ "$DATASET_FILTER" =~ ^(celeba|celebahq|ner)$ ]]; then
    echo "Error: --dataset must be celeba, celebahq, or ner"
    exit 1
fi
if [[ -n "$SPACE_FILTER" ]] && ! [[ "$SPACE_FILTER" =~ ^(pixel|latent)$ ]]; then
    echo "Error: --space must be pixel or latent"
    exit 1
fi
if [[ "$DATASET_FILTER" == "ner" && "$SPACE_FILTER" == "latent" ]]; then
    echo "Error: NER does not have a latent config"
    exit 1
fi
if [[ "$DATASET_FILTER" == "celebahq" && "$SPACE_FILTER" == "latent" ]]; then
    echo "Error: CelebA-HQ does not have a latent config"
    exit 1
fi

# Build CONFIGS_TO_CHECK from all known configs, then filter by dataset/space
ALL_CONFIGS=(
    "$NER_ISO_CFG"          "ner"       "pixel"
    "$NER_MANI_CFG"         "ner"       "pixel"
    "$CELEBA_ISO_CFG"       "celeba"    "pixel"
    "$CELEBA_MANI_CFG"      "celeba"    "pixel"
    "$CELEBA_ISO_LATENT_CFG" "celeba"   "latent"
    "$CELEBA_MANI_LATENT_CFG" "celeba"  "latent"
    "$CELEBAHQ_ISO_CFG"     "celebahq"  "pixel"
    "$CELEBAHQ_MANI_CFG"    "celebahq"  "pixel"
)

CONFIGS_TO_CHECK=()
i=0
while [[ $i -lt ${#ALL_CONFIGS[@]} ]]; do
    cfg="${ALL_CONFIGS[$i]}"
    ds="${ALL_CONFIGS[$((i+1))]}"
    sp="${ALL_CONFIGS[$((i+2))]}"
    if { [[ -z "$DATASET_FILTER" ]] || [[ "$ds" == "$DATASET_FILTER" ]]; } && \
       { [[ -z "$SPACE_FILTER" ]]  || [[ "$sp" == "$SPACE_FILTER" ]]; }; then
        CONFIGS_TO_CHECK+=("$cfg")
    fi
    i=$((i+3))
done

if [[ ${#CONFIGS_TO_CHECK[@]} -eq 0 ]]; then
    echo "Error: no configs match --dataset='$DATASET_FILTER' --space='$SPACE_FILTER'"
    exit 1
fi

for cfg in "${CONFIGS_TO_CHECK[@]}"; do
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

DATASET_FILTER="$DATASET_FILTER" SPACE_FILTER="$SPACE_FILTER" python3 - <<PY
import os
import yaml
from pathlib import Path

project_root = Path(r"$PROJECT_ROOT")
sweep_dir = Path(r"$SWEEP_CONFIG_DIR")
task_file = Path(r"$TASK_FILE")

dataset_filter = os.environ.get("DATASET_FILTER", "").strip().lower()
space_filter   = os.environ.get("SPACE_FILTER",   "").strip().lower()

entries = [
    ("ner_iso",           Path(r"$NER_ISO_CFG"),            "ner",     "pixel"),
    ("ner_mani",          Path(r"$NER_MANI_CFG"),           "ner",     "pixel"),
    ("celeb_iso",         Path(r"$CELEBA_ISO_CFG"),         "celeba",  "pixel"),
    ("celeb_mani",        Path(r"$CELEBA_MANI_CFG"),        "celeba",  "pixel"),
    ("celeb_iso_latent",  Path(r"$CELEBA_ISO_LATENT_CFG"),  "celeba",  "latent"),
    ("celeb_mani_latent", Path(r"$CELEBA_MANI_LATENT_CFG"), "celeba",  "latent"),
    ("celebahq_iso",      Path(r"$CELEBAHQ_ISO_CFG"),       "celebahq","pixel"),
    ("celebahq_mani",     Path(r"$CELEBAHQ_MANI_CFG"),      "celebahq","pixel"),
]

# Filter by --dataset and --space
if dataset_filter:
    entries = [e for e in entries if e[2] == dataset_filter]
if space_filter:
    entries = [e for e in entries if e[3] == space_filter]

if not entries:
    raise SystemExit("No configs selected for submission.")

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
    use_manifold = cfg.get("smoothing", {}).get("use_manifold", False)
    mode_tag = f"{mode}_{'manifold' if use_manifold else 'isotropic'}"
    ood_attr = cfg.get("dataset", {}).get("ood_attribute", None)
    if ood_attr:
        return base_dir / "certify_ood" / ood_attr.lower() / mode_tag / sigma_tag(sigma)
    return base_dir / "certify" / mode_tag / sigma_tag(sigma)

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

# Build per-config candidate task lists first, then interleave by sigma index
# so the array order is: ner_iso[0], ner_mani[0], celeb_iso[0], celeb_mani[0],
#                        ner_iso[1], ner_mani[1], celeb_iso[1], celeb_mani[1], ...
per_config_tasks = {}  # short_name -> list of formatted task strings
for short_name, cfg_path, ds, sp in entries:
    sigma_values = sigma_list(cfg_path)
    if not sigma_values:
        raise SystemExit(f"No sigma_values found in {cfg_path}")
    config_sigma_counts[short_name] = len(sigma_values)
    per_config_tasks[short_name] = []

    for sigma in sigma_values:
        sig_tag = sigma_tag(sigma)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        base_exp = cfg.get("experiment_name", short_name)
        job_exp_name = f"{base_exp}_sigma_{sig_tag}"
        cfg.setdefault("smoothing", {})["sigma"] = float(sigma)
        cfg["experiment_name"] = job_exp_name
        cfg.setdefault("checkpoint", {})["resume"] = True

        if ds == "ner":
            resolved_index_path = cfg.get("smoothing", {}).get("index_path")
            out_dir = ner_output_dir(cfg, sigma, resolved_index_path)
        else:  # celeba or celebahq
            out_dir = celeb_output_dir(cfg, sigma)

        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            skipped.append(f"{short_name} sigma={sigma} -> {metrics_path}")
            continue

        out_cfg = sweep_dir / f"{job_exp_name}_{short_name}_{sig_tag}.yaml"
        with out_cfg.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        per_config_tasks[short_name].append(f"{job_exp_name}|{out_cfg}|{ds}_{sp}|{sigma}")

# Interleave: for each sigma index, emit one task per config (if it exists)
max_len = max(len(v) for v in per_config_tasks.values())
for i in range(max_len):
    for short_name, _, _ds, _sp in entries:
        task_list = per_config_tasks[short_name]
        if i < len(task_list):
            lines.append(task_list[i])

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

JOB_SCRIPT="$PROJECT_ROOT/output/slurm/sigma_quad_runner_${RUN_ID}.sh"
cat > "$JOB_SCRIPT" << 'JOBEOF'
#!/usr/bin/env bash
set -euo pipefail
TASK_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "TASK_FILE_PLACEHOLDER")
IFS='|' read -r TASK_NAME TASK_CFG TASK_KIND TASK_SIGMA <<< "$TASK_LINE"
echo "[Task ${SLURM_ARRAY_TASK_ID}] $TASK_NAME (kind=$TASK_KIND, sigma=$TASK_SIGMA)"
cd PROJECT_ROOT_PLACEHOLDER
export PYTHONPATH=PROJECT_ROOT_PLACEHOLDER:${PYTHONPATH:-}
. /BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh
conda activate smoothing
if [[ "$TASK_KIND" == ner_* ]]; then
  CHECKPOINT=PROJECT_ROOT_PLACEHOLDER/output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt
  python -m src.experiments.certify.ner --config "$TASK_CFG" --checkpoint "$CHECKPOINT" --split test --resume --save-every-batches 5
else
  python -m src.experiments.certify.celeba --config "$TASK_CFG"
fi
JOBEOF

# Substitute placeholders with real paths
sed -i "s|TASK_FILE_PLACEHOLDER|${TASK_FILE}|g" "$JOB_SCRIPT"
sed -i "s|PROJECT_ROOT_PLACEHOLDER|${PROJECT_ROOT}|g" "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

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
  $JOB_SCRIPT"

if $DRY_RUN; then
    echo ""
    echo "[DRY-RUN] Job script: $JOB_SCRIPT"
    echo "[DRY-RUN] Command:"
    echo "$SBATCH_CMD"
else
    eval "$SBATCH_CMD"
    echo ""
    echo "Submitted array. Monitor with:"
    echo "  squeue -u \$USER"
fi
