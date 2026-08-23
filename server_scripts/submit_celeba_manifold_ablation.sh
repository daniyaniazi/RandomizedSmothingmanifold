#!/usr/bin/env bash
# Submit CelebA pixel-manifold ablations. Each array task handles one K/d
# variant and all configured sigmas, allowing PCA reuse across sigma values.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PARTITION="gpu20"
TIME="48:00:00"
CPUS=8
MEM_PER_CPU="8G"
MAX_CONCURRENT=4
STUDY="all"
DRY_RUN=false

LOCAL_CFG="src/configs/experiments/certify_celeba_pixel_ablation_local_size.yaml"
PCA_CFG="src/configs/experiments/certify_celeba_pixel_ablation_pca_dim.yaml"
PYTHON_BIN="/BS/dniazi_thesis/work/miniforge3_new/envs/smoothing/bin/python"

usage() {
    echo "Usage: $0 [--study local-size|pca-dim|all] [--dry-run]"
    echo "          [--partition PART] [--time TIME] [--cpus N]"
    echo "          [--mem-per-cpu MEM] [--max-concurrent N]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --study) STUDY="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --mem-per-cpu) MEM_PER_CPU="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if ! [[ "$STUDY" =~ ^(local-size|pca-dim|all)$ ]]; then
    echo "Error: --study must be local-size, pca-dim, or all"
    exit 1
fi
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --max-concurrent must be a positive integer"
    exit 1
fi

mkdir -p output/slurm output/sweep_configs
RUN_ID="$(date +%Y%m%d_%H%M%S)"
TASK_FILE="output/sweep_configs/celeba_manifold_ablation_${RUN_ID}.tsv"

STUDY="$STUDY" LOCAL_CFG="$LOCAL_CFG" PCA_CFG="$PCA_CFG" \
TASK_FILE="$TASK_FILE" RUN_ID="$RUN_ID" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import yaml

study = os.environ["STUDY"]
task_file = Path(os.environ["TASK_FILE"])
run_id = os.environ["RUN_ID"]
out_dir = task_file.parent
specs = []
if study in ("local-size", "all"):
    specs.append(("local_size", Path(os.environ["LOCAL_CFG"]), "knn_values"))
if study in ("pca-dim", "all"):
    specs.append(("pca_dim", Path(os.environ["PCA_CFG"]), "pca_dim_values"))

tasks = []
for short_name, source, values_key in specs:
    cfg = yaml.safe_load(source.read_text())
    smoothing = cfg["smoothing"]
    values = smoothing.pop(values_key)
    sigmas = [float(v) for v in smoothing.get("sigma_values", [smoothing["sigma"]])]
    sigma_args = " ".join(str(v) for v in sigmas)

    for value in values:
        generated = yaml.safe_load(source.read_text())
        generated_smoothing = generated["smoothing"]
        generated_smoothing.pop(values_key, None)
        if short_name == "local_size":
            generated_smoothing["knn_k"] = int(value)
            generated_smoothing["pca_dim"] = None
            variant = f"knn_{int(value)}_pca_auto"
        else:
            generated_smoothing["knn_k"] = 500
            generated_smoothing["pca_dim"] = int(value)
            variant = f"knn_500_pca_{int(value)}"

        generated["output"]["ablation_variant"] = variant
        generated["experiment_name"] = f"{generated['experiment_name']}_{variant}"
        generated_path = out_dir / f"{short_name}_{variant}_{run_id}.yaml"
        generated_path.write_text(yaml.safe_dump(generated, sort_keys=False))
        tasks.append(f"{short_name}_{variant}|{generated_path}|{sigma_args}")

task_file.write_text("\n".join(tasks) + "\n")
print(f"Created {len(tasks)} ablation tasks in {task_file}")
PY

TASK_COUNT="$(wc -l < "$TASK_FILE" | awk '{print $1}')"
if [[ "$TASK_COUNT" -eq 0 ]]; then
    echo "No ablation tasks generated."
    exit 0
fi

JOB_SCRIPT="output/slurm/celeba_manifold_ablation_${RUN_ID}.sh"
sed_task_file="$(realpath "$TASK_FILE")"
sed_project_root="$(realpath "$PROJECT_ROOT")"
apply_runner_template() {
    sed -e "s|__TASK_FILE__|$sed_task_file|g" \
        -e "s|__PROJECT_ROOT__|$sed_project_root|g" "$1" > "$2"
}
RUNNER_TEMPLATE="$(mktemp /tmp/celeba_ablation_runner.XXXXXX)"
trap 'rm -f "$RUNNER_TEMPLATE"' EXIT
cat > "$RUNNER_TEMPLATE" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail
TASK_LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "__TASK_FILE__")"
IFS='|' read -r TASK_NAME TASK_CFG TASK_SIGMAS <<< "$TASK_LINE"
echo "Task: $TASK_NAME"
echo "Config: $TASK_CFG"
echo "Sigmas: $TASK_SIGMAS"
cd "__PROJECT_ROOT__"
export PYTHONPATH="__PROJECT_ROOT__:${PYTHONPATH:-}"
. /BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh
conda activate smoothing
python -m src.experiments.certify.celeba --config "$TASK_CFG" --sigmas $TASK_SIGMAS
RUNNER
apply_runner_template "$RUNNER_TEMPLATE" "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

ARRAY_SPEC="1-${TASK_COUNT}%${MAX_CONCURRENT}"
SBATCH_ARGS=(
    --partition="$PARTITION"
    --time="$TIME"
    --gres=gpu:1
    --cpus-per-task="$CPUS"
    --mem-per-cpu="$MEM_PER_CPU"
    --array="$ARRAY_SPEC"
    --job-name=celeba-mani-ablation
    --output="$PROJECT_ROOT/output/slurm/celeba-ablation-%A_%a.out"
    --error="$PROJECT_ROOT/output/slurm/celeba-ablation-%A_%a.err"
)

echo "Study: $STUDY"
echo "Tasks: $TASK_COUNT"
echo "Array: $ARRAY_SPEC"
echo "Task file: $TASK_FILE"
if "$DRY_RUN"; then
    echo "DRY RUN: sbatch ${SBATCH_ARGS[*]} $JOB_SCRIPT"
else
    sbatch "${SBATCH_ARGS[@]}" "$JOB_SCRIPT"
fi
