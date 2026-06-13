#!/usr/bin/env bash
# =============================================================================
# SUBMIT SEGMENTATION SIGMA SWEEP — mirrors submit_sigma_quad_array.sh
# =============================================================================
#
# Strategy (auto-detected per config, same logic as CelebA quad script):
#   isotropic  → "array"  — one task per sigma, all run in parallel
#   manifold   → "multi"  — one task, all sigmas passed as --sigmas (PCA once)
#
# Tasks are interleaved: iso_σ0, mani_σ0, iso_σ1, mani_σ1 ...
# so both iso and mani tasks run side-by-side within the concurrency cap.
#
# Usage:
#   ./submit_seg_sigma_quad.sh [--dry-run] [--max-concurrent N]
#                              [--strategy multi|array|auto]
#                              [--partition PART] [--time TIME]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PARTITION="gpu20"
TIME="48:00:00"
GPUS=1
CPUS=8
MEM_PER_CPU="8G"
MAX_CONCURRENT=20
DRY_RUN=false
VIZ_ONLY=false
STRATEGY=""   # multi | array | "" (auto)

ISO_CFG="src/configs/experiments/certify_celebahq_seg_isotropic.yaml"
MANI_CFG="src/configs/experiments/certify_celebahq_seg_manifold.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=true;          shift ;;
        --viz-only)        VIZ_ONLY=true;         shift ;;
        --max-concurrent)  MAX_CONCURRENT="$2";    shift 2 ;;
        --strategy)        STRATEGY="$2";           shift 2 ;;
        --partition)       PARTITION="$2";          shift 2 ;;
        --time)            TIME="$2";               shift 2 ;;
        *)
            echo "Usage: $0 [--dry-run] [--viz-only] [--max-concurrent N] [--strategy multi|array|auto]"
            exit 1 ;;
    esac
done

if ! [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [[ "$MAX_CONCURRENT" -lt 1 ]]; then
    echo "Error: --max-concurrent must be a positive integer"; exit 1
fi
if [[ -n "$STRATEGY" ]] && ! [[ "$STRATEGY" =~ ^(multi|array|auto)$ ]]; then
    echo "Error: --strategy must be multi, array, or auto"; exit 1
fi

mkdir -p "$PROJECT_ROOT/output/slurm"
SWEEP_CONFIG_DIR="$PROJECT_ROOT/output/sweep_configs"
mkdir -p "$SWEEP_CONFIG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
TASK_FILE="$SWEEP_CONFIG_DIR/seg_sigma_tasks_${RUN_ID}.tsv"

# ── Build task list (Python) ──────────────────────────────────────────────────
STRATEGY="$STRATEGY" python3 - <<PY
import os, yaml
from pathlib import Path

project_root  = Path(r"$PROJECT_ROOT")
sweep_dir     = Path(r"$SWEEP_CONFIG_DIR")
task_file     = Path(r"$TASK_FILE")
strategy_override = os.environ.get("STRATEGY", "").strip().lower()

entries = [
    ("seg_iso",  Path(r"$ISO_CFG"),  "iso"),
    ("seg_mani", Path(r"$MANI_CFG"), "mani"),
]

def sigma_list(cfg_path):
    cfg = yaml.safe_load(cfg_path.read_text())
    vals = cfg.get("smoothing", {}).get("sigma_values")
    if not vals:
        s = cfg.get("smoothing", {}).get("sigma")
        vals = [s] if s is not None else []
    return [float(v) for v in vals]

def sigma_tag(s):
    return f"sigma_{s:.2f}".replace(".", "_")

def seg_output_dir(cfg_raw, sigma):
    dataset_tag = str(cfg_raw.get("dataset", {}).get("name", "celebahq")).lower().replace("-","").replace("_","")
    base_dir    = Path(cfg_raw.get("output", {}).get("output_dir", "output")) / "segmentation" / dataset_tag
    use_manifold = cfg_raw.get("smoothing", {}).get("use_manifold", False)
    mode_tag    = f"pixel_{'manifold' if use_manifold else 'isotropic'}"
    return base_dir / "certify" / mode_tag / sigma_tag(sigma)

iso_tasks  = []
mani_tasks = []
skipped    = []

for short_name, cfg_path, kind in entries:
    sigmas  = sigma_list(cfg_path)
    cfg_raw = yaml.safe_load(cfg_path.read_text())
    is_manifold = bool(cfg_raw.get("smoothing", {}).get("use_manifold", False))

    strategy = strategy_override if strategy_override in ("multi","array") else \
               ("multi" if is_manifold else "array")

    pending, done = [], []
    for sigma in sigmas:
        if (seg_output_dir(cfg_raw, sigma) / "metrics.json").exists():
            done.append(sigma)
        else:
            pending.append(sigma)

    if done:
        skipped.append(f"{short_name}: {len(done)} sigmas already done {done}")
    if not pending:
        skipped.append(f"{short_name}: ALL done — skipping")
        continue

    cfg_raw.setdefault("checkpoint", {})["resume"] = True
    bucket = mani_tasks if is_manifold else iso_tasks

    if strategy == "multi":
        out_cfg = sweep_dir / f"{short_name}_multisigma.yaml"
        out_cfg.write_text(yaml.safe_dump(cfg_raw, sort_keys=False))
        sigmas_str = " ".join(str(s) for s in pending)
        bucket.append(f"{short_name}|{out_cfg}|{sigmas_str}|multi")
        print(f"  {short_name}: strategy=multi  {len(pending)} sigmas in one job")
    else:
        for sigma in pending:
            cfg_s = dict(cfg_raw)
            cfg_s.setdefault("smoothing", {})["sigma"] = float(sigma)
            out_cfg = sweep_dir / f"{short_name}_{sigma_tag(sigma)}.yaml"
            out_cfg.write_text(yaml.safe_dump(cfg_s, sort_keys=False))
            bucket.append(f"{short_name}_{sigma_tag(sigma)}|{out_cfg}|{sigma}|array")
        print(f"  {short_name}: strategy=array  {len(pending)} tasks")

if skipped:
    print("SKIP INFO:")
    for s in skipped: print("  " + s)

# Interleave iso/mani: iso_σ0, mani_σ0, iso_σ1, mani_σ1, ...
lines = []
for i in range(max(len(iso_tasks), len(mani_tasks))):
    if i < len(iso_tasks):  lines.append(iso_tasks[i])
    if i < len(mani_tasks): lines.append(mani_tasks[i])

print(f"  Interleaved: {len(iso_tasks)} iso + {len(mani_tasks)} mani = {len(lines)} total tasks")

if not lines:
    print("All sigma jobs done — nothing to submit.")
    raise SystemExit(0)

task_file.write_text("\n".join(lines) + "\n")
print(f"TASK_COUNT={len(lines)}")
PY

TASK_COUNT=$(wc -l < "$TASK_FILE" | awk '{print $1}')
ARRAY_SPEC="1-${TASK_COUNT}%${MAX_CONCURRENT}"

echo "=============================================="
echo "SUBMITTING SEG SIGMA SWEEP"
echo "=============================================="
echo "Task file:       $TASK_FILE"
echo "Total tasks:     $TASK_COUNT"
echo "Concurrency cap: $MAX_CONCURRENT"
echo "Array spec:      $ARRAY_SPEC"
echo "Partition:       $PARTITION"
echo "Time:            $TIME"
echo "Dry-run:         $DRY_RUN"
echo "=============================================="

# ── Runner script ─────────────────────────────────────────────────────────────
JOB_SCRIPT="$PROJECT_ROOT/output/slurm/seg_sigma_runner_${RUN_ID}.sh"
cat > "$JOB_SCRIPT" << 'JOBEOF'
#!/usr/bin/env bash
set -euo pipefail
TASK_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "TASK_FILE_PLACEHOLDER")
# Format: name|config_path|sigmas_space_separated|strategy
IFS='|' read -r TASK_NAME TASK_CFG TASK_SIGMAS TASK_STRATEGY <<< "$TASK_LINE"
echo "[Task ${SLURM_ARRAY_TASK_ID}] $TASK_NAME  strategy=$TASK_STRATEGY  sigmas=$TASK_SIGMAS"
cd PROJECT_ROOT_PLACEHOLDER
export PYTHONPATH=PROJECT_ROOT_PLACEHOLDER:${PYTHONPATH:-}
. /BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh
conda activate smoothing

# Both multi and array route to the same entry point — Python handles single vs multi sigma
python -m src.experiments.certify.celebahq_segmentation \
    --config "$TASK_CFG" \
    --sigmas $TASK_SIGMAS VIZ_ONLY_PLACEHOLDER
JOBEOF

sed -i "s|TASK_FILE_PLACEHOLDER|${TASK_FILE}|g"     "$JOB_SCRIPT"
sed -i "s|PROJECT_ROOT_PLACEHOLDER|${PROJECT_ROOT}|g" "$JOB_SCRIPT"
VIZ_ARG=""; $VIZ_ONLY && VIZ_ARG="--viz-only"
sed -i "s|VIZ_ONLY_PLACEHOLDER|${VIZ_ARG}|g" "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

SBATCH_CMD="sbatch \
  --partition=$PARTITION \
  --time=$TIME \
  --gres=gpu:$GPUS \
  --cpus-per-task=$CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --array=$ARRAY_SPEC \
  --job-name=seg-sigma \
  --output=$PROJECT_ROOT/output/slurm/seg-sigma-%A_%a.out \
  --error=$PROJECT_ROOT/output/slurm/seg-sigma-%A_%a.err \
  $JOB_SCRIPT"

if $DRY_RUN; then
    echo ""
    echo "[DRY-RUN] Job script: $JOB_SCRIPT"
    echo "[DRY-RUN] Command:"
    echo "$SBATCH_CMD"
else
    eval "$SBATCH_CMD"
    echo ""
    echo "Submitted. Monitor with: squeue -u \$USER"
    echo "Outputs: output/segmentation/celebahq/certify/"
fi
