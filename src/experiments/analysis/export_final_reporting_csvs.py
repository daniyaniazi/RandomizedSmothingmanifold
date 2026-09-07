"""Export final thesis/reporting CSVs from completed server results.

This script only reads existing metrics.json files. It writes compact CSVs for:
  - CelebA classification
  - CelebA-HQ classification
  - CelebAMask-HQ segmentation

Reported columns include certified accuracy, abstention rate, isotropic radius,
manifold whitened radius, ambient isotropic volume, and Jonas's unscaled
sigma-based manifold geometry volume when recoverable from saved metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from scipy.special import gammaln

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


SIGMA_RE = re.compile(r"sigma_(\d+)_(\d+)")


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _sigma_from_path(path: Path) -> Optional[float]:
    for part in path.parts:
        m = SIGMA_RE.fullmatch(part)
        if m:
            return float(f"{m.group(1)}.{m.group(2)}")
    return None


def _log_c_ball(dim: int) -> float:
    return (dim / 2.0) * math.log(math.pi) - float(gammaln(dim / 2.0 + 1.0))


def _log_volume_ball(radius: Optional[float], dim: Optional[int]) -> Optional[float]:
    if radius is None or dim is None or radius <= 0 or dim <= 0:
        return None
    return _log_c_ball(int(dim)) + int(dim) * math.log(float(radius))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _mode_from_metrics(metrics: dict, path: Path) -> str:
    smoothing = metrics.get("smoothing", {}) or {}
    if "use_manifold" in smoothing:
        return "manifold" if smoothing.get("use_manifold") else "isotropic"
    text = str(path).lower()
    if "manifold" in text:
        return "manifold"
    if "isotropic" in text:
        return "isotropic"
    return "unknown"


def _space_from_path(path: Path) -> str:
    text = str(path).lower()
    if "latent_" in text:
        return "latent"
    if "pixel_" in text:
        return "pixel"
    return "unknown"


def _volume_fields(metrics: dict, sigma: Optional[float]) -> dict:
    vol = metrics.get("volume", {}) or {}
    geo = vol.get("geometry", {}) or {}
    k_pca = vol.get("k_pca")
    ambient_d = vol.get("ambient_D")
    mean_geom = vol.get("mean_geometry_factor")

    log_v_iso_geo = geo.get("log_v_iso_geo")
    if log_v_iso_geo is None and sigma is not None and k_pca:
        log_v_iso_geo = _log_volume_ball(sigma, int(k_pca))

    log_v_mani_unscaled_geo = None
    if log_v_iso_geo is not None and mean_geom is not None:
        log_v_mani_unscaled_geo = float(log_v_iso_geo) + float(mean_geom)

    return {
        "ambient_D": ambient_d,
        "k_pca": k_pca,
        "logV_iso_geo_k_sigma": log_v_iso_geo,
        "logV_mani_actual_unscaled_cert": vol.get("mean_log_vol_mani_actual"),
        "logV_mani_unscaled_geo_sigma": log_v_mani_unscaled_geo,
        "geometry_factor_raw": mean_geom,
        "mean_effective_rank": vol.get("mean_effective_rank") or geo.get("mean_effective_rank"),
    }


def _classification_row(metrics_path: Path, dataset_name: str) -> dict:
    metrics = _read_json(metrics_path)
    smoothing = metrics.get("smoothing", {}) or {}
    sigma = smoothing.get("sigma")
    if sigma is None:
        sigma = _sigma_from_path(metrics_path)
    sigma = float(sigma) if sigma is not None else None
    mode = _mode_from_metrics(metrics, metrics_path)
    mean_radius = metrics.get("mean_radius")

    row = {
        "task": "classification",
        "dataset": dataset_name,
        "space": _space_from_path(metrics_path),
        "smoother": mode,
        "sigma": sigma,
        "metrics_path": str(metrics_path),
        "total_test_samples": metrics.get("total_test_samples"),
        "certified_accuracy": metrics.get("certified_accuracy"),
        "certified_accuracy_pct": _pct(metrics.get("certified_accuracy")),
        "abstain_rate": metrics.get("abstain_rate"),
        "abstain_rate_pct": _pct(metrics.get("abstain_rate")),
        "mean_radius": mean_radius,
        "median_radius": metrics.get("median_radius"),
        "radius_iso": mean_radius if mode == "isotropic" else None,
        "rwhite_mani": mean_radius if mode == "manifold" else None,
        "class_smile_accuracy": metrics.get("class_smile_accuracy"),
        "class_no_smile_accuracy": metrics.get("class_no_smile_accuracy"),
        "n0_samples": smoothing.get("n0_samples"),
        "n_samples": smoothing.get("n_samples"),
        "knn_k": smoothing.get("knn_k"),
        "pca_dim": smoothing.get("pca_dim"),
    }
    row.update(_volume_fields(metrics, sigma))
    return row


def _segmentation_row(metrics_path: Path) -> dict:
    metrics = _read_json(metrics_path)
    smoothing = metrics.get("smoothing", {}) or {}
    sigma = smoothing.get("sigma")
    if sigma is None:
        sigma = _sigma_from_path(metrics_path)
    sigma = float(sigma) if sigma is not None else None
    mode = _mode_from_metrics(metrics, metrics_path)
    radius = metrics.get("certified_radius")
    image_size = 512
    ambient_d = 3 * image_size * image_size

    row = {
        "task": "segmentation",
        "dataset": "CelebAMask-HQ",
        "space": "pixel",
        "smoother": mode,
        "sigma": sigma,
        "metrics_path": str(metrics_path),
        "total_test_samples": metrics.get("total_test_samples"),
        "certified_accuracy": metrics.get("mean_certified_pixel_acc"),
        "certified_accuracy_pct": _pct(metrics.get("mean_certified_pixel_acc")),
        "certified_miou": metrics.get("mean_certified_miou"),
        "certified_miou_pct": _pct(metrics.get("mean_certified_miou")),
        "clean_pixel_acc": metrics.get("mean_pixel_acc"),
        "clean_miou": metrics.get("mean_miou"),
        "abstain_rate": metrics.get("mean_abstain_rate"),
        "abstain_rate_pct": _pct(metrics.get("mean_abstain_rate")),
        "mean_radius": radius,
        "median_radius": radius,
        "radius_iso": radius if mode == "isotropic" else None,
        "rwhite_mani": radius if mode == "manifold" else None,
        "n0_samples": smoothing.get("n0_samples"),
        "n_samples": smoothing.get("n_samples"),
        "knn_k": smoothing.get("knn_k"),
        "pca_dim": None,
        "ambient_D": ambient_d,
        "k_pca": smoothing.get("knn_k") if mode == "manifold" else None,
        "logV_iso_cert_ambient": _log_volume_ball(radius, ambient_d) if mode == "isotropic" else None,
        "logV_iso_geo_k_sigma": _log_volume_ball(sigma, smoothing.get("knn_k")) if mode == "manifold" else None,
        "logV_mani_actual_unscaled_cert": None,
        "logV_mani_unscaled_geo_sigma": None,
        "geometry_factor_raw": None,
        "mean_effective_rank": None,
    }
    return row


RUN_COLUMNS = [
    "task",
    "dataset",
    "space",
    "smoother",
    "sigma",
    "certified_accuracy_pct",
    "abstain_rate_pct",
    "mean_radius",
    "median_radius",
    "radius_iso",
    "rwhite_mani",
    "logV_iso_geo_k_sigma",
    "logV_mani_unscaled_geo_sigma",
    "logV_mani_actual_unscaled_cert",
    "geometry_factor_raw",
    "mean_effective_rank",
    "k_pca",
    "ambient_D",
    "total_test_samples",
    "n0_samples",
    "n_samples",
    "knn_k",
    "pca_dim",
    "certified_miou_pct",
    "clean_pixel_acc",
    "clean_miou",
    "metrics_path",
]


PAIRED_COLUMNS = [
    "task",
    "dataset",
    "space",
    "sigma",
    "iso_certified_accuracy_pct",
    "mani_certified_accuracy_pct",
    "delta_certified_accuracy_pct_mani_minus_iso",
    "iso_abstain_rate_pct",
    "mani_abstain_rate_pct",
    "delta_abstain_rate_pct_mani_minus_iso",
    "radius_iso",
    "rwhite_mani",
    "logV_iso_geo_k_sigma",
    "logV_mani_unscaled_geo_sigma",
    "logV_mani_actual_unscaled_cert",
    "mani_geometry_factor_raw",
    "mani_effective_rank",
    "mani_k_pca",
    "ambient_D",
]


def _pct(value) -> Optional[float]:
    return None if value is None else 100.0 * float(value)


def _write_csv(rows: List[dict], path: Path, columns: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if columns is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
    else:
        keys = [key for key in columns if any(key in row for row in rows)]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _collect_classification(root: Path, dataset_tag: str) -> List[dict]:
    base = root / "output" / "smile_classification" / dataset_tag / "certify"
    rows = []
    for metrics_path in sorted(base.glob("pixel_*/sigma_*/metrics.json")):
        rows.append(_classification_row(metrics_path, dataset_tag))
    for metrics_path in sorted(base.glob("latent_*/sigma_*/metrics.json")):
        rows.append(_classification_row(metrics_path, dataset_tag))
    return sorted(rows, key=lambda r: (str(r["space"]), str(r["smoother"]), float(r["sigma"] or -1)))


def _collect_segmentation(root: Path) -> List[dict]:
    base = root / "output" / "segmentation" / "celebamaskhq" / "certify"
    rows = [_segmentation_row(p) for p in sorted(base.glob("pixel_*/sigma_*/metrics.json"))]
    return sorted(rows, key=lambda r: (str(r["smoother"]), float(r["sigma"] or -1)))


def _paired_table(rows: List[dict]) -> List[dict]:
    grouped = {}
    for row in rows:
        key = (row["task"], row["dataset"], row["space"], row["sigma"])
        grouped.setdefault(key, {})[row["smoother"]] = row

    out = []
    for (task, dataset, space, sigma), modes in sorted(grouped.items()):
        iso = modes.get("isotropic", {})
        mani = modes.get("manifold", {})
        out.append({
            "task": task,
            "dataset": dataset,
            "space": space,
            "sigma": sigma,
            "iso_certified_accuracy_pct": iso.get("certified_accuracy_pct"),
            "mani_certified_accuracy_pct": mani.get("certified_accuracy_pct"),
            "delta_certified_accuracy_pct_mani_minus_iso": _diff(
                mani.get("certified_accuracy_pct"), iso.get("certified_accuracy_pct")
            ),
            "iso_abstain_rate_pct": iso.get("abstain_rate_pct"),
            "mani_abstain_rate_pct": mani.get("abstain_rate_pct"),
            "delta_abstain_rate_pct_mani_minus_iso": _diff(
                mani.get("abstain_rate_pct"), iso.get("abstain_rate_pct")
            ),
            "radius_iso": iso.get("radius_iso") or iso.get("mean_radius"),
            "rwhite_mani": mani.get("rwhite_mani") or mani.get("mean_radius"),
            "logV_iso_geo_k_sigma": mani.get("logV_iso_geo_k_sigma") or iso.get("logV_iso_geo_k_sigma"),
            "logV_mani_unscaled_geo_sigma": mani.get("logV_mani_unscaled_geo_sigma"),
            "logV_mani_actual_unscaled_cert": mani.get("logV_mani_actual_unscaled_cert"),
            "mani_geometry_factor_raw": mani.get("geometry_factor_raw"),
            "mani_effective_rank": mani.get("mean_effective_rank"),
            "mani_k_pca": mani.get("k_pca"),
            "ambient_D": mani.get("ambient_D") or iso.get("ambient_D"),
        })
    return out


def _diff(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export final reporting CSVs")
    p.add_argument("--repo-root", type=Path, default=_ROOT)
    p.add_argument("--output-dir", type=Path, default=_ROOT / "output" / "final_reporting")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    celeba = _collect_classification(root, "celeba")
    celebahq = _collect_classification(root, "celebahq")
    seg = _collect_segmentation(root)

    _write_csv(celeba, out / "celeba_classification_runs.csv", RUN_COLUMNS)
    _write_csv(celebahq, out / "celebahq_classification_runs.csv", RUN_COLUMNS)
    _write_csv(seg, out / "celebamaskhq_segmentation_runs.csv", RUN_COLUMNS)

    _write_csv(_paired_table(celeba), out / "celeba_classification_iso_vs_mani.csv", PAIRED_COLUMNS)
    _write_csv(_paired_table(celebahq), out / "celebahq_classification_iso_vs_mani.csv", PAIRED_COLUMNS)
    _write_csv(_paired_table(seg), out / "celebamaskhq_segmentation_iso_vs_mani.csv", PAIRED_COLUMNS)
    _write_csv(_paired_table(celeba + celebahq + seg), out / "all_tasks_iso_vs_mani.csv", PAIRED_COLUMNS)

    manifest = {
        "output_dir": str(out),
        "n_celeba_classification_runs": len(celeba),
        "n_celebahq_classification_runs": len(celebahq),
        "n_celebamaskhq_segmentation_runs": len(seg),
        "notes": {
            "radius_iso": "Mean certified radius for isotropic classification; certified radius for segmentation.",
            "rwhite_mani": "Mean manifold certified radius in whitened coordinates for classification; certified radius for segmentation.",
            "logV_iso_geo_k_sigma": "Dimension-matched k-dimensional isotropic geometry baseline: log(C_k sigma^k).",
            "mani_logV_unscaled_geo_sigma": "Jonas unscaled sigma-based geometry volume: log(C_k sigma^k sqrt(det Lambda)); recovered as logV_iso_geo + geometry_factor_raw.",
            "segmentation_volume_limitation": "Segmentation runs do not save manifold eigenvalues/geometry_factor, so manifold Jonas volume is blank unless those are added in future runs.",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    _log(f"Saved final reporting CSVs to {out}")
    _log(f"CelebA classification rows: {len(celeba)}")
    _log(f"CelebA-HQ classification rows: {len(celebahq)}")
    _log(f"Segmentation rows: {len(seg)}")


if __name__ == "__main__":
    main()
