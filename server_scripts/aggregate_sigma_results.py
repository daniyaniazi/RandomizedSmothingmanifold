#!/usr/bin/env python3
"""
Aggregate certification results across sigma values for analysis.

Reads metrics.json from each sigma directory and creates:
1. Combined CSV with all results
2. Summary table for comparison
3. Data ready for plotting certified accuracy vs radius curves

Usage:
    python server_scripts/aggregate_sigma_results.py \
        --results-dir output/smile_classification/celeba/certify/latent_manifold \
        --output output/smile_classification/celeba/certify/latent_manifold/sigma_comparison.csv

    python server_scripts/aggregate_sigma_results.py \
        --results-dir output/ner_conll2003_bert/certify \
        --output output/ner_conll2003_bert/certify/sigma_comparison.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import csv


def find_sigma_dirs(results_dir: Path) -> List[Path]:
    """Find all sigma_X_XX directories."""
    sigma_dirs = []
    for d in results_dir.iterdir():
        if d.is_dir() and d.name.startswith("sigma_"):
            sigma_dirs.append(d)
    return sorted(sigma_dirs, key=lambda x: float(x.name.replace("sigma_", "").replace("_", ".")))


def parse_sigma_from_dirname(dirname: str) -> float:
    """Extract sigma value from directory name like 'sigma_0_25'."""
    return float(dirname.replace("sigma_", "").replace("_", "."))


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics.json file."""
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def aggregate_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Aggregate results from all sigma directories."""
    results_dir = Path(results_dir)
    
    # Handle nested structure: look for sigma dirs or mode dirs containing sigma dirs
    sigma_dirs = find_sigma_dirs(results_dir)
    
    if not sigma_dirs:
        # Maybe results_dir is the certify dir with mode subdirs
        all_results = []
        for mode_dir in results_dir.iterdir():
            if mode_dir.is_dir():
                mode_sigma_dirs = find_sigma_dirs(mode_dir)
                for sd in mode_sigma_dirs:
                    metrics = load_metrics(sd / "metrics.json")
                    if metrics:
                        metrics["sigma"] = parse_sigma_from_dirname(sd.name)
                        metrics["mode"] = mode_dir.name
                        metrics["path"] = str(sd)
                        all_results.append(metrics)
        return all_results
    
    # Direct sigma dirs
    results = []
    for sigma_dir in sigma_dirs:
        metrics = load_metrics(sigma_dir / "metrics.json")
        if metrics:
            metrics["sigma"] = parse_sigma_from_dirname(sigma_dir.name)
            metrics["path"] = str(sigma_dir)
            results.append(metrics)
    
    return results


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested metrics dict for CSV output."""
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f"{key}_{subkey}"] = subvalue
        else:
            flat[key] = value
    return flat


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a formatted summary table."""
    if not results:
        print("No results found!")
        return
    
    print("\n" + "=" * 80)
    print("SIGMA SWEEP RESULTS SUMMARY")
    print("=" * 80)
    
    # Common metrics to display
    key_metrics = [
        "sigma",
        "clean_accuracy", 
        "certified_accuracy",
        "certified_rate",
        "mean_certified_radius",
        "abstain_rate",
    ]
    
    # Check which metrics exist
    available_metrics = []
    for m in key_metrics:
        for r in results:
            flat = flatten_metrics(r)
            if m in flat or m in r:
                available_metrics.append(m)
                break
    
    # Print header
    header = " | ".join(f"{m:>20}" for m in available_metrics)
    print(header)
    print("-" * len(header))
    
    # Print rows
    for r in sorted(results, key=lambda x: x.get("sigma", 0)):
        flat = flatten_metrics(r)
        flat.update(r)  # Ensure top-level keys are included
        
        row_values = []
        for m in available_metrics:
            val = flat.get(m, "N/A")
            if isinstance(val, float):
                row_values.append(f"{val:>20.4f}")
            else:
                row_values.append(f"{str(val):>20}")
        
        print(" | ".join(row_values))
    
    print("=" * 80)


def save_to_csv(results: List[Dict[str, Any]], output_path: Path):
    """Save aggregated results to CSV."""
    if not results:
        print("No results to save!")
        return
    
    # Flatten all results
    flat_results = [flatten_metrics(r) for r in results]
    
    # Get all unique keys
    all_keys = set()
    for r in flat_results:
        all_keys.update(r.keys())
    
    # Sort keys with important ones first
    priority_keys = ["sigma", "mode", "clean_accuracy", "certified_accuracy", 
                     "certified_rate", "mean_certified_radius", "abstain_rate"]
    sorted_keys = [k for k in priority_keys if k in all_keys]
    sorted_keys += sorted(k for k in all_keys if k not in priority_keys)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(flat_results, key=lambda x: (x.get("mode", ""), x.get("sigma", 0))):
            writer.writerow(r)
    
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate sigma sweep results")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing sigma_X_XX subdirectories")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results_dir/sigma_comparison.csv)")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "sigma_comparison.csv"
    
    print(f"Scanning: {results_dir}")
    results = aggregate_results(results_dir)
    
    print(f"Found {len(results)} result sets")
    
    print_summary_table(results)
    save_to_csv(results, output_path)


if __name__ == "__main__":
    main()
