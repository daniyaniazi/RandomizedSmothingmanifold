"""Metrics output formatting for certification experiments.

Provides clean, standardized metrics output:
- JSON for programmatic access
- CSV for spreadsheet analysis
- Markdown for reports
- Console pretty-print

Supports per-experiment and cross-experiment comparison.
"""

from __future__ import annotations

import json
import csv
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np

from .output_schema import (
    CertificationMetricsSummary,
    PerClassMetrics,
    PerClassMetricsCollection,
    ExperimentPaths,
)


def compute_metrics_summary(
    predictions: List[int],
    true_labels: List[int],
    radii: List[float],
    abstained: List[bool],
    sigma: float,
) -> CertificationMetricsSummary:
    """Compute summary metrics from certification results.
    
    Args:
        predictions: Predicted labels
        true_labels: Ground truth labels
        radii: Certified radii
        abstained: Whether each sample abstained
        sigma: Noise sigma used
        
    Returns:
        CertificationMetricsSummary
    """
    n = len(predictions)
    
    # Task accuracy (ignoring abstentions)
    non_abstained_mask = [not a for a in abstained]
    non_abstained_correct = sum(
        1 for p, t, a in zip(predictions, true_labels, abstained)
        if not a and p == t
    )
    non_abstained_total = sum(non_abstained_mask)
    task_accuracy = non_abstained_correct / non_abstained_total if non_abstained_total > 0 else 0.0
    
    # Certified accuracy (correct AND not abstained)
    certified_correct = sum(
        1 for p, t, a in zip(predictions, true_labels, abstained)
        if not a and p == t
    )
    certified_accuracy = certified_correct / n
    
    # Abstention rate
    abstention_rate = sum(abstained) / n
    
    # Radius statistics (for certified samples only)
    valid_radii = [r for r, a, p, t in zip(radii, abstained, predictions, true_labels)
                   if not a and p == t and r > 0]
    
    return CertificationMetricsSummary(
        certified_accuracy=certified_accuracy,
        task_accuracy=task_accuracy,
        abstention_rate=abstention_rate,
        mean_radius=float(np.mean(valid_radii)) if valid_radii else 0.0,
        median_radius=float(np.median(valid_radii)) if valid_radii else 0.0,
        std_radius=float(np.std(valid_radii)) if valid_radii else 0.0,
        min_radius=float(np.min(valid_radii)) if valid_radii else 0.0,
        max_radius=float(np.max(valid_radii)) if valid_radii else 0.0,
        total_samples=n,
        certified_samples=certified_correct,
        sigma=sigma,
    )


def compute_per_class_metrics(
    predictions: List[int],
    true_labels: List[int],
    radii: List[float],
    abstained: List[bool],
    label_names: Optional[Dict[int, str]] = None,
) -> PerClassMetricsCollection:
    """Compute per-class certification metrics.
    
    Args:
        predictions: Predicted labels
        true_labels: Ground truth labels  
        radii: Certified radii
        abstained: Whether each sample abstained
        label_names: Optional mapping from label int to name
        
    Returns:
        PerClassMetricsCollection
    """
    classes = set(true_labels)
    per_class = {}
    
    for cls in classes:
        mask = [t == cls for t in true_labels]
        cls_preds = [p for p, m in zip(predictions, mask) if m]
        cls_true = [t for t, m in zip(true_labels, mask) if m]
        cls_radii = [r for r, m in zip(radii, mask) if m]
        cls_abstained = [a for a, m in zip(abstained, mask) if m]
        
        total = len(cls_preds)
        certified = sum(1 for p, t, a in zip(cls_preds, cls_true, cls_abstained)
                       if not a and p == t)
        abs_count = sum(cls_abstained)
        
        valid_radii = [r for r, a, p, t in zip(cls_radii, cls_abstained, cls_preds, cls_true)
                       if not a and p == t and r > 0]
        
        class_name = label_names[cls] if label_names else str(cls)
        
        per_class[class_name] = PerClassMetrics(
            class_label=class_name,
            total_samples=total,
            certified_samples=certified,
            abstained_samples=abs_count,
            certified_accuracy=certified / total if total > 0 else 0.0,
            mean_radius=float(np.mean(valid_radii)) if valid_radii else 0.0,
            std_radius=float(np.std(valid_radii)) if valid_radii else 0.0,
        )
    
    return PerClassMetricsCollection(metrics=per_class)


def save_metrics_json(
    summary: CertificationMetricsSummary,
    per_class: Optional[PerClassMetricsCollection] = None,
    output_path: Path = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save metrics to JSON format.
    
    Args:
        summary: Summary metrics
        per_class: Optional per-class metrics
        output_path: Path to save JSON
        additional_metadata: Extra fields to include
        
    Returns:
        Dict representation of metrics
    """
    result = {
        "summary": asdict(summary),
        "metadata": additional_metadata or {},
    }
    
    if per_class:
        result["per_class"] = {k: asdict(v) for k, v in per_class.metrics.items()}
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    
    return result


def save_metrics_csv(
    experiments: List[Dict[str, Any]],
    output_path: Path,
    include_per_class: bool = False,
) -> None:
    """Save multiple experiments to CSV for comparison.
    
    Args:
        experiments: List of dicts with experiment name and CertificationMetricsSummary
        output_path: Path to save CSV
        include_per_class: Whether to include per-class breakdown
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "experiment",
        "smoothing_mode",
        "space",
        "layer",
        "masking",
        "sigma",
        "certified_accuracy",
        "task_accuracy",
        "abstention_rate",
        "mean_radius",
        "median_radius",
        "std_radius",
        "total_samples",
        "certified_samples",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for exp in experiments:
            summary = exp.get("summary")
            if summary is None:
                continue
            
            row = {
                "experiment": exp.get("name", ""),
                "smoothing_mode": exp.get("smoothing_mode", ""),
                "space": exp.get("space", ""),
                "layer": exp.get("layer", ""),
                "masking": exp.get("masking", ""),
                "sigma": summary.sigma,
                "certified_accuracy": f"{summary.certified_accuracy:.4f}",
                "task_accuracy": f"{summary.task_accuracy:.4f}",
                "abstention_rate": f"{summary.abstention_rate:.4f}",
                "mean_radius": f"{summary.mean_radius:.4f}",
                "median_radius": f"{summary.median_radius:.4f}",
                "std_radius": f"{summary.std_radius:.4f}",
                "total_samples": summary.total_samples,
                "certified_samples": summary.certified_samples,
            }
            writer.writerow(row)


def format_metrics_markdown(
    summary: CertificationMetricsSummary,
    per_class: Optional[PerClassMetricsCollection] = None,
    title: str = "Certification Results",
) -> str:
    """Format metrics as Markdown for reports.
    
    Args:
        summary: Summary metrics
        per_class: Optional per-class metrics
        title: Section title
        
    Returns:
        Markdown string
    """
    lines = [
        f"# {title}",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Certified Accuracy | {summary.certified_accuracy*100:.2f}% |",
        f"| Task Accuracy | {summary.task_accuracy*100:.2f}% |",
        f"| Abstention Rate | {summary.abstention_rate*100:.2f}% |",
        f"| Mean Certified Radius | {summary.mean_radius:.4f} |",
        f"| Median Certified Radius | {summary.median_radius:.4f} |",
        f"| Std Certified Radius | {summary.std_radius:.4f} |",
        f"| Min Certified Radius | {summary.min_radius:.4f} |",
        f"| Max Certified Radius | {summary.max_radius:.4f} |",
        f"| Total Samples | {summary.total_samples} |",
        f"| Certified Samples | {summary.certified_samples} |",
        f"| Sigma (σ) | {summary.sigma} |",
        "",
    ]
    
    if per_class:
        lines.extend([
            "## Per-Class Metrics",
            "",
            "| Class | Total | Certified | Abstained | Cert. Acc. | Mean Radius |",
            "|-------|-------|-----------|-----------|------------|-------------|",
        ])
        
        for name, metrics in sorted(per_class.metrics.items()):
            lines.append(
                f"| {name} | {metrics.total_samples} | {metrics.certified_samples} | "
                f"{metrics.abstained_samples} | {metrics.certified_accuracy*100:.2f}% | "
                f"{metrics.mean_radius:.4f} |"
            )
        lines.append("")
    
    return "\n".join(lines)


def format_metrics_console(
    summary: CertificationMetricsSummary,
    per_class: Optional[PerClassMetricsCollection] = None,
    title: str = "Certification Results",
) -> str:
    """Format metrics for console output.
    
    Args:
        summary: Summary metrics
        per_class: Optional per-class metrics
        title: Section title
        
    Returns:
        Console-formatted string
    """
    sep = "=" * 60
    lines = [
        sep,
        f"  {title}",
        sep,
        "",
        "  SUMMARY METRICS",
        "  " + "-" * 40,
        f"  Certified Accuracy:      {summary.certified_accuracy*100:6.2f}%",
        f"  Task Accuracy:           {summary.task_accuracy*100:6.2f}%",
        f"  Abstention Rate:         {summary.abstention_rate*100:6.2f}%",
        "",
        f"  Mean Certified Radius:   {summary.mean_radius:.4f}",
        f"  Median Certified Radius: {summary.median_radius:.4f}",
        f"  Std Certified Radius:    {summary.std_radius:.4f}",
        f"  Range:                   [{summary.min_radius:.4f}, {summary.max_radius:.4f}]",
        "",
        f"  Total Samples:           {summary.total_samples}",
        f"  Certified Samples:       {summary.certified_samples}",
        f"  Sigma (σ):               {summary.sigma}",
        "",
    ]
    
    if per_class:
        lines.extend([
            "  PER-CLASS METRICS",
            "  " + "-" * 40,
            f"  {'Class':<12} {'Total':>6} {'Cert':>6} {'Abs':>5} {'Acc':>7} {'Radius':>8}",
            "  " + "-" * 48,
        ])
        
        for name, metrics in sorted(per_class.metrics.items()):
            lines.append(
                f"  {name:<12} {metrics.total_samples:>6} {metrics.certified_samples:>6} "
                f"{metrics.abstained_samples:>5} {metrics.certified_accuracy*100:>6.1f}% "
                f"{metrics.mean_radius:>8.4f}"
            )
        lines.append("")
    
    lines.append(sep)
    return "\n".join(lines)


def load_experiment_metrics(experiment_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment output directory
        
    Returns:
        Dict with metrics or None if not found
    """
    metrics_path = experiment_dir / "metrics.json"
    summary_path = experiment_dir / "results_summary.json"
    
    for path in [metrics_path, summary_path]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    
    return None


def compare_experiments(
    experiment_dirs: List[Path],
    output_path: Optional[Path] = None,
) -> str:
    """Compare multiple experiments side-by-side.
    
    Args:
        experiment_dirs: List of experiment directories
        output_path: Optional path to save comparison CSV
        
    Returns:
        Markdown comparison table
    """
    experiments = []
    
    for exp_dir in experiment_dirs:
        metrics = load_experiment_metrics(exp_dir)
        if metrics:
            # Parse experiment name for metadata
            name = exp_dir.name
            parts = name.split("_")
            
            summary_data = metrics.get("summary", metrics)
            
            summary = CertificationMetricsSummary(
                certified_accuracy=summary_data.get("certified_accuracy", 0),
                task_accuracy=summary_data.get("task_accuracy", summary_data.get("accuracy", 0)),
                abstention_rate=summary_data.get("abstention_rate", 0),
                mean_radius=summary_data.get("mean_radius", 0),
                median_radius=summary_data.get("median_radius", 0),
                std_radius=summary_data.get("std_radius", 0),
                min_radius=summary_data.get("min_radius", 0),
                max_radius=summary_data.get("max_radius", 0),
                total_samples=summary_data.get("total_samples", summary_data.get("n_samples", 0)),
                certified_samples=summary_data.get("certified_samples", 0),
                sigma=summary_data.get("sigma", 0),
            )
            
            experiments.append({
                "name": name,
                "summary": summary,
                "smoothing_mode": summary_data.get("smoothing_mode", ""),
                "space": summary_data.get("space", ""),
                "layer": summary_data.get("layer", ""),
                "masking": summary_data.get("masking", ""),
            })
    
    if output_path:
        save_metrics_csv(experiments, output_path)
    
    # Generate markdown table
    lines = [
        "# Experiment Comparison",
        "",
        "| Experiment | Cert. Acc. | Task Acc. | Abs. Rate | Mean R | Sigma |",
        "|------------|------------|-----------|-----------|--------|-------|",
    ]
    
    for exp in experiments:
        s = exp["summary"]
        lines.append(
            f"| {exp['name'][:30]} | {s.certified_accuracy*100:.1f}% | "
            f"{s.task_accuracy*100:.1f}% | {s.abstention_rate*100:.1f}% | "
            f"{s.mean_radius:.3f} | {s.sigma} |"
        )
    
    return "\n".join(lines)
