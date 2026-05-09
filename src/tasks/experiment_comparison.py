"""Cross-experiment comparison utilities.

Tools for comparing certification results across:
- Different smoothing modes (isotropic vs manifold)
- Different spaces (pixel, latent, hidden state)
- Different layers (for transformer models)
- Different masking strategies
- Different sigma values

Generates comparison tables, plots, and summary reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json
import re

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .output_schema import CertificationMetricsSummary
from .metrics_output import load_experiment_metrics, compare_experiments


@dataclass
class ExperimentConfig:
    """Parsed experiment configuration."""
    name: str
    smoothing_mode: str  # "isotropic" or "manifold"
    space: str  # "pixel", "latent", "hidden_state"
    layer: Optional[str]  # For transformers: "6", "12", etc.
    masking: bool  # Whether masking was used
    sigma: float
    task: str  # "image_classification", "token_classification"
    path: Path


def parse_experiment_name(name: str, path: Path) -> ExperimentConfig:
    """Parse experiment name to extract configuration.
    
    Expected format: {smoothing_mode}_{space}_{layer}_{masking}_{sigma}
    Examples:
        - manifold_hidden_state_layer6_masked_0.5
        - isotropic_hidden_state_layer12_unmasked_1.0
        - manifold_latent_vae_nomask_0.25
    """
    parts = name.lower().split("_")
    
    # Detect smoothing mode
    smoothing_mode = "manifold" if "manifold" in parts or "pca" in parts else "isotropic"
    
    # Detect space
    space = "hidden_state"
    if "pixel" in parts:
        space = "pixel"
    elif "latent" in parts or "vae" in parts:
        space = "latent"
    
    # Detect layer
    layer = None
    for part in parts:
        if part.startswith("layer"):
            layer = part.replace("layer", "")
        elif re.match(r"^\d+$", part):
            layer = part
    
    # Detect masking
    masking = any(m in parts for m in ["masked", "mask", "masking"])
    
    # Detect sigma
    sigma = 0.5  # default
    for part in parts:
        try:
            val = float(part)
            if 0 < val < 10:  # reasonable sigma range
                sigma = val
        except ValueError:
            pass
    
    # Detect task from path
    task = "token_classification"
    if "celeba" in str(path).lower() or "image" in str(path).lower():
        task = "image_classification"
    
    return ExperimentConfig(
        name=name,
        smoothing_mode=smoothing_mode,
        space=space,
        layer=layer,
        masking=masking,
        sigma=sigma,
        task=task,
        path=path,
    )


def discover_experiments(base_dir: Path) -> List[ExperimentConfig]:
    """Discover all experiment directories in a base directory.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of ExperimentConfig for each found experiment
    """
    experiments = []
    
    # Look for directories with metrics.json or results_summary.json
    for metrics_file in base_dir.rglob("metrics.json"):
        exp_dir = metrics_file.parent
        config = parse_experiment_name(exp_dir.name, exp_dir)
        experiments.append(config)
    
    for metrics_file in base_dir.rglob("results_summary.json"):
        exp_dir = metrics_file.parent
        # Avoid duplicates
        if not any(e.path == exp_dir for e in experiments):
            config = parse_experiment_name(exp_dir.name, exp_dir)
            experiments.append(config)
    
    return experiments


def load_experiment_results(config: ExperimentConfig) -> Optional[Dict[str, Any]]:
    """Load metrics from an experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dict with metrics data
    """
    return load_experiment_metrics(config.path)


def compare_smoothing_modes(
    experiments: List[ExperimentConfig],
    output_path: Optional[Path] = None,
    title: str = "Isotropic vs Manifold Smoothing",
) -> None:
    """Compare isotropic vs manifold smoothing performance.
    
    Creates a grouped bar chart comparing certified accuracy and radius.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    
    # Group experiments by (space, layer, masking, sigma)
    groups: Dict[Tuple, Dict[str, Any]] = {}
    
    for exp in experiments:
        key = (exp.space, exp.layer, exp.masking, exp.sigma)
        if key not in groups:
            groups[key] = {"isotropic": None, "manifold": None}
        
        metrics = load_experiment_results(exp)
        if metrics:
            summary = metrics.get("summary", metrics)
            groups[key][exp.smoothing_mode] = {
                "certified_accuracy": summary.get("certified_accuracy", 0),
                "mean_radius": summary.get("mean_radius", 0),
                "abstention_rate": summary.get("abstention_rate", 0),
            }
    
    # Filter to groups with both modes
    complete_groups = {k: v for k, v in groups.items() if v["isotropic"] and v["manifold"]}
    
    if not complete_groups:
        print("No complete comparison pairs found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(complete_groups))
    width = 0.35
    
    labels = [f"{k[0]}\n{k[1] or 'all'}\n{'mask' if k[2] else 'no-mask'}\nσ={k[3]}" 
              for k in complete_groups.keys()]
    
    iso_acc = [v["isotropic"]["certified_accuracy"] * 100 for v in complete_groups.values()]
    man_acc = [v["manifold"]["certified_accuracy"] * 100 for v in complete_groups.values()]
    
    iso_rad = [v["isotropic"]["mean_radius"] for v in complete_groups.values()]
    man_rad = [v["manifold"]["mean_radius"] for v in complete_groups.values()]
    
    # Certified accuracy
    ax = axes[0]
    bars1 = ax.bar(x - width/2, iso_acc, width, label="Isotropic", color="coral", edgecolor="black")
    bars2 = ax.bar(x + width/2, man_acc, width, label="Manifold", color="steelblue", edgecolor="black")
    ax.set_xlabel("Configuration", fontsize=10)
    ax.set_ylabel("Certified Accuracy (%)", fontsize=10)
    ax.set_title("Certified Accuracy Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Mean radius
    ax = axes[1]
    bars1 = ax.bar(x - width/2, iso_rad, width, label="Isotropic", color="coral", edgecolor="black")
    bars2 = ax.bar(x + width/2, man_rad, width, label="Manifold", color="steelblue", edgecolor="black")
    ax.set_xlabel("Configuration", fontsize=10)
    ax.set_ylabel("Mean Certified Radius", fontsize=10)
    ax.set_title("Certified Radius Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def compare_layers(
    experiments: List[ExperimentConfig],
    output_path: Optional[Path] = None,
    title: str = "Performance Across Layers",
) -> None:
    """Compare certification performance across different transformer layers.
    
    Creates a line plot showing accuracy and radius vs layer.
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Group by (smoothing_mode, masking, sigma)
    series: Dict[Tuple, Dict[int, Dict]] = {}
    
    for exp in experiments:
        if exp.layer is None:
            continue
        
        key = (exp.smoothing_mode, exp.masking, exp.sigma)
        if key not in series:
            series[key] = {}
        
        metrics = load_experiment_results(exp)
        if metrics:
            summary = metrics.get("summary", metrics)
            try:
                layer_num = int(exp.layer)
                series[key][layer_num] = {
                    "certified_accuracy": summary.get("certified_accuracy", 0),
                    "mean_radius": summary.get("mean_radius", 0),
                }
            except ValueError:
                pass
    
    if not series:
        print("No layer comparison data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(series)))
    
    for (sm, mask, sigma), layers_data in series.items():
        label = f"{sm}, {'mask' if mask else 'no-mask'}, σ={sigma}"
        color = colors[list(series.keys()).index((sm, mask, sigma))]
        
        sorted_layers = sorted(layers_data.keys())
        acc = [layers_data[l]["certified_accuracy"] * 100 for l in sorted_layers]
        rad = [layers_data[l]["mean_radius"] for l in sorted_layers]
        
        axes[0].plot(sorted_layers, acc, marker="o", label=label, color=color)
        axes[1].plot(sorted_layers, rad, marker="o", label=label, color=color)
    
    axes[0].set_xlabel("Layer", fontsize=10)
    axes[0].set_ylabel("Certified Accuracy (%)", fontsize=10)
    axes[0].set_title("Certified Accuracy vs Layer", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Layer", fontsize=10)
    axes[1].set_ylabel("Mean Certified Radius", fontsize=10)
    axes[1].set_title("Certified Radius vs Layer", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def compare_sigma_sensitivity(
    experiments: List[ExperimentConfig],
    output_path: Optional[Path] = None,
    title: str = "Sigma Sensitivity Analysis",
) -> None:
    """Analyze how certification metrics change with sigma.
    
    Creates plots showing accuracy vs sigma trade-off.
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Group by (smoothing_mode, space, layer, masking)
    series: Dict[Tuple, Dict[float, Dict]] = {}
    
    for exp in experiments:
        key = (exp.smoothing_mode, exp.space, exp.layer, exp.masking)
        if key not in series:
            series[key] = {}
        
        metrics = load_experiment_results(exp)
        if metrics:
            summary = metrics.get("summary", metrics)
            series[key][exp.sigma] = {
                "certified_accuracy": summary.get("certified_accuracy", 0),
                "mean_radius": summary.get("mean_radius", 0),
                "abstention_rate": summary.get("abstention_rate", 0),
            }
    
    if not series:
        print("No sigma comparison data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(series)))
    
    for (sm, space, layer, mask), sigma_data in series.items():
        label = f"{sm}/{space}"
        if layer:
            label += f"/L{layer}"
        label += f"/{'m' if mask else 'nm'}"
        
        color = colors[list(series.keys()).index((sm, space, layer, mask))]
        
        sorted_sigmas = sorted(sigma_data.keys())
        acc = [sigma_data[s]["certified_accuracy"] * 100 for s in sorted_sigmas]
        rad = [sigma_data[s]["mean_radius"] for s in sorted_sigmas]
        abst = [sigma_data[s]["abstention_rate"] * 100 for s in sorted_sigmas]
        
        axes[0].plot(sorted_sigmas, acc, marker="o", label=label, color=color)
        axes[1].plot(sorted_sigmas, rad, marker="o", label=label, color=color)
        axes[2].plot(sorted_sigmas, abst, marker="o", label=label, color=color)
    
    axes[0].set_xlabel("Sigma (σ)", fontsize=10)
    axes[0].set_ylabel("Certified Accuracy (%)", fontsize=10)
    axes[0].set_title("Accuracy vs Sigma", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Sigma (σ)", fontsize=10)
    axes[1].set_ylabel("Mean Certified Radius", fontsize=10)
    axes[1].set_title("Radius vs Sigma", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=7, loc="best")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel("Sigma (σ)", fontsize=10)
    axes[2].set_ylabel("Abstention Rate (%)", fontsize=10)
    axes[2].set_title("Abstention vs Sigma", fontsize=12, fontweight="bold")
    axes[2].legend(fontsize=7, loc="best")
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def generate_comparison_report(
    base_dir: Path,
    output_dir: Path,
) -> str:
    """Generate comprehensive comparison report.
    
    Args:
        base_dir: Directory containing all experiments
        output_dir: Directory to save report and plots
        
    Returns:
        Markdown report string
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover all experiments
    experiments = discover_experiments(base_dir)
    
    if not experiments:
        return "No experiments found."
    
    # Generate comparison plots
    compare_smoothing_modes(experiments, output_dir / "smoothing_comparison.png")
    compare_layers(experiments, output_dir / "layer_comparison.png")
    compare_sigma_sensitivity(experiments, output_dir / "sigma_sensitivity.png")
    
    # Generate markdown report
    lines = [
        "# Certification Experiment Comparison Report",
        "",
        f"**Total Experiments Found:** {len(experiments)}",
        "",
        "## Experiments Summary",
        "",
        "| Name | Mode | Space | Layer | Masking | Sigma |",
        "|------|------|-------|-------|---------|-------|",
    ]
    
    for exp in sorted(experiments, key=lambda e: (e.smoothing_mode, e.space, str(e.layer or ""), e.masking)):
        lines.append(
            f"| {exp.name[:30]} | {exp.smoothing_mode} | {exp.space} | "
            f"{exp.layer or '-'} | {'Yes' if exp.masking else 'No'} | {exp.sigma} |"
        )
    
    lines.extend([
        "",
        "## Comparison Plots",
        "",
        "### Smoothing Mode Comparison",
        "![Smoothing Comparison](smoothing_comparison.png)",
        "",
        "### Layer-wise Analysis",
        "![Layer Comparison](layer_comparison.png)",
        "",
        "### Sigma Sensitivity",
        "![Sigma Sensitivity](sigma_sensitivity.png)",
        "",
    ])
    
    # Add detailed metrics table
    lines.extend([
        "## Detailed Metrics",
        "",
        compare_experiments([exp.path for exp in experiments]),
    ])
    
    report = "\n".join(lines)
    
    # Save report
    with open(output_dir / "comparison_report.md", "w") as f:
        f.write(report)
    
    return report
