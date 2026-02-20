#!/usr/bin/env python3
"""Generate comparison visualizations from training results."""

import sys
sys.path.insert(0, ".")

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"mamba": "#EF553B", "cnn": "#636EFA", "transformer": "#00CC96"}
LABELS = {"mamba": "Mamba SSM", "cnn": "1D-CNN", "transformer": "Transformer"}

TASK_CATEGORIES = {
    "Promoter": ["promoter_all", "promoter_tata", "promoter_no_tata"],
    "Enhancer": ["enhancers", "enhancers_types"],
    "Splice Site": ["splice_sites_all", "splice_sites_acceptor", "splice_sites_donor"],
    "Histone": ["H3", "H4", "H3K9ac", "H3K14ac", "H4ac",
                "H3K4me1", "H3K4me2", "H3K4me3", "H3K36me3", "H3K79me3"],
}


def plot_accuracy_comparison(results, results_dir):
    """Per-task accuracy comparison bar chart."""
    models = [m for m in ["mamba", "cnn", "transformer"] if m in results]
    tasks = list(results[models[0]].keys())
    tasks = [t for t in tasks if "accuracy" in results[models[0]].get(t, {})]

    n_tasks = len(tasks)
    n_models = len(models)
    x = np.arange(n_tasks)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(16, n_tasks * 0.9), 6))

    for i, model in enumerate(models):
        accs = [results[model].get(t, {}).get("accuracy", 0) for t in tasks]
        bars = ax.bar(x + i * width - width * (n_models - 1) / 2, accs,
                      width, label=LABELS.get(model, model),
                      color=COLORS.get(model, "#999"), edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Task Accuracy Comparison", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Random")

    plt.tight_layout()
    path = results_dir / "accuracy_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_category_summary(results, results_dir):
    """Category-level accuracy summary."""
    models = [m for m in ["mamba", "cnn", "transformer"] if m in results]
    categories = list(TASK_CATEGORIES.keys())

    fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 5))

    for idx, (cat, cat_tasks) in enumerate(TASK_CATEGORIES.items()):
        ax = axes[idx]
        for model in models:
            valid_accs = [results[model].get(t, {}).get("accuracy", 0) for t in cat_tasks
                         if t in results[model]]
            if valid_accs:
                mean = np.mean(valid_accs)
                std = np.std(valid_accs) if len(valid_accs) > 1 else 0
                ax.bar(LABELS.get(model, model), mean,
                       yerr=std, capsize=5,
                       color=COLORS.get(model, "#999"),
                       edgecolor="white", linewidth=1)

        ax.set_title(cat, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy" if idx == 0 else "")

    plt.suptitle("Accuracy by Task Category", fontweight="bold", fontsize=14)
    plt.tight_layout()
    path = results_dir / "category_summary.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_metrics_radar(results, results_dir):
    """Radar plot of MCC scores by category."""
    models = [m for m in ["mamba", "cnn", "transformer"] if m in results]
    categories = list(TASK_CATEGORIES.keys())
    n_cats = len(categories)

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in models:
        values = []
        for cat_tasks in TASK_CATEGORIES.values():
            mccs = [results[model].get(t, {}).get("mcc", 0) for t in cat_tasks
                    if t in results[model]]
            values.append(np.mean(mccs) if mccs else 0)
        values += values[:1]

        ax.plot(angles, values, "-o", linewidth=2, label=LABELS.get(model, model),
                color=COLORS.get(model, "#999"))
        ax.fill(angles, values, alpha=0.1, color=COLORS.get(model, "#999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("MCC by Category (Radar)", fontweight="bold", fontsize=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    path = results_dir / "mcc_radar.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_efficiency(results, results_dir):
    """Training time and parameter count comparison."""
    models = [m for m in ["mamba", "cnn", "transformer"] if m in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Total training time
    for model in models:
        total_time = sum(r.get("train_time", 0) for r in results[model].values()
                        if "train_time" in r)
        axes[0].bar(LABELS.get(model, model), total_time / 60,
                    color=COLORS.get(model, "#999"), edgecolor="white", linewidth=1)
    axes[0].set_ylabel("Total Training Time (min)")
    axes[0].set_title("Training Efficiency", fontweight="bold")

    # Parameter count
    for model in models:
        first_task = next((r for r in results[model].values() if "n_params" in r), {})
        n_params = first_task.get("n_params", 0) / 1e6
        axes[1].bar(LABELS.get(model, model), n_params,
                    color=COLORS.get(model, "#999"), edgecolor="white", linewidth=1)
    axes[1].set_ylabel("Parameters (M)")
    axes[1].set_title("Model Size", fontweight="bold")

    plt.tight_layout()
    path = results_dir / "efficiency.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def main():
    results_dir = Path("results")
    results_path = results_dir / "all_results.json"

    if not results_path.exists():
        logger.error(f"No results found at {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    plot_accuracy_comparison(results, results_dir)
    plot_category_summary(results, results_dir)
    plot_metrics_radar(results, results_dir)
    plot_efficiency(results, results_dir)

    logger.info("All visualizations generated.")


if __name__ == "__main__":
    main()
