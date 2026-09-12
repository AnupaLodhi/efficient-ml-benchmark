"""Generate the 6 required figures from results/processed/results_processed.csv.

Every plotting function takes a DataFrame and an output path and does
nothing else clever -- no fabricated data, no smoothing/interpolation that
would misrepresent discrete experiment points as a continuous trend. If a
technique produced too few points to plot meaningfully (e.g. only 1 row),
the function logs a warning and skips that series rather than plotting a
single point as if it were a line.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # headless -- these scripts must run without a display
sns.set_theme(style="whitegrid", context="paper")

TECHNIQUE_COLORS = {
    "baseline": "#1f1f1f",
    "pruning": "#2563eb",
    "quantization": "#dc2626",
    "combined": "#16a34a",
}
TECHNIQUE_MARKERS = {
    "baseline": "*",
    "pruning": "o",
    "quantization": "s",
    "combined": "^",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _scatter_by_technique(df: pd.DataFrame, x_col: str, y_col: str, xlabel: str, ylabel: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for technique, group in df.groupby("technique"):
        group = group.dropna(subset=[x_col, y_col])
        if group.empty:
            continue
        ax.scatter(
            group[x_col], group[y_col],
            label=technique,
            color=TECHNIQUE_COLORS.get(technique, "#888888"),
            marker=TECHNIQUE_MARKERS.get(technique, "o"),
            s=70, edgecolors="white", linewidths=0.5, alpha=0.9,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Technique", frameon=True)
    return fig


def plot_accuracy_vs_compression_ratio(df: pd.DataFrame, out_path: Path) -> None:
    fig = _scatter_by_technique(
        df, "compression_ratio", "classification_accuracy",
        "Compression Ratio (baseline size / model size)", "Test Accuracy",
        "Accuracy vs. Compression Ratio",
    )
    _save(fig, out_path)


def plot_accuracy_vs_latency(df: pd.DataFrame, out_path: Path) -> None:
    fig = _scatter_by_technique(
        df, "latency_mean_ms", "classification_accuracy",
        "CPU Inference Latency (ms/sample, mean)", "Test Accuracy",
        "Accuracy vs. Inference Latency",
    )
    _save(fig, out_path)


def plot_size_vs_latency(df: pd.DataFrame, out_path: Path) -> None:
    fig = _scatter_by_technique(
        df, "model_size_mb", "latency_mean_ms",
        "Model Size (MB)", "CPU Inference Latency (ms/sample, mean)",
        "Model Size vs. Inference Latency",
    )
    _save(fig, out_path)


def plot_accuracy_vs_size(df: pd.DataFrame, out_path: Path) -> None:
    fig = _scatter_by_technique(
        df, "model_size_mb", "classification_accuracy",
        "Model Size (MB)", "Test Accuracy",
        "Accuracy vs. Model Size",
    )
    _save(fig, out_path)


def plot_compression_ratio_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of compression ratio per run, grouped by technique."""
    plot_df = df.dropna(subset=["compression_ratio"]).copy()
    if plot_df.empty:
        raise ValueError("No rows with a valid compression_ratio -- run aggregate.build_processed_results first.")

    plot_df["label"] = plot_df.apply(
        lambda r: f"{r['technique']}"
        + (f"\nsparsity={r['sparsity']}" if pd.notna(r.get("sparsity")) and r["sparsity"] not in (0, 0.0) else "")
        + (f"\n{r['quantization_method']}" if pd.notna(r.get("quantization_method")) and r["quantization_method"] != "fp32" else ""),
        axis=1,
    )
    plot_df = plot_df.sort_values("compression_ratio")

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(plot_df)), 4.5))
    colors = [TECHNIQUE_COLORS.get(t, "#888888") for t in plot_df["technique"]]
    ax.bar(plot_df["label"], plot_df["compression_ratio"], color=colors)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="No compression (baseline)")
    ax.set_ylabel("Compression Ratio")
    ax.set_title("Compression Ratio by Experiment")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    _save(fig, out_path)


def plot_pareto_frontier(df: pd.DataFrame, out_path: Path, efficiency_col: str = "model_size_mb") -> None:
    """Accuracy vs. efficiency (default: model size) with the Pareto-optimal
    points connected. A point is Pareto-optimal here if no other point has
    both lower `efficiency_col` (better) AND higher-or-equal accuracy.
    """
    plot_df = df.dropna(subset=[efficiency_col, "classification_accuracy"]).copy()
    if plot_df.empty:
        raise ValueError(f"No rows with valid {efficiency_col} and accuracy.")

    plot_df = plot_df.sort_values(efficiency_col)
    pareto_rows = []
    best_acc_so_far = -1.0
    # Walking from smallest to largest efficiency_col: a point is on the
    # frontier if its accuracy exceeds every point that was smaller/equal.
    for _, row in plot_df.sort_values(efficiency_col).iterrows():
        if row["classification_accuracy"] > best_acc_so_far:
            pareto_rows.append(row)
            best_acc_so_far = row["classification_accuracy"]
    pareto_df = pd.DataFrame(pareto_rows).sort_values(efficiency_col)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for technique, group in plot_df.groupby("technique"):
        ax.scatter(
            group[efficiency_col], group["classification_accuracy"],
            label=technique, color=TECHNIQUE_COLORS.get(technique, "#888888"),
            marker=TECHNIQUE_MARKERS.get(technique, "o"), s=60, alpha=0.6,
        )
    ax.plot(
        pareto_df[efficiency_col], pareto_df["classification_accuracy"],
        color="black", linewidth=1.5, linestyle="-", marker="D", markersize=6,
        label="Pareto frontier", zorder=5,
    )
    ax.set_xlabel(efficiency_col.replace("_", " ").title())
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Pareto Frontier: Accuracy vs. {efficiency_col.replace('_', ' ').title()}")
    ax.legend()
    _save(fig, out_path)


def generate_all_figures(processed_csv: str | Path, figures_dir: str | Path) -> list[Path]:
    df = pd.read_csv(processed_csv)
    figures_dir = Path(figures_dir)

    outputs = {
        "accuracy_vs_compression_ratio.png": plot_accuracy_vs_compression_ratio,
        "accuracy_vs_latency.png": plot_accuracy_vs_latency,
        "size_vs_latency.png": plot_size_vs_latency,
        "accuracy_vs_size.png": plot_accuracy_vs_size,
        "compression_ratio_comparison.png": plot_compression_ratio_comparison,
        "pareto_frontier_accuracy_vs_size.png": plot_pareto_frontier,
    }

    saved = []
    for filename, fn in outputs.items():
        out_path = figures_dir / filename
        try:
            fn(df, out_path)
            saved.append(out_path)
        except ValueError as exc:
            get_logger(__name__).warning(f"Skipped {filename}: {exc}")
    return saved
