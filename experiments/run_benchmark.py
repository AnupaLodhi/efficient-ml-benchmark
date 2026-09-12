"""Phase 6 entry point: aggregate every experiment's results and generate
the required publication figures + summary tables.

Usage:
    python -m experiments.run_benchmark

Requires results/raw/results.csv to already contain rows -- i.e. at least
experiments/train_baseline.py must have been run. This script does not run
any new experiments itself; it only aggregates and visualizes what's
already in results/raw/results.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.benchmarking.aggregate import build_processed_results
from src.benchmarking.visualize import generate_all_figures
from src.utils.config import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    """A compact per-run summary table (one row per experiment) formatted
    for direct inclusion in the paper's Results section -- same numbers as
    results_processed.csv, just the subset of columns a reader actually
    needs and rounded for readability.
    """
    cols = [
        "run_id", "technique", "pruning_method", "sparsity", "quantization_method",
        "classification_accuracy", "classification_f1_macro", "model_size_mb",
        "compression_ratio", "latency_mean_ms", "latency_std_ms",
    ]
    available = [c for c in cols if c in df.columns]
    summary = df[available].copy()
    for col in summary.select_dtypes(include="float").columns:
        summary[col] = summary[col].round(4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info(f"Summary table ({len(summary)} rows) -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results and generate all benchmark figures/tables.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_csv = Path(cfg.paths.results_raw) / "results.csv"
    processed_csv = Path(cfg.paths.results_processed) / "results_processed.csv"
    figures_dir = Path(cfg.paths.results_figures)
    tables_dir = Path(cfg.paths.results_tables)

    logger.info(f"Aggregating {raw_csv} -> {processed_csv} ...")
    df = build_processed_results(raw_csv, processed_csv)
    logger.info(f"{len(df)} total experiment rows across techniques: {df['technique'].value_counts().to_dict()}")

    logger.info("Generating figures ...")
    figure_paths = generate_all_figures(processed_csv, figures_dir)
    for p in figure_paths:
        logger.info(f"  -> {p}")

    build_summary_table(df, tables_dir / "results_summary.csv")

    logger.info(
        "Phase 6 complete. NOTE: figures/tables reflect exactly what is in "
        "results/raw/results.csv -- if that file is empty or only has the "
        "baseline row, most figures will be sparse or skipped (see "
        "src/benchmarking/visualize.py docstring: no fabricated points)."
    )


if __name__ == "__main__":
    main()
