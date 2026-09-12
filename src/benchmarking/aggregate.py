"""Aggregate raw per-experiment rows (results/raw/results.csv) into a
processed table (results/processed/results_processed.csv) with
compression_ratio filled in relative to the FP32 baseline.

Why compression_ratio is computed here rather than by each experiment
script: every technique's `full_evaluation_report` measures model_size_mb
independently and correctly, but "compression ratio" is inherently relative
to a reference model. Computing it once, centrally, from whichever baseline
row is actually present in results.csv (rather than duplicating a
hard-coded baseline size into every experiment script) means the ratio is
always computed against the real, current baseline measurement -- if the
baseline is ever re-run, every downstream ratio updates consistently
instead of silently going stale.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_compression_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add/overwrite `compression_ratio` = baseline_size_mb / row_size_mb.

    The baseline reference is the row with technique=="baseline". If no
    baseline row is present, compression_ratio is left as-is (NaN) rather
    than guessed -- a missing baseline is a real gap that should be visible,
    not papered over with an assumed number.
    """
    df = df.copy()
    baseline_rows = df[df["technique"] == "baseline"]

    if baseline_rows.empty:
        logger.warning(
            "No 'baseline' row found in results.csv -- compression_ratio "
            "cannot be computed and will be left blank. Run "
            "experiments/train_baseline.py first."
        )
        return df

    if len(baseline_rows) > 1:
        logger.warning(
            f"{len(baseline_rows)} baseline rows found; using the most recent "
            "by timestamp_utc as the reference for compression_ratio."
        )
        baseline_size = baseline_rows.sort_values("timestamp_utc").iloc[-1]["model_size_mb"]
    else:
        baseline_size = baseline_rows.iloc[0]["model_size_mb"]

    df["compression_ratio"] = baseline_size / df["model_size_mb"]
    return df


def build_processed_results(raw_csv: str | Path, processed_csv: str | Path) -> pd.DataFrame:
    raw_csv = Path(raw_csv)
    processed_csv = Path(processed_csv)

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run at least experiments/train_baseline.py "
            "before building processed results."
        )

    df = pd.read_csv(raw_csv)
    if df.empty:
        raise ValueError(f"{raw_csv} exists but has no rows.")

    df = compute_compression_ratios(df)

    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_csv, index=False)
    logger.info(f"Processed results ({len(df)} rows) -> {processed_csv}")
    return df
