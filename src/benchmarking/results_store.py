"""Single results store used by every experiment script (baseline, pruning,
quantization, combined) so everything ends up in one master table with a
consistent schema -- the "single reproducible benchmarking framework"
requirement.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_SCHEMA_COLUMNS = [
    "run_id", "timestamp_utc", "model_name", "technique", "sparsity",
    "pruning_method", "quantization_method", "device",
    "classification_accuracy", "classification_precision_macro",
    "classification_recall_macro", "classification_f1_macro",
    "total_params", "nonzero_params", "model_size_mb", "compression_ratio",
    "latency_mean_ms", "latency_std_ms", "latency_throughput_samples_per_sec",
    "memory_peak_rss_delta_mb", "training_time_sec", "notes",
]


def append_result_row(row: dict[str, Any], csv_path: str | Path) -> None:
    """Append one experiment's results as a row to the master CSV.

    Any column not present in `row` is filled with None so the schema stays
    consistent even as different experiment types report slightly different
    fields (e.g. `pruning_method` is only meaningful for pruning runs).
    Also mirrors every row to a parallel .jsonl file (one JSON object per
    line) so raw results survive even if the CSV becomes malformed.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    full_row = {col: row.get(col) for col in RESULTS_SCHEMA_COLUMNS}
    # keep any extra fields the caller passed, appended after the fixed schema
    for k, v in row.items():
        if k not in full_row:
            full_row[k] = v

    df_row = pd.DataFrame([full_row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)

    jsonl_path = csv_path.with_suffix(".jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(full_row, default=str) + "\n")


def load_results(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=RESULTS_SCHEMA_COLUMNS)
    return pd.read_csv(csv_path)
