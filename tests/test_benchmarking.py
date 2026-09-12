import pandas as pd
import pytest

from src.benchmarking.aggregate import build_processed_results, compute_compression_ratios
from src.benchmarking.results_store import RESULTS_SCHEMA_COLUMNS
from src.benchmarking.visualize import generate_all_figures


def _fake_rows():
    return [
        {
            "run_id": "baseline_test", "timestamp_utc": "2026-01-01T00:00:00",
            "model_name": "resnet18", "technique": "baseline", "sparsity": 0.0,
            "pruning_method": None, "quantization_method": "fp32", "device": "cpu",
            "classification_accuracy": 0.93, "classification_precision_macro": 0.93,
            "classification_recall_macro": 0.93, "classification_f1_macro": 0.93,
            "total_params": 11173962, "nonzero_params": 11173962, "model_size_mb": 42.7,
            "compression_ratio": None, "latency_mean_ms": 12.0, "latency_std_ms": 0.5,
            "latency_throughput_samples_per_sec": 83.3, "memory_peak_rss_delta_mb": 5.0,
            "training_time_sec": 3600, "notes": "unit-test fixture, not a real result",
        },
        {
            "run_id": "quant_test", "timestamp_utc": "2026-01-01T01:00:00",
            "model_name": "resnet18", "technique": "quantization", "sparsity": 0.0,
            "pruning_method": None, "quantization_method": "static_int8", "device": "cpu",
            "classification_accuracy": 0.92, "classification_precision_macro": 0.92,
            "classification_recall_macro": 0.92, "classification_f1_macro": 0.92,
            "total_params": 11173962, "nonzero_params": 11173962, "model_size_mb": 11.2,
            "compression_ratio": None, "latency_mean_ms": 4.5, "latency_std_ms": 0.3,
            "latency_throughput_samples_per_sec": 222.0, "memory_peak_rss_delta_mb": 3.0,
            "training_time_sec": 0, "notes": "unit-test fixture, not a real result",
        },
    ]


def test_compute_compression_ratios_uses_baseline_reference():
    df = pd.DataFrame(_fake_rows(), columns=RESULTS_SCHEMA_COLUMNS)
    result = compute_compression_ratios(df)
    baseline_ratio = result.loc[result["technique"] == "baseline", "compression_ratio"].iloc[0]
    quant_ratio = result.loc[result["technique"] == "quantization", "compression_ratio"].iloc[0]
    assert baseline_ratio == 1.0
    assert abs(quant_ratio - (42.7 / 11.2)) < 1e-6


def test_compute_compression_ratios_missing_baseline_leaves_nan():
    rows = [r for r in _fake_rows() if r["technique"] != "baseline"]
    df = pd.DataFrame(rows, columns=RESULTS_SCHEMA_COLUMNS)
    result = compute_compression_ratios(df)
    assert result["compression_ratio"].isna().all()


def test_build_processed_results_end_to_end(tmp_path):
    raw_csv = tmp_path / "results.csv"
    processed_csv = tmp_path / "results_processed.csv"
    df = pd.DataFrame(_fake_rows(), columns=RESULTS_SCHEMA_COLUMNS)
    df.to_csv(raw_csv, index=False)

    result = build_processed_results(raw_csv, processed_csv)
    assert processed_csv.exists()
    assert len(result) == 2
    assert "compression_ratio" in result.columns


def test_build_processed_results_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_processed_results(tmp_path / "does_not_exist.csv", tmp_path / "out.csv")


def test_generate_all_figures_produces_all_six(tmp_path):
    raw_csv = tmp_path / "results.csv"
    processed_csv = tmp_path / "results_processed.csv"
    figures_dir = tmp_path / "figures"

    df = pd.DataFrame(_fake_rows(), columns=RESULTS_SCHEMA_COLUMNS)
    df.to_csv(raw_csv, index=False)
    build_processed_results(raw_csv, processed_csv)

    figure_paths = generate_all_figures(processed_csv, figures_dir)
    assert len(figure_paths) == 6
    for p in figure_paths:
        assert p.exists()
        assert p.stat().st_size > 0
