import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch_pruning as tp

from src.models.resnet import build_resnet18_cifar
from src.evaluation.metrics import measure_latency
from src.utils.config import load_config
from src.utils.seed import set_seed


def build_model(cfg):
    return build_resnet18_cifar(
        num_classes=cfg.data.num_classes,
        cifar_stem=cfg.model.cifar_stem,
    )


def load_models(cfg):
    baseline_checkpoint = (
        "models/baseline/resnet18_fp32_best.pt"
    )
    control_checkpoint = (
        "models/baseline/resnet18_unpruned_control_3extra_epochs.pt"
    )
    pruned_checkpoint = (
        "models/pruning/resnet18_structured_physical_s0.2_best.pt"
    )

    control = build_model(cfg)
    control.load_state_dict(
        torch.load(control_checkpoint, map_location="cpu")
    )
    control.eval()

    pruned = build_model(cfg)
    pruned.load_state_dict(
        torch.load(baseline_checkpoint, map_location="cpu")
    )
    pruned.eval()

    pruner = tp.pruner.MagnitudePruner(
        pruned,
        example_inputs=torch.randn(1, 3, 32, 32),
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=0.2,
        iterative_steps=1,
        ignored_layers=[pruned.fc],
    )
    pruner.step()

    pruned.load_state_dict(
        torch.load(pruned_checkpoint, map_location="cpu")
    )
    pruned.eval()

    return {
        "unpruned_control": control,
        "structured_pruned": pruned,
    }


def main():
    cfg = load_config("configs/config_mac.yaml")
    set_seed(
        cfg.project.seed,
        deterministic=cfg.project.deterministic,
    )
    torch.set_num_threads(4)

    models = load_models(cfg)

    output_dir = Path("results/raw/structured")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for round_number in range(1, 6):
        names = list(models)

        if round_number % 2 == 0:
            names.reverse()

        for name in names:
            result = measure_latency(
                models[name],
                input_shape=(1, 3, 32, 32),
                num_warmup=20,
                num_runs=100,
                device="cpu",
            )

            row = {
                "round": round_number,
                "model": name,
                "latency_mean_ms": result.mean_ms,
                "latency_std_ms": result.std_ms,
                "latency_min_ms": result.min_ms,
                "latency_max_ms": result.max_ms,
                "throughput_images_per_sec":
                    result.throughput_samples_per_sec,
                "batch_size": result.batch_size,
                "num_runs": result.num_runs,
                "cpu_threads": 4,
            }

            rows.append(row)

            print(
                f"Round {round_number} | {name} | "
                f"{result.mean_ms:.3f} ms | "
                f"{result.throughput_samples_per_sec:.2f} images/s",
                flush=True,
            )

    raw_path = output_dir / "matched_cpu_benchmark_rounds.csv"

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = []

    for name in models:
        subset = [r for r in rows if r["model"] == name]
        latencies = np.array(
            [r["latency_mean_ms"] for r in subset]
        )
        throughputs = np.array(
            [r["throughput_images_per_sec"] for r in subset]
        )

        summary.append({
            "model": name,
            "rounds": len(subset),
            "latency_mean_ms": float(latencies.mean()),
            "latency_std_between_rounds_ms":
                float(latencies.std(ddof=1)),
            "throughput_mean_images_per_sec":
                float(throughputs.mean()),
            "checkpoint": (
                "models/baseline/"
                "resnet18_unpruned_control_3extra_epochs.pt"
                if name == "unpruned_control"
                else "models/pruning/"
                "resnet18_structured_physical_s0.2_best.pt"
            ),
        })

    summary_path = (
        output_dir / "matched_cpu_benchmark_summary.csv"
    )

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=summary[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n===== MATCHED CPU BENCHMARK =====")

    for item in summary:
        print(
            f"{item['model']}: "
            f"{item['latency_mean_ms']:.3f} ms, "
            f"{item['throughput_mean_images_per_sec']:.2f} images/s"
        )

    print("\nSaved:", raw_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
