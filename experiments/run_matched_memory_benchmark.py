import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch
import torch_pruning as tp

from src.models.resnet import build_resnet18_cifar
from src.utils.config import load_config
from src.utils.seed import set_seed

OUT = Path("results/raw/structured")
MIB = 1024 ** 2


def rss_mib():
    return psutil.Process().memory_info().rss / MIB


def load_model(name, cfg):
    model = build_resnet18_cifar(
        num_classes=cfg.data.num_classes,
        cifar_stem=cfg.model.cifar_stem,
    )

    if name == "unpruned_control":
        path = "models/baseline/resnet18_unpruned_control_3extra_epochs.pt"
        model.load_state_dict(torch.load(path, map_location="cpu"))

    elif name == "structured_pruned":
        baseline = "models/baseline/resnet18_fp32_best.pt"
        path = "models/pruning/resnet18_structured_physical_s0.2_best.pt"

        model.load_state_dict(torch.load(baseline, map_location="cpu"))
        model.eval()

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=torch.randn(1, 3, 32, 32),
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=0.2,
            iterative_steps=1,
            ignored_layers=[model.fc],
        )
        pruner.step()

        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        raise ValueError(name)

    model.eval()
    return model


def worker(name, repeat):
    cfg = load_config("configs/config_mac.yaml")
    set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
    torch.set_num_threads(4)

    gc.collect()
    before = rss_mib()

    model = load_model(name, cfg)
    gc.collect()
    after_load = rss_mib()

    x = torch.randn(1, 3, 32, 32)
    peak = rss_mib()

    with torch.inference_mode():
        for _ in range(20):
            model(x)
            peak = max(peak, rss_mib())

        for _ in range(100):
            model(x)
            peak = max(peak, rss_mib())

    result = {
        "model": name,
        "repeat": repeat,
        "rss_before_load_mib": before,
        "rss_after_load_mib": after_load,
        "rss_peak_sampled_mib": peak,
        "load_rss_delta_mib": after_load - before,
        "inference_peak_over_initial_mib": peak - before,
        "batch_size": 1,
        "cpu_threads": 4,
        "inference_runs": 100,
    }

    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=[
        "unpruned_control", "structured_pruned"
    ])
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    if args.worker:
        worker(args.worker, args.repeat)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for repeat in range(1, 6):
        names = ["unpruned_control", "structured_pruned"]
        if repeat % 2 == 0:
            names.reverse()

        for name in names:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.run_matched_memory_benchmark",
                    "--worker", name,
                    "--repeat", str(repeat),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            result = json.loads(proc.stdout.strip().splitlines()[-1])
            rows.append(result)

            print(
                f"Repeat {repeat} | {name} | "
                f"loaded={result['rss_after_load_mib']:.2f} MiB | "
                f"sampled_peak={result['rss_peak_sampled_mib']:.2f} MiB",
                flush=True,
            )

    raw = OUT / "matched_memory_benchmark_rounds.csv"

    with raw.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for name in ["unpruned_control", "structured_pruned"]:
        subset = [r for r in rows if r["model"] == name]

        result = {"model": name, "repeats": len(subset)}

        for metric in [
            "rss_before_load_mib",
            "rss_after_load_mib",
            "rss_peak_sampled_mib",
            "load_rss_delta_mib",
            "inference_peak_over_initial_mib",
        ]:
            values = [r[metric] for r in subset]
            result[f"{metric}_mean"] = sum(values) / len(values)

        summary.append(result)

    path = OUT / "matched_memory_benchmark_summary.csv"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("\n===== MEMORY BENCHMARK SUMMARY =====")
    for row in summary:
        print(row)

    print("\nSaved:", raw)
    print("Saved:", path)


if __name__ == "__main__":
    main()
