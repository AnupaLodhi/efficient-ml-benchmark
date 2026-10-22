import csv
import json
import platform
import statistics
import time
from pathlib import Path

import torch

torch.set_num_threads(4)

OUTPUT = Path("results/raw/research_pruning/cpu_benchmark.csv")
ROUNDS = 10
RUNS_PER_ROUND = 100
WARMUPS = 50

results = []

for ratio in [0, 10, 20, 30, 40]:
    path = Path(
        f"models/research_pruning/ratio_{ratio:02d}/full_model.pt"
    )

    model = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    model.eval()

    x = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        for _ in range(WARMUPS):
            model(x)

        round_latencies = []

        for _ in range(ROUNDS):
            start = time.perf_counter()

            for _ in range(RUNS_PER_ROUND):
                model(x)

            elapsed = time.perf_counter() - start

            round_latencies.append(
                elapsed * 1000 / RUNS_PER_ROUND
            )

    mean_ms = statistics.mean(round_latencies)
    std_ms = statistics.stdev(round_latencies)

    result = {
        "pruning_ratio": ratio,
        "parameters": sum(p.numel() for p in model.parameters()),
        "mean_latency_ms": mean_ms,
        "std_round_latency_ms": std_ms,
        "throughput_images_per_sec": 1000 / mean_ms,
        "batch_size": 1,
        "cpu_threads": 4,
        "warmups": WARMUPS,
        "rounds": ROUNDS,
        "inferences_per_round": RUNS_PER_ROUND,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "round_latencies_ms": json.dumps(round_latencies),
    }

    results.append(result)

    print(
        f"Ratio {ratio:02d}% | "
        f"Latency: {mean_ms:.3f} ± {std_ms:.3f} ms | "
        f"Throughput: {1000 / mean_ms:.2f} images/s",
        flush=True,
    )

    del model

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nBenchmark complete:", OUTPUT)
