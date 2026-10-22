import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

OUTPUT = Path(
    "results/raw/research_pruning/cpu_benchmark_repeated.csv"
)

if OUTPUT.exists():
    raise SystemExit(
        "Repeated benchmark already exists; refusing to overwrite."
    )

worker = r'''
import json
import statistics
import sys
import time
import torch

torch.set_num_threads(4)

ratio = int(sys.argv[1])
path = (
    f"models/research_pruning/ratio_{ratio:02d}/full_model.pt"
)

model = torch.load(
    path, map_location="cpu", weights_only=False
)
model.eval()

x = torch.randn(1, 3, 32, 32)

with torch.inference_mode():
    for _ in range(30):
        model(x)

    rounds = []

    for _ in range(5):
        start = time.perf_counter()

        for _ in range(100):
            model(x)

        elapsed = time.perf_counter() - start
        rounds.append(elapsed * 1000 / 100)

print(json.dumps({
    "ratio": ratio,
    "round_latencies_ms": rounds,
    "median_ms": statistics.median(rounds),
    "mean_ms": statistics.mean(rounds),
}))
'''

rows = []

# Rotating order reduces systematic order effects.
for session in range(5):
    ratios = [0, 10, 20, 30, 40]
    shift = session % len(ratios)
    ratios = ratios[shift:] + ratios[:shift]

    print(f"\n===== SESSION {session + 1}/5 =====", flush=True)

    for ratio in ratios:
        completed = subprocess.run(
            [sys.executable, "-c", worker, str(ratio)],
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(completed.stdout)

        row = {
            "session": session + 1,
            "ratio": ratio,
            "median_ms": data["median_ms"],
            "mean_ms": data["mean_ms"],
            "round_latencies_ms": json.dumps(
                data["round_latencies_ms"]
            ),
        }

        rows.append(row)

        print(
            f"Ratio {ratio:02d}% | "
            f"Median: {data['median_ms']:.3f} ms | "
            f"Mean: {data['mean_ms']:.3f} ms",
            flush=True,
        )

# Save all completed measurements.
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n===== REPEATED BENCHMARK SUMMARY =====")

for ratio in [0, 10, 20, 30, 40]:
    values = [
        row["median_ms"]
        for row in rows
        if row["ratio"] == ratio
    ]

    print(
        f"Ratio {ratio:02d}% | "
        f"Median across sessions: "
        f"{statistics.median(values):.3f} ms | "
        f"Session range: "
        f"{min(values):.3f}–{max(values):.3f} ms"
    )

print("\nSaved:", OUTPUT)
