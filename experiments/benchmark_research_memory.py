import csv
import subprocess
import sys
from pathlib import Path

OUTPUT = Path("results/raw/research_pruning/memory_benchmark.csv")

worker = r'''
import json
import resource
import torch
import psutil
import os
import sys

torch.set_num_threads(4)
ratio = int(sys.argv[1])
path = f"models/research_pruning/ratio_{ratio:02d}/full_model.pt"

process = psutil.Process(os.getpid())
before = process.memory_info().rss

model = torch.load(path, map_location="cpu", weights_only=False)
model.eval()

with torch.inference_mode():
    model(torch.randn(1, 3, 32, 32))

after = process.memory_info().rss

print(json.dumps({
    "ratio": ratio,
    "rss_before_mib": before / 1024**2,
    "rss_after_mib": after / 1024**2,
    "rss_delta_mib": (after - before) / 1024**2,
}))
'''

results = []

for ratio in [0, 10, 20, 30, 40]:
    result = subprocess.run(
        [sys.executable, "-c", worker, str(ratio)],
        capture_output=True,
        text=True,
        check=True,
    )

    import json
    data = json.loads(result.stdout)
    results.append(data)

    print(
        f"Ratio {ratio:02d}% | "
        f"RSS after load+inference: {data['rss_after_mib']:.2f} MiB | "
        f"Delta: {data['rss_delta_mib']:.2f} MiB",
        flush=True,
    )

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nMemory benchmark complete:", OUTPUT)
