import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/raw/research_pruning")
OUT = Path("results/figures/research_pruning")
OUT.mkdir(parents=True, exist_ok=True)

ratios = [0, 10, 20, 30, 40]

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

test = {
    round(float(r["requested_pruning_ratio"]) * 100): r
    for r in read_csv(ROOT / "final_test_results.csv")
}

memory = {
    int(r["ratio"]): r
    for r in read_csv(ROOT / "memory_benchmark.csv")
}

bench = read_csv(ROOT / "cpu_benchmark_repeated.csv")

latency = {
    ratio: statistics.median(
        float(r["median_ms"])
        for r in bench
        if int(r["ratio"]) == ratio
    )
    for ratio in ratios
}

params = {
    ratio: int(test[ratio]["parameters"])
    for ratio in ratios
}

baseline_params = params[0]

reduction = [
    100 * (1 - params[r] / baseline_params)
    for r in ratios
]

accuracy = [
    100 * float(test[r]["test_accuracy"])
    for r in ratios
]

latencies = [latency[r] for r in ratios]

rss = [
    float(memory[r]["rss_after_mib"])
    for r in ratios
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def save(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close()
    print("Saved:", name)

# Figure 1: Accuracy versus actual compression
plt.figure(figsize=(6, 4))
plt.plot(reduction, accuracy, marker="o", linewidth=2)
plt.xlabel("Actual parameter reduction (%)")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy versus physical parameter reduction")
plt.grid(alpha=0.25)
save("accuracy_vs_compression")

# Figure 2: CPU latency
plt.figure(figsize=(6, 4))
plt.plot(ratios, latencies, marker="s", linewidth=2)
plt.xlabel("Requested pruning ratio (%)")
plt.ylabel("Median CPU latency (ms)")
plt.title("CPU inference latency")
plt.xticks(ratios)
plt.grid(alpha=0.25)
save("cpu_latency")

# Figure 3: Resident memory
plt.figure(figsize=(6, 4))
plt.plot(ratios, rss, marker="^", linewidth=2)
plt.xlabel("Requested pruning ratio (%)")
plt.ylabel("Process RSS after inference (MiB)")
plt.title("Resident memory footprint")
plt.xticks(ratios)
plt.grid(alpha=0.25)
save("resident_memory")

# Figure 4: Accuracy-latency tradeoff
plt.figure(figsize=(6, 4))
plt.plot(latencies, accuracy, marker="o", linewidth=2)

for ratio, x, y in zip(ratios, latencies, accuracy):
    plt.annotate(
        f"{ratio}%",
        (x, y),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.xlabel("Median CPU latency (ms)")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy-latency tradeoff")
plt.grid(alpha=0.25)
save("accuracy_latency_tradeoff")

print("\nAll four figures generated as PDF and PNG.")
