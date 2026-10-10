from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results")
STRUCT = ROOT / "raw" / "structured"
OUT = ROOT / "publication"
OUT.mkdir(parents=True, exist_ok=True)

comparison = pd.read_csv(STRUCT / "matched_model_comparison.csv")
cpu = pd.read_csv(STRUCT / "matched_cpu_benchmark_rounds.csv")
memory = pd.read_csv(STRUCT / "matched_memory_benchmark_rounds.csv")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "axes.grid": False,
})

COLORS = ["#355C7D", "#2A9D8F"]
LABELS = ["Unpruned control", "Structured pruning"]

def save(name):
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close()
    print("Saved:", name)

# Figure 1: accuracy and storage
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
metrics = [
    ("validation_accuracy", "Validation accuracy (%)", 100),
    ("checkpoint_mib", "Checkpoint size (MiB)", 1),
]
for ax, (column, ylabel, multiplier) in zip(axes, metrics):
    values = comparison[column].to_numpy() * multiplier
    bars = ax.bar(LABELS, values, color=COLORS, width=0.55)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis="x", labelrotation=12)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(values)*0.025,
            f"{value:.2f}",
            ha="center",
            fontsize=9,
        )
save("fig01_accuracy_storage")

# Figure 2: paired CPU latency across rounds
pivot = cpu.pivot(index="round", columns="model", values="latency_mean_ms")
fig, ax = plt.subplots(figsize=(6.7, 3.6))
for name, label, color, marker in [
    ("unpruned_control", LABELS[0], COLORS[0], "o"),
    ("structured_pruned", LABELS[1], COLORS[1], "s"),
]:
    ax.plot(
        pivot.index, pivot[name],
        marker=marker, linewidth=1.8,
        color=color, label=label,
    )
ax.set_xlabel("Benchmark round")
ax.set_ylabel("Mean inference latency (ms)")
ax.set_xticks(pivot.index)
ax.legend(frameon=False)
save("fig02_cpu_latency_rounds")

# Figure 3: CPU throughput with between-round variability
fig, ax = plt.subplots(figsize=(5.8, 3.5))
groups = [
    cpu.loc[cpu.model == "unpruned_control",
            "throughput_images_per_sec"].to_numpy(),
    cpu.loc[cpu.model == "structured_pruned",
            "throughput_images_per_sec"].to_numpy(),
]
means = [a.mean() for a in groups]
stds = [a.std(ddof=1) for a in groups]
ax.bar(LABELS, means, yerr=stds, capsize=5, color=COLORS, width=0.55)
ax.set_ylabel("Throughput (images/s)")
ax.set_ylim(0, max(means) * 1.25)
ax.tick_params(axis="x", labelrotation=10)
save("fig03_cpu_throughput")

# Figure 4: loaded process RSS with variability
fig, ax = plt.subplots(figsize=(5.8, 3.5))
groups = [
    memory.loc[memory.model == "unpruned_control",
               "rss_after_load_mib"].to_numpy(),
    memory.loc[memory.model == "structured_pruned",
               "rss_after_load_mib"].to_numpy(),
]
means = [a.mean() for a in groups]
stds = [a.std(ddof=1) for a in groups]
ax.bar(LABELS, means, yerr=stds, capsize=5, color=COLORS, width=0.55)
ax.set_ylabel("Process RSS after loading (MiB)")
ax.set_ylim(0, max(means) * 1.18)
ax.tick_params(axis="x", labelrotation=10)
save("fig04_process_memory")

print("\nPublication figures generated in:", OUT)
