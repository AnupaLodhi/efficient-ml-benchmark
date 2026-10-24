import csv
from pathlib import Path

import torch
import torch_pruning as tp

torch.set_num_threads(4)

output = Path("results/raw/research_pruning/compute_profile.csv")
output.parent.mkdir(parents=True, exist_ok=True)

dummy_input = torch.randn(1, 3, 32, 32)
rows = []

for ratio in [0, 10, 20, 30, 40]:
    path = Path(
        f"models/research_pruning/ratio_{ratio:02d}/full_model.pt"
    )

    # Load only locally generated, trusted checkpoints.
    model = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(model, dict):
        model = model.get("model", model)

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Invalid model checkpoint: {path}")

    model.eval()

    with torch.no_grad():
        macs, params = tp.utils.count_ops_and_params(
            model, dummy_input
        )

    rows.append({
        "requested_pruning_ratio": ratio,
        "macs": int(macs),
        "macs_millions": round(macs / 1e6, 4),
        "parameters": int(params),
        "parameters_millions": round(params / 1e6, 4),
    })

baseline_macs = rows[0]["macs"]
baseline_params = rows[0]["parameters"]

for row in rows:
    row["mac_reduction_pct"] = round(
        100 * (1 - row["macs"] / baseline_macs), 2
    )
    row["parameter_reduction_pct"] = round(
        100 * (1 - row["parameters"] / baseline_params), 2
    )

    print(
        f"{row['requested_pruning_ratio']:2d}% | "
        f"{row['macs_millions']:.2f} M MACs | "
        f"{row['mac_reduction_pct']:.2f}% MAC reduction"
    )

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved: {output}")
