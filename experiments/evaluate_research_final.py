import csv
import json
from pathlib import Path

import torch

from src.data.cifar10 import make_dataloaders
from src.evaluation.metrics import evaluate_classification
from src.utils.config import load_config
from src.utils.seed import set_seed

OUTPUT = Path("results/raw/research_pruning")
OUTPUT.mkdir(parents=True, exist_ok=True)

csv_path = OUTPUT / "final_test_results.csv"
json_path = OUTPUT / "final_test_results.json"

if csv_path.exists() or json_path.exists():
    raise SystemExit(
        "Final test results already exist. "
        "Refusing to overwrite the original evaluation."
    )

cfg = load_config("configs/config_mac.yaml")
set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
torch.set_num_threads(4)

_, _, test_loader, _ = make_dataloaders(
    cfg, download=False
)

results = []

print("===== FINAL HELD-OUT TEST EVALUATION =====", flush=True)

for ratio in [0, 10, 20, 30, 40]:
    model_path = Path(
        f"models/research_pruning/ratio_{ratio:02d}/full_model.pt"
    )

    # Load only our own locally generated model objects.
    model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    model.eval()

    metrics = evaluate_classification(
        model, test_loader, device="cpu"
    )

    result = {
        "requested_pruning_ratio": ratio / 100,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "test_samples": metrics.n_samples,
        "test_accuracy": metrics.accuracy,
        "test_precision_macro": metrics.precision_macro,
        "test_recall_macro": metrics.recall_macro,
        "test_f1_macro": metrics.f1_macro,
    }

    results.append(result)

    print(
        f"Ratio {ratio:02d}% | "
        f"Accuracy: {metrics.accuracy:.2%} | "
        f"Macro F1: {metrics.f1_macro:.4f} | "
        f"Samples: {metrics.n_samples}",
        flush=True,
    )

    del model

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

with json_path.open("w") as f:
    json.dump(results, f, indent=2)

print("\n===== EVALUATION COMPLETE =====")
print("CSV:", csv_path)
print("JSON:", json_path)
