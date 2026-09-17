"""Phase 5 entry point: combined pruning + quantization.

Experimental design (explained before implementation, per project spec):

Full cross product would be 2 pruning methods x 5 sparsities x 4 quantization
methods = 40 combined runs, on top of the 10 pruning-only and 4
quantization-only runs already covering each technique independently. That's
excessive for what RQ3 actually asks ("does combining beat either alone?")
and most of the grid would be uninformative: e.g. combining a barely-pruned
model (20% sparsity) with FP16 tells us little beyond what each techique
already showed independently.

Restricted grid used here (configs/config.yaml `combined:` section):
  - **Pruning**: only `unstructured_l1`, only at sparsities [0.4, 0.6].
    Unstructured is used because it's this project's better-understood,
    fully-implemented pruning path; 0.4/0.6 are the middle of the sparsity
    sweep -- high enough to be a meaningful compression claim, not so high
    (0.8) that accuracy is likely already destroyed on its own (that
    interaction is exactly what RQ3 wants to see once real numbers exist).
    `structured_ln` is excluded from the combined grid: it only masks
    channels in this project's implementation (see pruning.py docstring)
    and combining two techniques that both only "mask" would double-report
    a size/latency benefit that neither alone actually provides.
  - **Quantization**: `dynamic_int8` and `static_int8` (not `fp16`, not
    `fp32_baseline`). `fp32_baseline` combined with pruning is just the
    pruning-only experiment already covered in Phase 3. `fp16` is excluded
    from the combined grid because pruning's zeroed weights still occupy a
    full fp16 slot each -- there's no interaction with sparsity to observe
    that isn't already visible from the fp16-alone and pruning-alone
    results separately.
  - Result: 2 sparsities x 2 quantization methods = **4 combined runs**.

Order of operations: prune -> fine-tune -> quantize. Quantization is applied
last (closest to deployment), consistent with post-training quantization's
usual place at the end of a compression pipeline; fine-tuning happens on the
pruned-but-not-yet-quantized model, since PyTorch's quantized modules aren't
trainable with the standard optimizer used elsewhere in this project.

Honesty note carried over from Phases 3-4: unstructured pruning's "sparsity"
is a masking-based, dense-storage sparsity (see pruning.py). Combining it
with quantization does NOT provide additional size/latency benefit beyond
what quantization alone provides -- an int8 zero and an int8 non-zero value
occupy the same storage in a dense int8 tensor. Any accuracy difference
between "quantization alone" and "pruning + quantization" at the same
quantization method is the only thing this experiment can actually speak to;
results/notes for every combined row says this explicitly, not just here.

Usage:
    python -m experiments.run_combined --checkpoint models/baseline/resnet18_fp32_best.pt
    python -m experiments.run_combined --smoke-test
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.ao.nn.quantized.dynamic as nnqd

from src.benchmarking.results_store import append_result_row
from src.compression.pruning import apply_unstructured_pruning, finalize_pruning, get_sparsity_report
from src.compression.quantization import apply_dynamic_quantization, apply_static_quantization
from src.evaluation.metrics import full_evaluation_report
from src.models.resnet import build_resnet18_cifar
from src.training.train import train_model
from src.utils.config import load_config
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def load_baseline_model(checkpoint_path: str | Path, cfg) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {checkpoint_path}. "
            "Run `python -m experiments.train_baseline` first (Phase 2)."
        )
    model = build_resnet18_cifar(num_classes=cfg.data.num_classes, cifar_stem=cfg.model.cifar_stem)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model


def run_combined_sweep(cfg, checkpoint_path: str | Path) -> None:
    from src.data.cifar10 import make_dataloaders

    if not Path(checkpoint_path).exists():
        load_baseline_model(checkpoint_path, cfg)  # raises the clear FileNotFoundError

    train_loader, val_loader, test_loader, _ = make_dataloaders(cfg, download=True)

    for sparsity in cfg.combined.sparsities:
        set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
        logger.info(f"=== Combined: reusing corrected pruning checkpoint sparsity={sparsity} ===")

        pruning_ckpt = (
            Path(cfg.paths.models_dir)
            / "pruning"
            / f"resnet18_unstructured_l1_s{sparsity}.pt"
        )
        if not pruning_ckpt.exists():
            raise FileNotFoundError(
                f"Corrected pruning checkpoint not found at {pruning_ckpt}. "
                "Run the Phase 3 pruning sweep first."
            )

        pruned_model = load_baseline_model(checkpoint_path, cfg)
        apply_unstructured_pruning(
            pruned_model,
            sparsity,
            remove_reparam=False,
        )
        pruned_model.load_state_dict(
            torch.load(pruning_ckpt, map_location="cpu")
        )

        masked_report = get_sparsity_report(pruned_model)
        if abs(masked_report["overall_sparsity"] - sparsity) >= 0.01:
            raise RuntimeError(
                f"Loaded pruning checkpoint has unexpected sparsity: "
                f"target={sparsity:.4f}, "
                f"actual={masked_report['overall_sparsity']:.4f}"
            )

        finalize_pruning(pruned_model)
        sparsity_report = get_sparsity_report(pruned_model)

        if abs(sparsity_report["overall_sparsity"] - sparsity) >= 0.01:
            raise RuntimeError(
                f"Finalized checkpoint has unexpected sparsity: "
                f"target={sparsity:.4f}, "
                f"actual={sparsity_report['overall_sparsity']:.4f}"
            )

        logger.info(
            f"Corrected checkpoint verified and finalized: "
            f"target={sparsity:.3f}, "
            f"actual={sparsity_report['overall_sparsity']:.3f}"
        )

        for q_method in cfg.combined.quantization_methods:
            logger.info(f"=== Combined: sparsity={sparsity} + {q_method} ===")

            if q_method == "dynamic_int8":
                combined_model = apply_dynamic_quantization(pruned_model)
                extra_note = "dynamic_int8 only quantizes the fc layer (see quantization.py) -- pruning of Conv2d layers is the only part of this combination affecting the backbone."
            elif q_method == "static_int8":
                combined_model = apply_static_quantization(
                    pruned_model, train_loader, num_calibration_batches=cfg.quantization.static_calibration_batches
                )
                extra_note = "static_int8 quantizes the full backbone; combined with pruning's masking, size reduction here is attributable to quantization, not the pruning step -- see module docstring."
            else:
                raise ValueError(f"Unsupported combined quantization method: {q_method}")

            report = full_evaluation_report(
                combined_model, test_loader,
                num_warmup=cfg.evaluation.latency_num_warmup,
                num_latency_runs=cfg.evaluation.latency_num_runs,
                device="cpu",
            )

            run_id = f"combined_s{sparsity}_{q_method}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            row = {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_name": "resnet18",
                "technique": "combined",
                "sparsity": sparsity,
                "pruning_method": "unstructured_l1",
                "quantization_method": q_method,
                "device": "cpu",
                "training_time_sec": 0.0,
                "notes": (
                    f"Reused corrected Phase 3 pruning checkpoint at "
                    f"sparsity={sparsity_report['overall_sparsity']:.4f}; "
                    f"no additional fine-tuning in combined phase; "
                    f"then applied {q_method} quantization. {extra_note}"
                ),
                **report,
            }
            results_csv = Path(cfg.paths.results_raw) / "results.csv"
            append_result_row(row, results_csv)
            logger.info(f"Result appended -> {results_csv}")


def run_smoke_test(cfg) -> None:
    """Verify the full prune -> fine-tune -> quantize pipeline runs
    end-to-end for both combined quantization methods, on synthetic data.
    Does not write to results.csv."""
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class _Tiny(Dataset):
        def __init__(self, n, seed):
            rng = np.random.default_rng(seed)
            self.images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
            self.labels = rng.integers(0, 10, size=(n,)).tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            tf = transforms.Compose([transforms.ToTensor()])
            return tf(Image.fromarray(self.images[idx])), self.labels[idx]

    train_loader = DataLoader(_Tiny(32, 0), batch_size=8)
    val_loader = DataLoader(_Tiny(16, 1), batch_size=8)

    model = build_resnet18_cifar(num_classes=cfg.data.num_classes, cifar_stem=cfg.model.cifar_stem)
    apply_unstructured_pruning(model, 0.4, remove_reparam=False)

    before = get_sparsity_report(model)["overall_sparsity"]
    assert abs(before - 0.4) < 0.01

    history = train_model(model, train_loader, val_loader, cfg, epochs=1, device="cpu")

    during = get_sparsity_report(model)["overall_sparsity"]
    assert abs(during - 0.4) < 0.01

    finalize_pruning(model)

    after = get_sparsity_report(model)["overall_sparsity"]
    assert abs(after - 0.4) < 0.01

    logger.info(
        f"SMOKE TEST: sparsity before={before:.3f}, "
        f"after fine-tuning={during:.3f}, finalized={after:.3f}"
    )
    logger.info(f"SMOKE TEST: pruned + 1-epoch fine-tune done, train_loss={history.train_loss[0]:.4f}")

    mdyn = apply_dynamic_quantization(model)
    x = torch.randn(2, 3, 32, 32)
    y = mdyn(x)
    assert y.shape == (2, 10)
    logger.info("SMOKE TEST [pruned + dynamic_int8]: forward pass OK")

    qstatic = apply_static_quantization(model, train_loader, num_calibration_batches=2)
    y2 = qstatic(x)
    assert y2.shape == (2, 10)
    logger.info("SMOKE TEST [pruned + static_int8]: forward pass OK")

    logger.info("SMOKE TEST complete: combined prune -> fine-tune -> quantize pipeline runs end-to-end.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined pruning + quantization sweep.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/baseline/resnet18_fp32_best.pt")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.smoke_test:
        set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)
        run_smoke_test(cfg)
        return

    run_combined_sweep(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
