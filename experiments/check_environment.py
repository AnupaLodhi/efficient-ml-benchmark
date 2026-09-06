"""Phase 1 entry point: record environment/hardware info and sanity-check
that the model and data pipeline are wired up correctly.

Usage:
    python -m experiments.check_environment
    python -m experiments.check_environment --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.resnet import build_resnet18_cifar, count_parameters
from src.utils.config import load_config
from src.utils.env_info import collect_environment_info
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record environment info and sanity-check the pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)

    logger.info("Collecting environment/hardware info ...")
    env_info = collect_environment_info()

    out_path = Path(cfg.paths.results_raw) / "environment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(env_info, f, indent=2)
    logger.info(f"Saved environment info -> {out_path}")

    logger.info(f"CPU: {env_info['cpu'].get('model_name', env_info['cpu']['processor'])}")
    logger.info(f"Logical cores: {env_info['cpu']['logical_cores']}")
    logger.info(f"CUDA available: {env_info['gpu']['cuda_available']}")
    logger.info(f"PyTorch: {env_info['torch_version']} | Python: {env_info['python_version'].split()[0]}")

    logger.info("Building ResNet-18 (CIFAR stem) to sanity-check the model definition ...")
    model = build_resnet18_cifar(
        num_classes=cfg.data.num_classes,
        cifar_stem=cfg.model.cifar_stem,
    )
    params = count_parameters(model)
    logger.info(f"Model parameters: {params['total_params']:,} total, {params['trainable_params']:,} trainable")

    import torch

    dummy = torch.randn(4, 3, cfg.data.image_size, cfg.data.image_size)
    out = model(dummy)
    assert out.shape == (4, cfg.data.num_classes), f"unexpected output shape {out.shape}"
    logger.info(f"Forward pass OK: input {tuple(dummy.shape)} -> output {tuple(out.shape)}")

    logger.info(
        "Phase 1 environment check complete. "
        "NOTE: this does not attempt to download CIFAR-10 -- see README "
        "'Environment & Data Availability' section before running Phase 2."
    )


if __name__ == "__main__":
    main()
