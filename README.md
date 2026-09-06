# Accuracy–Efficiency Trade-offs in Neural Network Compression

A reproducible benchmark of pruning and quantization on CIFAR-10 / ResNet-18,
built toward a conference/journal-style empirical study.

**Status: Phase 1 of 9 complete** (project architecture, config system,
environment recording, model definition). See `PHASE_LOG.md` for what's done
and what's next.

## Research Questions

- **RQ1.** How much compression can pruning achieve before accuracy degrades significantly?
- **RQ2.** How does quantization affect accuracy, size, memory, and latency?
- **RQ3.** Does pruning + quantization combined beat either alone?
- **RQ4.** How do compressed models perform under CPU inference/deployment?

## Environment & Data Availability (read this first)

This repository was scaffolded and unit-tested inside a sandboxed container
with:
- **1 logical CPU core, ~4 GB RAM, no GPU** (`torch.cuda.is_available() == False`)
- **No network access to `www.cs.toronto.edu`**, which is torchvision's only
  source URL for CIFAR-10 — the official dataset host is unreachable from
  that sandbox's network egress rules.

Because of this, two things are true and important:

1. **The code in this repo is real, tested, and runs correctly** — every
   module has passing unit tests (`pytest tests/`), and
   `experiments/check_environment.py` runs the model through a real forward
   pass. Tests that need dataset-shaped input use a synthetic drop-in
   dataset (`tests/test_data.py::_FakeCIFAR10`) so pipeline *logic* can be
   verified without network access.
2. **No CIFAR-10 training has actually been run yet**, and no accuracy/
   latency numbers exist yet. Per this project's own rules ("never fabricate
   results, never invent numbers"), none are reported. Phase 2 onward must
   be run on a machine that (a) can reach the CIFAR-10 host or has the
   archive placed locally, and (b) has enough compute that ResNet-18
   training finishes in a reasonable time (a handful of CPU cores or any
   GPU; on 1 CPU core, 100 epochs of ResNet-18/CIFAR-10 is impractically
   slow — see `PHASE_LOG.md` for the estimate).

**To run Phase 2+ yourself:**
```bash
# Option A: your machine can reach the internet
python -m experiments.check_environment   # confirms env + model, no download

# Option B: air-gapped / restricted network
# Download cifar-10-python.tar.gz (163 MB) on any machine with access, e.g.:
#   wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# then place it at:
#   data/cifar-10-python.tar.gz
# torchvision will detect and extract the local archive without re-downloading.
```

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate

# CPU-only PyTorch (recommended unless you have a CUDA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Or, simplest (may pull a much larger CUDA build depending on your platform):
pip install -r requirements.txt
```

## Reproducibility

Every run seeds Python/NumPy/PyTorch (`src/utils/seed.py`) and can capture a
full environment snapshot (`src/utils/env_info.py` — Python/PyTorch/CUDA
versions, CPU model, core count, RAM):

```bash
python -m experiments.check_environment
# -> results/raw/environment.json
```

All experiment parameters live in `configs/config.yaml`, loaded via
`src/utils/config.py`. Nothing should be hard-coded in experiment scripts —
override any field via CLI flags without editing source.

## Project Structure

```
efficient-ml-benchmark/
├── configs/config.yaml       # single source of truth for all hyperparameters
├── src/
│   ├── data/                 # CIFAR-10 loading, transforms, reproducible split
│   ├── models/                # ResNet-18 (CIFAR stem variant)
│   ├── training/               # (Phase 2)
│   ├── compression/            # (Phase 3-5: pruning, quantization, combined)
│   ├── evaluation/             # (Phase 2+: accuracy/latency/memory metrics)
│   ├── benchmarking/           # (Phase 6: unified results table + figures)
│   └── utils/                 # config, seeding, logging, env-info
├── experiments/               # CLI entry points, one per phase
├── models/{baseline,pruning,quantization,combined}/  # checkpoints
├── results/{raw,processed,figures,tables}/
├── deployment/{api,frontend}/  # (Phase 7)
├── tests/
└── paper/                      # (Phase 9)
```

## Running Tests

```bash
pytest tests/ -v
```

Current status: **17/17 tests passing** (config loading, model construction/
shapes/param-counts, transform pipeline, reproducible-split logic, seeding,
environment capture). No dataset download or GPU required to run these.

## Model

ResNet-18 with the standard **CIFAR stem** (3x3 stride-1 conv, no max-pool)
instead of the ImageNet stem (7x7 stride-2 conv + max-pool) — the ImageNet
stem discards too much spatial detail on 32x32 input. See
`src/models/resnet.py` docstring for the full rationale and the literature
this follows. ~11.17M parameters, trained from scratch (no ImageNet
pretraining, to avoid confounding compression results with a transfer-
learning effect).

## License

MIT — see `LICENSE`.
