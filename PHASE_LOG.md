# Phase Log

## Phase 1 — Architecture + Environment + Baseline Model (COMPLETE)

**Built:**
- Full project directory structure (see README).
- Central YAML config system (`configs/config.yaml`, `src/utils/config.py`)
  with dotted-key overrides and per-run config snapshotting (`save_config`).
- Reproducibility utilities: seeding (`src/utils/seed.py`), environment/
  hardware capture (`src/utils/env_info.py`), structured logging
  (`src/utils/logging_config.py`).
- CIFAR-10 data pipeline: transforms (train aug vs. deterministic eval),
  reproducible train/val split carved out of the official training set with
  its own independent seed, DataLoader construction
  (`src/data/cifar10.py`, `src/data/split_utils.py`).
- ResNet-18 model with CIFAR stem, parameter counting incl. non-zero count
  for later pruning experiments (`src/models/resnet.py`).
- `experiments/check_environment.py`: Phase 1 CLI entry point — records
  environment info and runs a real forward pass to sanity-check the model.
- 17 unit tests across config/model/data/env/seed, all passing, none of
  which require a dataset download or GPU (`tests/`).
- `git init` + initial commit.

**Verified in this sandbox (1 CPU core, 4 GB RAM, no GPU, CIFAR-10 host
unreachable):**
- `python -m experiments.check_environment` runs end-to-end: builds the
  model, confirms output shape `(4, 10)` from a `(4, 3, 32, 32)` dummy
  input, reports ~11.17M parameters.
- `pytest tests/ -v` → 17/17 passed.
- One real bug caught by the tests, recorded rather than hidden: an initial
  test assumed a freshly-initialized model has zero zero-valued parameters.
  It failed (11,169,162 non-zero vs. 11,173,962 total). Root cause:
  `nn.BatchNorm2d` initializes `bias=0` by default in PyTorch, so ~4.8k
  params are exactly zero before any pruning happens. The test was
  corrected to assert a small, explained tolerance instead of silently
  loosened or deleted — see `tests/test_model.py::test_nonzero_close_to_total_before_pruning`.

**Known constraints / risks carried forward to later phases:**

1. **CIFAR-10 cannot be downloaded from this sandbox.** torchvision's only
   source is `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`,
   which this container's network egress policy blocks (confirmed: direct
   fetch → HTTP 403; several candidate GitHub mirrors checked → none host
   a real, complete copy). This is a hard infrastructure limit, not
   something to route around by fabricating data. **Action:** Phase 2
   (baseline training) must run either (a) on a machine with unrestricted
   internet, or (b) in this project directory after manually placing
   `cifar-10-python.tar.gz` at `data/cifar-10-python.tar.gz`.

2. **This sandbox has 1 CPU core and no GPU.** Rough estimate: ResNet-18 on
   CIFAR-10 typically needs ~50-100 epochs to reach reported literature
   accuracy (~93-95%) on a single modern GPU in well under an hour; on a
   single CPU core, per-epoch time is roughly 50-150x slower depending on
   BLAS threading, putting a full 100-epoch run at many hours to over a day
   — and that's for *one* of the ~10+ model variants this project's design
   calls for (baseline + 4 pruning levels x 2 methods + 4 quantization
   methods + combined variants). **Action:** either (a) run training on
   better hardware (a few CPU cores minimum, ideally one GPU) and bring
   checkpoints/results back into this repo, or (b) if continuing in a
   CPU-constrained environment, deliberately reduce scope for Phase 2
   (fewer epochs, documented explicitly as a limitation, not hidden) —
   this trade-off should be an explicit decision before Phase 2 starts,
   not discovered mid-run.

3. **This sandbox's filesystem is ephemeral / scratch space** — it does not
   persist between separate conversations/sessions. The working copy of
   this repo needs to be pushed to a real Git remote (or downloaded) before
   the session ends, or the work is lost. A `git init` + commit has been
   done locally as a first step; connect it to a real remote (e.g.
   `git remote add origin <your-repo-url>`) as soon as possible.

4. **Disk space in this sandbox is limited** (~5-6 GB free after installing
   PyTorch). This is enough for CIFAR-10 (~170 MB) plus a modest number of
   ResNet-18 checkpoints (~45 MB each in FP32) but should be watched once
   the pruning/quantization/combined checkpoint grids are all populated —
   plan on external/cloud storage for the full experiment grid.

## Phase 2 — NOT STARTED
Baseline training + full evaluation (accuracy/precision/recall/F1, params,
size, CPU latency, memory, throughput). Blocked on the data-access decision
above — needs sign-off on which path (A/B from README) to take before writing
the training loop against a compute budget.

## Phases 3-9 — NOT STARTED
Pruning, quantization, combined, benchmarking/visualization, deployment,
literature review, paper.
