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

## Phase 2 — Baseline Training + Evaluation (CODE COMPLETE, NOT RUN ON REAL DATA)
Built `src/evaluation/metrics.py` (accuracy/precision/recall/F1 via sklearn,
on-disk model size, repeated-run CPU latency mean±std, psutil-based memory
delta, throughput -- the one evaluation function every later phase reuses),
`src/training/train.py` (SGD+cosine+warmup, checkpointing on best val
accuracy, early stopping), `src/benchmarking/results_store.py` (master
results.csv/.jsonl writer with a fixed schema), and
`experiments/train_baseline.py`. Verified with `--smoke-test` (1 epoch,
synthetic 32x32 random-label data, no CIFAR-10 needed) and a direct
evaluation-pipeline smoke test -- both run end-to-end. 27/27 tests passing
at this point (10 new: `test_metrics.py`, `test_training.py`).

One more real bug caught and fixed rather than hidden: a test asserted the
optimizer's LR equals the configured base LR right after building the
scheduler; it failed (0.02 != 0.1) because `LambdaLR` applies the warmup
factor immediately at construction. Test corrected to check `initial_lr`
plus the expected warmup-scaled live LR.

**Still blocked on real numbers**: no actual CIFAR-10 training has run --
same data-access and compute constraints as Phase 1.

## Phase 3 — Pruning (CODE COMPLETE, NOT RUN ON REAL DATA)
`src/compression/pruning.py`: unstructured L1 magnitude pruning and
structured (channel/Ln) pruning via `torch.nn.utils.prune`, at the
sparsities in `configs/config.yaml`. `experiments/run_pruning.py` sweeps
method x sparsity, fine-tunes each pruned model briefly, and appends
results. **Load-bearing honesty finding, verified empirically (not just
asserted in a docstring)**: pruning via `torch.nn.utils.prune` masks
weights to zero but does NOT shrink on-disk model size or reduce CPU
latency, because PyTorch's dense tensors still store/compute over the
zeros. Confirmed directly: after 80% unstructured pruning, `model_size_mb`
was unchanged (42.70 MB before and after) while achieved sparsity measured
correctly at 0.800. `run_pruning.py` refuses to run against a missing
baseline checkpoint (raises `FileNotFoundError` with an actionable message)
rather than pruning random weights and reporting the result. 34/34 tests
passing at this point.

## Phase 4 — Quantization (CODE COMPLETE, NOT RUN ON REAL DATA)
`src/compression/quantization.py` implements FP16, dynamic INT8, and
static INT8 (eager-mode fuse -> QuantStub/DeQuantStub -> calibrate ->
convert), and explicitly does NOT implement quantization-aware training
(out of scope given the CPU training budget -- noted for the paper's
Limitations section, not silently dropped).

Two real, non-obvious problems were hit and resolved while building this,
both documented in the module docstring/code rather than just fixed
silently:
1. **Dynamic quantization's real coverage on a CNN.** PyTorch's eager-mode
   `quantize_dynamic` only supports `nn.Linear` (and RNN types) -- verified
   directly on this model: only 5,130 of 11,173,962 parameters (~0.05%,
   just the final `fc` layer) are actually converted to a quantized module.
   `count_actually_quantized_params()` measures and reports this so a large
   claimed compression ratio can never be attributed to dynamic
   quantization here.
2. **ResNet's skip connection breaks eager-mode static quantization.**
   `out += identity` in `torchvision`'s `BasicBlock.forward` raises
   `NotImplementedError: Could not run 'aten::add.out' ... 'QuantizedCPU'`
   at convert time -- quantized tensors can't use the plain `+` operator.
   Fixed with `_QuantizableBasicBlock`, a `BasicBlock` subclass that routes
   the residual add through `torch.ao.nn.quantized.FloatFunctional().add()`
   instead, which does have a QuantizedCPU kernel. Verified static
   quantization now runs end-to-end and produces a real size reduction on
   a freshly-initialized model (not yet on a real trained one).
3. Also surfaced (not yet acted on, noted for awareness): PyTorch emits a
   `DeprecationWarning` that eager-mode `torch.ao.quantization` will be
   removed in PyTorch 2.10 in favor of `torchao`'s `quantize_` /
   `pt2e` APIs. This project's PyTorch pin should be checked against that
   timeline before the paper is finalized; noted as a maintenance risk, not
   fixed now since the current API still works and migrating is nontrivial.
44/44 tests passing at this point (`test_quantization.py`).

## Phase 5 — Combined Pruning + Quantization (CODE COMPLETE, NOT RUN ON REAL DATA)
`experiments/run_combined.py` sweeps a deliberately restricted grid
(`configs/config.yaml` `combined.sparsities: [0.4, 0.6]` x
`combined.quantization_methods: [dynamic_int8, static_int8]` = 4
combinations, not the full 5x4 cross product) to keep the experiment count
interpretable, per the spec's "do not run unnecessary combinations"
instruction -- rationale documented in the script itself. Order of
operations: prune + fine-tune first, then quantize the fine-tuned pruned
model (quantizing before pruning would make magnitude-based pruning
thresholds operate on already-quantized values, which is a different and
less standard design). `test_combined.py` verifies a pruned-then-statically-
quantized model runs end-to-end.

## Phase 6 — Benchmarking + Visualization (CODE COMPLETE, VERIFIED WITH SYNTHETIC ROWS)
`src/benchmarking/aggregate.py` computes `compression_ratio` centrally
against whichever baseline row is actually in `results.csv` (not a
hard-coded reference size, so it can't silently go stale). `src/benchmarking/
visualize.py` generates all 6 required figures (accuracy vs. compression
ratio, accuracy vs. latency, size vs. latency, accuracy vs. size,
compression ratio comparison, Pareto frontier) and explicitly skips
plotting a technique with too few points as if it were a trend line.
`experiments/run_benchmark.py` wires aggregation + figures + a paper-ready
summary CSV together.

Verified by writing 4 clearly-labeled fake rows (one per technique,
notes field says "smoke test") into `results/raw/results.csv`, running the
full pipeline, and visually inspecting the output figures (e.g. the Pareto
frontier plot correctly traced the non-dominated points). **All fake rows
and generated figures were deleted immediately after verification** so
nothing resembling a real result is sitting in the repo -- `results/raw/`,
`results/processed/`, `results/figures/`, `results/tables/` are empty
(just `.gitkeep`) again. 49/49 tests passing at this point.

## Phase 7 — NOT STARTED
Deployment: FastAPI backend, Streamlit frontend, Docker.

## Phases 8-9 — NOT STARTED
Literature review + research gap, research paper.

## Cross-phase status
No real CIFAR-10 training has occurred anywhere in this project. Every
number that would go into `results/raw/results.csv` from a real run is
still absent -- by design, per "never fabricate results, never invent
numbers." All six experiment scripts (`check_environment.py`,
`train_baseline.py`, `run_pruning.py`, `run_quantization.py`,
`run_combined.py`, `run_benchmark.py`) are implemented, individually
smoke-tested against synthetic data or freshly-initialized models, and
covered by unit tests -- they are ready to run for real the moment (a)
CIFAR-10 is available (network access or a manually-placed archive) and
(b) enough compute is available to make the training budget practical (see
Phase 1 risk #2). This is the single most important remaining gap before
any paper section beyond Methodology can be written.
