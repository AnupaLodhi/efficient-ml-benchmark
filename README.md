# Efficient ML Benchmark

> **A reproducible benchmarking framework for studying the accuracy–efficiency trade-offs of neural network compression techniques, including pruning, quantization, and combined compression.**

This project investigates a simple but important question:

**How much can we reduce the computational and storage cost of a deep learning model before its predictive performance becomes unacceptable?**

The framework uses **ResNet-18 on CIFAR-10** as the reference workload and evaluates multiple model-compression strategies under a common benchmarking pipeline.

---

## Overview

Modern deep neural networks can achieve strong predictive performance, but their computational cost, memory footprint, and inference requirements can make deployment difficult on resource-constrained hardware.

This repository provides an experimental framework for evaluating:

* Baseline FP32 models
* Unstructured magnitude pruning
* Structured pruning support
* FP16 quantization
* Dynamic INT8 quantization
* Static INT8 quantization
* Combined pruning + quantization
* Accuracy and F1-score
* Model size and compression ratio
* CPU inference latency
* Throughput
* Memory usage
* Training/fine-tuning cost

Rather than reporting compression alone, the project studies the **trade-off between model quality and computational efficiency**.

---

## Research Question

The central research question is:

> **What accuracy–efficiency trade-offs emerge when pruning and quantization are applied individually and jointly to convolutional neural networks?**

The experiments are designed to compare three major scenarios:

**Baseline → Pruning → Quantization → Combined Compression**

This allows us to investigate not only whether a model becomes smaller, but also what is sacrificed in return.

---

## Model and Dataset

### Model

The benchmark currently uses:

**ResNet-18 with a CIFAR-compatible input stem**

The baseline architecture contains approximately:

```text
11.17 million parameters
```

### Dataset

Experiments use the **CIFAR-10** image classification dataset.

CIFAR-10 contains 60,000 color images across 10 classes.

The benchmark uses:

```text
Training set:     45,000 images
Validation set:    5,000 images
Test set:         10,000 images
```

Classes include:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

---

## Compression Techniques

### 1. Baseline FP32

A standard FP32 ResNet-18 serves as the reference model.

All compression experiments are compared against this baseline using common evaluation metrics.

---

### 2. Unstructured Pruning

Unstructured magnitude pruning removes individual low-importance weights by setting them to zero.

The benchmark supports experiments at multiple sparsity levels, including:

```text
20%
40%
60%
80%
```

Conceptually:

```text
Dense Network
      ↓
Identify low-magnitude weights
      ↓
Prune weights
      ↓
Fine-tune network
      ↓
Evaluate accuracy and efficiency
```

An important distinction is that the current unstructured implementation uses **dense masking**.

Therefore, increasing sparsity does **not automatically reduce the serialized model size or guarantee lower CPU latency**. Sparse storage formats or sparse execution kernels would be required to obtain those benefits reliably.

---

### 3. FP16 Quantization

FP16 reduces the numerical precision of model parameters from 32-bit floating point to 16-bit floating point.

Conceptually:

```text
FP32
32 bits / value

↓

FP16
16 bits / value
```

This can substantially reduce model storage.

However, lower precision does not guarantee faster inference on every processor. Hardware support is an important part of the benchmark.

---

### 4. Dynamic INT8 Quantization

Dynamic quantization converts supported operations to INT8 dynamically during inference.

In the current ResNet-18 implementation, dynamic quantization primarily affects the final fully connected layer.

Therefore, it should **not be interpreted as full-network INT8 quantization**.

This distinction is explicitly tracked so compression results are not overstated.

---

### 5. Static INT8 Quantization

Static quantization applies INT8 quantization more broadly to supported network operations.

The process includes:

```text
FP32 model
     ↓
Module fusion
     ↓
Observer insertion
     ↓
Calibration data
     ↓
INT8 conversion
     ↓
Evaluation
```

Static INT8 can provide substantially greater storage compression, but it may also produce a larger accuracy penalty depending on the model, calibration procedure, and hardware.

---

### 6. Combined Compression

The final compression experiment combines pruning and quantization.

The pipeline follows:

```text
Baseline ResNet-18
        ↓
Unstructured pruning
        ↓
Fine-tuning
        ↓
Quantization
        ↓
Evaluation
```

The restricted combined benchmark studies:

```text
40% pruning + Dynamic INT8
40% pruning + Static INT8

60% pruning + Dynamic INT8
60% pruning + Static INT8
```

The order is intentionally:

**Prune → Fine-tune → Quantize**

Quantization is applied after the pruned model has been fine-tuned.

---

## Metrics

Each experiment records several dimensions of model performance.

### Predictive Performance

```text
Accuracy
Macro Precision
Macro Recall
Macro F1-score
```

### Model Efficiency

```text
Total parameters
Non-zero parameters
Sparsity
Model size
Compression ratio
```

### Inference Performance

```text
Mean latency
Latency standard deviation
Minimum latency
Maximum latency
Throughput
```

### Resource Usage

```text
Peak RSS memory change
Training/fine-tuning time
```

This multi-dimensional evaluation prevents a compressed model from being considered "better" simply because its file size is smaller.

---

## Experimental Pipeline

The overall workflow is:

```text
                 ┌─────────────────┐
                 │    CIFAR-10     │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    ResNet-18    │
                 │  FP32 Baseline  │
                 └────────┬────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
     ┌─────────┐     ┌──────────┐   ┌─────────────┐
     │ Pruning │     │Quantize  │   │  Combined   │
     └────┬────┘     └────┬─────┘   └──────┬──────┘
          │               │                │
          └───────────────┼────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Evaluation    │
                 ├─────────────────┤
                 │ Accuracy / F1   │
                 │ Model Size      │
                 │ Latency         │
                 │ Throughput      │
                 │ Memory          │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Benchmark Data  │
                 │ + Visualizations│
                 └─────────────────┘
```

---

## Repository Structure

```text
efficient-ml-benchmark/
│
├── configs/
│   ├── config.yaml
│   └── config_mac.yaml
│
├── data/
│
├── deployment/
│
├── experiments/
│   ├── check_environment.py
│   ├── train_baseline.py
│   ├── run_pruning.py
│   ├── run_quantization.py
│   ├── run_combined.py
│   └── run_benchmark.py
│
├── models/
│   ├── baseline/
│   ├── pruning/
│   └── combined/
│
├── notebooks/
│
├── paper/
│
├── results/
│   ├── raw/
│   └── visualizations/
│
├── src/
│   ├── benchmarking/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── pruning/
│   ├── quantization/
│   ├── training/
│   └── utils/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AnupaLodhi/efficient-ml-benchmark.git
cd efficient-ml-benchmark
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Intel macOS Compatibility

The project has also been validated locally on an **Intel-based macOS system using CPU-only PyTorch**.

A compatible environment used during development includes:

```text
Python        3.12.10
PyTorch       2.2.2
torchvision   0.17.2
NumPy         1.26.4
SciPy         1.13.1
scikit-learn  1.5.2
matplotlib    3.9.2
seaborn       0.13.2
```

For reproducibility on this environment, see:

```text
requirements-lock.txt
```

Hardware-specific results should not be assumed to generalize to GPUs, ARM processors, or newer CPUs.

---

## Running the Benchmark

### Check the environment

```bash
python -m experiments.check_environment
```

### Train the baseline

```bash
python -m experiments.train_baseline
```

For a short pipeline-validation run:

```bash
python -m experiments.train_baseline --epochs 3
```

### Run pruning experiments

```bash
python -m experiments.run_pruning \
  --checkpoint models/baseline/resnet18_fp32_best.pt
```

### Run quantization experiments

```bash
python -m experiments.run_quantization \
  --checkpoint models/baseline/resnet18_fp32_best.pt
```

### Run combined compression

```bash
python -m experiments.run_combined \
  --config configs/config_mac.yaml \
  --checkpoint models/baseline/resnet18_fp32_best.pt
```

### Smoke-test the combined pipeline

```bash
python -m experiments.run_combined \
  --config configs/config_mac.yaml \
  --smoke-test
```

---

## Current Local Validation Results

The current repository has been validated end-to-end on CPU.

The short-run FP32 baseline produced approximately:

```text
Test Accuracy:      64.75%
Macro F1:           64.83%
Parameters:         11.17 M
Model Size:         42.70 MB
Mean CPU Latency:   18.28 ms
```

These results come from a deliberately shortened local training run and are provided primarily to demonstrate that the complete experimental pipeline executes correctly.

### Quantization observations

Local CPU experiments demonstrated several interesting hardware-dependent behaviors:

```text
FP32
Size:      ~42.70 MB
Accuracy:  ~64.75%

FP16
Size:      ~21.37 MB
Accuracy:  ~64.78%

Dynamic INT8
Size:      ~42.69 MB
Accuracy:  ~64.79%

Static INT8
Size:      ~10.78 MB
Accuracy:  ~33.28%
```

These results demonstrate why compression should be evaluated across **multiple metrics rather than model size alone**.

For example, FP16 substantially reduced storage but was not faster on the tested Intel CPU, while Static INT8 achieved much stronger compression at a significant accuracy cost.

---

## Important Experimental Note

The current local results should be interpreted as **pipeline-validation and hardware-specific experimental results**, not as final publication-quality benchmark claims.

Some local experiments intentionally used shortened training schedules because full ResNet-18 sweeps are computationally expensive on the development machine.

In particular, different pruning runs have used different fine-tuning budgets during local validation.

Therefore:

> **Accuracy differences across those preliminary runs should not be attributed solely to pruning.**

A final controlled research benchmark should use:

* A properly converged baseline
* Identical training/fine-tuning budgets
* Consistent hyperparameters
* Multiple random seeds
* Repeated latency measurements
* Controlled hardware
* Statistical summaries
* Identical evaluation datasets

This distinction is important for reproducible and scientifically defensible conclusions.

---

## Testing

The repository includes automated tests for the major components.

Run:

```bash
pytest
```

The current validated development environment passes:

```text
57 tests passed
```

Tests cover components including configuration handling, metrics, benchmarking utilities, pruning, quantization, training infrastructure, and deployment-related functionality.

---

## Deployment

The repository includes infrastructure for model serving and demonstration.

The deployment layer includes support for:

* FastAPI-based inference
* Streamlit interface
* Model registry
* Docker
* Docker Compose

This makes the project more than an offline experiment: compressed models can eventually be evaluated from both a **research perspective and a deployment perspective**.

---

## Reproducibility

Reproducibility is a central design goal of this repository.

The framework records:

```text
Configuration
Environment information
Model checkpoints
Raw experimental results
Training time
Inference measurements
Compression metadata
```

Raw benchmark results are stored in:

```text
results/raw/
```

Experiment configurations are stored in:

```text
configs/
```

Model checkpoints are organized under:

```text
models/
```

---

## Current Project Status

```text
Environment & architecture        ✅
Baseline training pipeline        ✅
Pruning pipeline                  ✅
Quantization pipeline             ✅
Combined compression pipeline     ✅
Automated tests                   ✅
Benchmark aggregation             🚧
Final controlled experiments      🚧
Final figures / analysis          🚧
Deployment validation             🚧
Research manuscript               🚧
```

The engineering pipeline is functional end-to-end.

The next research milestone is to move from local pipeline-validation experiments to a **fully controlled benchmark protocol suitable for formal analysis and publication**.

---

## Future Work

Planned extensions include:

* Fully converged baseline training
* Multi-seed controlled experiments
* GPU benchmarking
* Additional CNN architectures
* Additional datasets
* Sparse inference kernels
* Hardware-aware benchmarking
* Quantization-aware training
* Improved INT8 calibration
* ONNX / runtime-specific benchmarking
* Energy and power measurements
* Statistical significance analysis
* Pareto-front analysis of accuracy vs. efficiency
* Automated experiment reporting

---

## Research Philosophy

A smaller model is not automatically a better model.

A faster model is not automatically a better model.

And a compressed model that loses too much predictive performance may not be useful at all.

This project therefore treats efficient machine learning as a **multi-objective optimization problem** involving:

```text
Accuracy
   ↕
Model Size
   ↕
Latency
   ↕
Memory
   ↕
Throughput
   ↕
Hardware
```

The goal is to identify useful operating points rather than simply maximizing compression.

---

## Author

**Anupa Lodhi**

B.Tech — Computer Science & Engineering

Research interests include:

* Efficient Machine Learning
* Model Compression
* Artificial Intelligence
* Machine Learning Systems
* Responsible and Deployable AI

---

## License

This project is licensed under the **MIT License**.

---

## Citation

If this repository contributes to academic work, please cite the repository.

A formal paper citation will be added after publication.

---

⭐ If you find this project useful, consider starring the repository.

