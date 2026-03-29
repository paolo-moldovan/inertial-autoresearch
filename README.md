# IMU Denoising Auto-Research

Autonomous research pipeline for IMU (Inertial Measurement Unit) signal denoising. Combines deep learning models with classical signal processing baselines, evaluated on EuRoC Machine Hall and MIT Blackbird datasets.

## Quick Start

### Setup
```bash
git clone <repo>
cd inertial-autoresearch
uv sync --extra dev  # Install with dev tools
```

### Training
```bash
# Train LSTM on EuRoC
uv run scripts/train.py --config configs/models/lstm.yaml

# Quick debug run on synthetic data (2 epochs)
uv run scripts/train.py --config configs/training/quick.yaml

# Override from CLI
uv run scripts/train.py --config configs/models/lstm.yaml --set training.lr=0.0001
```

## Key Features

✓ **Multi-Device Support**: CUDA, MPS (Apple Silicon), CPU with auto-detection  
✓ **Model Zoo**: LSTM, Conv1D, Transformer denoisers with residual connections  
✓ **Data Pipelines**: EuRoC (200Hz), Blackbird (100Hz), Synthetic IMU generators  
✓ **Config System**: YAML-based hierarchical configuration with CLI overrides  
✓ **Training Engine**: Time-budgeted training, callbacks, reproducibility  
✓ **Evaluation**: RMSE, MAE, spectral divergence, visualization  
✓ **Classical Baselines**: Kalman filter, complementary filter  
✓ **Auto-Research**: Fast iteration loop + full 23-stage ResearchClaw pipeline  
✓ **Tests**: 33 unit tests, linting (ruff), type checking (mypy)  

## Architecture

```
src/imu_denoise/
├── config/       # YAML loading + frozen dataclasses
├── device/       # CUDA/MPS/CPU abstraction
├── data/         # Dataset registry, transforms, splits
├── models/       # BaseDenoiser + LSTM/Conv1D/Transformer
├── training/     # Trainer, callbacks, losses, optimizers
├── evaluation/   # Metrics, evaluator, visualization
├── classical/    # Kalman, complementary filter
└── utils/        # Logging, I/O, quaternion math
```

## Testing

```bash
uv run -m pytest tests/unit/ -v  # 33 tests pass ✓
uv run ruff check src/           # All checks pass ✓
uv run mypy src/                 # Type checking
```

## Device Support

Auto-detects: **CUDA > MPS > CPU**

Override via config:
```bash
uv run scripts/train.py --set device.preferred=cpu
```

MPS limitations: no bfloat16, no torch.compile

## Auto-Research

**Mode 1: Fast Loop**
```bash
python autoresearch_loop/loop.py --max-iterations 50 --time-budget-sec 600
```

**Mode 2: Full Pipeline**
```bash
uv run -m vendor.AutoResearchClaw run --config autoresearch_loop/researchclaw_config.yaml
```

## References

- EuRoC: http://projects.asl.ethz.ch/datasets/
- Blackbird: https://blackbird-dataset.mit.edu/
- AutoResearchClaw: 23-stage autonomous research pipeline
- Hermes Agent: AI agent orchestration

See CLAUDE.md for detailed architecture documentation.
