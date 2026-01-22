# Surrogate-Assisted Language Model Training

Train language models from scratch with optional surrogate model guidance (SDCE - Surrogate-guided Distillation Cross-Entropy).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single GPU
python train.py --config config.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train.py --config config.yaml

# Apple Silicon
python train.py --config config.yaml --device mps

# TPU
USE_TPU=1 python train.py --config config.yaml
```

## Supported Devices

| Device | Command |
|--------|---------|
| CUDA (single) | `python train.py --config config.yaml` |
| CUDA (multi) | `torchrun --nproc_per_node=N train.py --config config.yaml` |
| MPS | `python train.py --config config.yaml --device mps` |
| TPU | `USE_TPU=1 python train.py --config config.yaml` |
| CPU | `python train.py --config config.yaml --device cpu` |

## Key Features

- **SDCE Loss**: Surrogate model guides training via soft token targets
- **Mixed Precision**: bf16/fp16 with automatic device adaptation
- **Flash Attention 2**: Auto-enabled on supported GPUs
- **Z-Loss**: PaLM-style auxiliary loss for stability
- **Distributed**: Full DDP support via torchrun

## Configuration

Edit `config.yaml` to customize:
- `model.name_or_path`: Base model architecture
- `sdce.mode`: `"surrogate"`, `"kd"`, or `"none"` (controls surrogate usage)
- `surrogate.prob_threshold`: Token selection threshold (default: 0.03)
- `training.mixed_precision`: `"bf16"`, `"fp16"`, or `"fp32"`

## Files

- `train.py` - Main training script
- `losses.py` - Loss functions (CE, SDCE, KD, Z-loss)
- `config.yaml` - Training configuration
- `requirements.txt` - Dependencies
