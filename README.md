# Surrogate-Assisted Language Model Training

A generalized training framework for language models with surrogate model guidance. The surrogate model provides token-level perplexity signals that guide the primary model's learning process.

## Overview

This implementation is based on a training approach where:

1. A **surrogate model** (e.g., Qwen-0.6B) provides guidance on which tokens the base model should focus on
2. The loss function incorporates both standard cross-entropy loss and a weighted surrogate term
3. Token vocabulary intersection between models is handled via bidirectional lookup tables

### Key Features

- **Flexible Model Selection**: Use any HuggingFace-compatible model as base or surrogate
- **Vocabulary Alignment**: Automatic handling of vocabulary differences between tokenizers
- **Mixed Precision Training**: Support for FP16 and BF16 training
- **Distributed Training**: DDP support for multi-GPU training
- **Configurable via YAML or CLI**: Easy configuration through config files or command line
- **Weights & Biases Integration**: Optional experiment tracking
- **Checkpoint Management**: Automatic checkpoint saving and cleanup

## Installation

```bash
# Clone or download this repository
cd surrogate_lm_trainer

# Install dependencies
pip install -r requirements.txt

# Optional: Install flash-attention for faster training
pip install flash-attn --no-build-isolation
```

## Quick Start

### Using Command Line Arguments

```bash
# Basic training with default settings
python train.py \
    --base_model gpt2 \
    --surrogate_model Qwen/Qwen3-0.6B \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./outputs \
    --batch_size 4 \
    --learning_rate 1e-4

# Training without surrogate guidance (standard training)
python train.py \
    --base_model gpt2 \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --no_surrogate \
    --output_dir ./outputs

# With custom dataset files
python train.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --surrogate_model Qwen/Qwen3-0.6B \
    --train_file /path/to/train.json \
    --eval_file /path/to/eval.json \
    --text_column text \
    --output_dir ./outputs
```

### Using Configuration File

```bash
# Train using config file
python train.py --config config.yaml

# Override specific settings
python train.py --config config.yaml --learning_rate 5e-5 --batch_size 8
```

## Configuration

### Full Configuration Reference

```yaml
# Base model configuration
model:
  name_or_path: "gpt2"              # Model name or path
  dtype: "float16"                   # float16, bfloat16, float32
  use_flash_attention: false         # Enable flash attention
  gradient_checkpointing: false      # Enable gradient checkpointing
  trust_remote_code: false           # Trust remote code for custom models

# Surrogate model configuration
surrogate:
  name_or_path: "Qwen/Qwen3-0.6B"   # Surrogate model
  dtype: "float16"                   # Data type for surrogate
  k: 6                               # Top-k tokens to consider
  enabled: true                      # Enable/disable surrogate
  trust_remote_code: true            # Trust remote code

# Data configuration
data:
  dataset_name: "wikitext"           # HuggingFace dataset
  dataset_config: "wikitext-2-raw-v1"
  dataset_split: "train"
  eval_split: "validation"
  text_column: "text"
  max_seq_length: 1024
  preprocessing_num_workers: 4
  train_file: null                   # For custom datasets
  eval_file: null

# Training configuration
training:
  output_dir: "./outputs"
  num_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  lr_scheduler_type: "cosine"
  mixed_precision: "fp16"
  logging_steps: 10
  eval_steps: 500
  save_steps: 1000
  save_total_limit: 3
  use_z_loss: false
  z_loss_multiplier: 1.0e-4
  seed: 42
  wandb_project: null
  wandb_run_name: null
```

## Training Approaches

### 1. Standard Training (No Surrogate)

For baseline comparison or when you don't need surrogate guidance:

```bash
python train.py \
    --base_model gpt2 \
    --dataset wikitext \
    --no_surrogate
```

### 2. Surrogate-Guided Training

The main training mode with surrogate model providing token guidance:

```bash
python train.py \
    --base_model gpt2 \
    --surrogate_model Qwen/Qwen3-0.6B \
    --surrogate_k 6 \
    --dataset wikitext
```

### 3. Distributed Training

For multi-GPU training:

```bash
torchrun --nproc_per_node=4 train.py \
    --config config.yaml
```

## How the Surrogate Loss Works

The surrogate-assisted loss combines two components:

1. **Standard Cross-Entropy Loss**: 
   $$\mathcal{L}_{CE} = -\sum_{i} \log p(y_i | x_{<i})$$

2. **Surrogate-Guided Loss**:
   - The surrogate model computes perplexity for each token position
   - Top-k tokens with lowest perplexity (highest confidence) are selected
   - The actual target token is **masked out** from consideration
   - These tokens are weighted by softmax of negative perplexity
   - The base model is encouraged to also assign probability to these tokens

The combined loss:
$$\mathcal{L} = \mathcal{L}_{CE} + \lambda(t) \cdot \mathcal{L}_{surrogate}$$

Where:
$$\mathcal{L}_{surrogate} = \sum_{i} \sum_{j \in \text{top-k}} w_{i,j} \cdot (-\log p_{base}(t_j | x_{<i}))$$

And $w_{i,j}$ is the softmax-normalized weight based on surrogate perplexity.

### Surrogate Loss Weight Scheduler

The surrogate loss weight $\lambda(t)$ follows a **cosine decay schedule** (without warmup):

$$\lambda(t) = \lambda_{final} + \frac{1}{2}(\lambda_{initial} - \lambda_{final})\left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)$$

Where:
- $t$ is the current training step
- $T$ is the total number of training steps
- $\lambda_{initial}$ is `loss_weight_initial` (default: 1.0)
- $\lambda_{final}$ is `loss_weight_final` (default: 0.0)

This means the surrogate guidance is strongest at the beginning of training and gradually fades out, allowing the model to learn from the surrogate early on while eventually relying on standard cross-entropy loss.

Configure via CLI:
```bash
python train.py \
    --surrogate_loss_weight_initial 1.0 \
    --surrogate_loss_weight_final 0.0
```

Or in config.yaml:
```yaml
surrogate:
  loss_weight_initial: 1.0
  loss_weight_final: 0.0
```

### Learning Rate Scheduler

The learning rate uses a **cosine annealing schedule with warmup** (via HuggingFace's `get_scheduler`):
- Warmup: Linear warmup from 0 to `learning_rate` over `warmup_steps` (or `warmup_ratio * total_steps`)
- Decay: Cosine decay from `learning_rate` to 0 over the remaining steps

## Vocabulary Alignment

The framework automatically handles vocabulary differences between base and surrogate models:

1. **Intersection Computation**: Finds tokens present in both vocabularies
2. **Lookup Tables**: Creates bidirectional mappings between token IDs
3. **Masking**: Tokens not in intersection are masked in surrogate outputs

## Custom Datasets

### JSON Format

```json
[
    {"text": "First document text..."},
    {"text": "Second document text..."}
]
```

### Plain Text Format

One document per line in a `.txt` file.

### Usage

```bash
python train.py \
    --train_file /path/to/train.json \
    --eval_file /path/to/eval.json \
    --text_column text
```

## Monitoring and Logging

### Weights & Biases

```bash
python train.py \
    --config config.yaml \
    --wandb_project my-project \
    --wandb_run_name experiment-1
```

### Console Logging

Training progress is logged to console with:
- Loss and perplexity
- Learning rate
- Gradient norm
- Tokens per second

## Checkpointing

Checkpoints are saved automatically and include:
- Model weights
- Optimizer state
- Scheduler state
- RNG states for reproducibility
- Training configuration

Resume training:
```bash
python train.py --config config.yaml --resume_from_checkpoint ./outputs/checkpoint-1000
```

## Tips and Best Practices

1. **Choosing Surrogate Model**: Select a surrogate model that is:
   - Smaller than your base model (for efficiency)
   - Has reasonable vocabulary overlap with your base tokenizer
   - Pre-trained on similar data distributions

2. **Setting k**: Start with `k=6` and adjust based on:
   - Higher k: More diverse guidance, slower training
   - Lower k: Focused guidance, faster training

3. **Memory Optimization**:
   - Use gradient checkpointing for large models
   - Reduce batch size and increase gradient accumulation
   - Use FP16/BF16 mixed precision

4. **Vocabulary Overlap**: Check the logged "vocabulary intersection size" - if it's too small, the surrogate guidance may be limited.

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing`
- Use `mixed_precision: "fp16"`

### Slow Training
- Increase `gradient_accumulation_steps` instead of batch size
- Use `num_workers > 0` in data loading
- Consider using Flash Attention

### Poor Results
- Verify vocabulary overlap is sufficient
- Try adjusting `k` parameter
- Check learning rate and warmup settings

## Citation

If you use this code in your research, please cite:

```bibtex
@software{surrogate_lm_trainer,
  title = {Surrogate-Assisted Language Model Training},
  year = {2024},
  url = {https://github.com/your-repo/surrogate-lm-trainer}
}
```

## License

MIT License
