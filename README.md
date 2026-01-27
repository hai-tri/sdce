# Surrogate-Assisted Language Model Training

A generalized training framework for language models with surrogate model guidance. The surrogate model provides token-level perplexity signals that guide the primary model's learning process.

## Overview

This implementation is based on a training approach where:

1. A **surrogate model** (e.g., Qwen-0.6B) provides guidance on which tokens the base model should focus on
2. The loss function incorporates both standard cross-entropy loss and a weighted surrogate term
3. Token vocabulary intersection between models is handled via bidirectional lookup tables

### Key Features

- **Flexible Model Selection**: Use any HuggingFace-compatible model as base or surrogate
- **Train from Scratch**: Initialize models with random weights instead of pretrained weights
- **Vocabulary Alignment**: Automatic handling of vocabulary differences between tokenizers
- **Mixed Precision Training**: Support for FP16 and BF16 training
- **Distributed Training**: DDP support via torchrun for multi-GPU training
- **TPU Support**: Native PyTorch XLA support for Google Cloud TPUs
- **Auto Eval Split**: Automatically creates evaluation split if not available in dataset
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

# Optional: Install PyTorch XLA for TPU support
pip install torch-xla
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

# Training from scratch (random weights initialization)
python train.py \
    --base_model gpt2 \
    --init_from_scratch \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./outputs

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

# Train from scratch with config
python train.py --config config.yaml --init_from_scratch
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
  init_from_scratch: false           # Initialize with random weights

# Surrogate model configuration
surrogate:
  name_or_path: "Qwen/Qwen3-0.6B"   # Surrogate model
  dtype: "float16"                   # Data type for surrogate
  k: 30                              # Top-k tokens to consider (candidate pool)
  probability_threshold: 0.02        # Min probability to include (0.0 = no filter)
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
  eval_split_ratio: 0.05             # Auto-create eval split ratio
  eval_split_seed: 42                # Seed for reproducible split
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
  loss_type: "surrogate"            # standard, surrogate, or kl
  device: "auto"                     # auto, cuda, mps, tpu, cpu
  tpu_cores: 1                       # For TPU: 1, 8, etc.
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

## Loss Types

The training script supports three different loss types:

### Standard Training (`--standard_training` or `--loss_type standard`)

Pure cross-entropy loss without any surrogate guidance. Use this for typical language model training:

```bash
python train.py \
    --base_model gpt2 \
    --dataset wikitext \
    --standard_training
```

### Surrogate-Guided Training (`--loss_type surrogate`, default)

Cross-entropy plus a surrogate-guided auxiliary loss. The token selection works as follows:

1. **Select by probability**: Top-k tokens (default k=30) are selected from the surrogate model's probability distribution
2. **Filter by threshold**: Tokens below the probability threshold (default 0.02) are masked out
3. **Weight by perplexity**: Remaining tokens are weighted using `softmax(-perplexity)` - lower perplexity (higher confidence) gets higher weight

```bash
python train.py \
    --base_model gpt2 \
    --surrogate_model Qwen/Qwen3-0.6B \
    --loss_type surrogate \
    --surrogate_k 30 \
    --probability_threshold 0.02 \
    --dataset wikitext
```

The probability threshold helps focus learning on tokens the surrogate is confident about, filtering out noise from the long tail of the distribution.

### KL Divergence Training (`--kl_divergence` or `--loss_type kl`)

Cross-entropy plus KL divergence from the surrogate model's distribution. This is a standard knowledge distillation approach:

```bash
python train.py \
    --base_model gpt2 \
    --surrogate_model Qwen/Qwen3-0.6B \
    --kl_divergence \
    --dataset wikitext
```

Or equivalently:
```bash
python train.py \
    --base_model gpt2 \
    --surrogate_model Qwen/Qwen3-0.6B \
    --loss_type kl \
    --dataset wikitext
```

## Training Approaches

### 1. Training from Scratch (Random Initialization)

To train a model from random weights instead of pretrained weights:

```bash
python train.py \
    --base_model gpt2 \
    --init_from_scratch \
    --dataset wikitext \
    --surrogate_model Qwen/Qwen3-0.6B
```

Or in config.yaml:
```yaml
model:
  name_or_path: "gpt2"
  init_from_scratch: true
```

### 2. Continual Training (Pretrained Initialization)

The default behavior loads pretrained weights:

```bash
python train.py \
    --base_model gpt2 \
    --dataset wikitext
```

See the **Loss Types** section above for details on `--standard_training`, `--loss_type surrogate`, and `--kl_divergence` options.

## Distributed Training

### Multi-GPU Training with torchrun

Use `torchrun` for distributed data parallel (DDP) training across multiple GPUs:

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py --config config.yaml

# Single node, all available GPUs
torchrun --nproc_per_node=auto train.py --config config.yaml

# Multi-node training
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=master_node:29500 \
    train.py --config config.yaml
```

### TPU Training

For Google Cloud TPU:

```bash
# Single TPU core
python train.py --config config.yaml --device tpu

# TPU v3-8 (single host, 8 cores)
python train.py --config config.yaml --device tpu --tpu_cores 8

# TPU with bfloat16 (recommended)
python train.py --config config.yaml --device tpu --mixed_precision bf16
```

### TPU Pod Training (Multi-Host)

For TPU Pods (v3-32, v4-64, etc.) with multiple hosts:

```bash
# TPU v3-32 (4 hosts × 8 cores = 32 cores total)
# Run this command on EACH host with proper environment variables

# On each host, set environment variables:
export PJRT_DEVICE=TPU
export TPU_PROCESS_COUNT=4  # Total number of hosts
export TPU_PROCESS_ID=<host_id>  # 0, 1, 2, or 3 for this host

# Then run training:
python train.py --config config.yaml --device tpu --tpu_cores 8 --tpu_num_hosts 4
```

#### TPU Pod Environment Setup

**Using PJRT Runtime (PyTorch XLA 2.0+, recommended):**

On each host:
```bash
export PJRT_DEVICE=TPU
export TPU_PROCESS_ADDRESSES="host0:8476,host1:8476,host2:8476,host3:8476"
export TPU_PROCESS_COUNT=4
export TPU_PROCESS_ID=<this_host_id>  # 0-indexed
```

**Using Legacy XRT Runtime:**

On each host:
```bash
export XRT_TPU_CONFIG="tpu_worker;0;host0:8470|tpu_worker;1;host1:8470|..."
export TPU_WORKER_ID=<this_host_id>
```

#### TPU Pod Configuration in YAML

```yaml
training:
  device: "tpu"
  tpu_cores: 8          # Cores per host
  tpu_num_hosts: 4      # Total hosts in the pod
  tpu_use_pjrt: true    # Use PJRT runtime (recommended)
  mixed_precision: "bf16"  # bfloat16 recommended for TPU
```

#### Common TPU Configurations

| TPU Type | Hosts | Cores/Host | Total Cores | Command |
|----------|-------|------------|-------------|---------|
| v3-8     | 1     | 8          | 8           | `--tpu_cores 8` |
| v3-32    | 4     | 8          | 32          | `--tpu_cores 8 --tpu_num_hosts 4` |
| v3-128   | 16    | 8          | 128         | `--tpu_cores 8 --tpu_num_hosts 16` |
| v4-8     | 1     | 4          | 4           | `--tpu_cores 4` |
| v4-32    | 4     | 4          | 16          | `--tpu_cores 4 --tpu_num_hosts 4` |
| v4-64    | 8     | 4          | 32          | `--tpu_cores 4 --tpu_num_hosts 8` |

## Auto Eval Split Creation

If your dataset doesn't have an evaluation split, the trainer automatically creates one:

```bash
# Uses 5% of training data for evaluation by default
python train.py \
    --base_model gpt2 \
    --dataset your_dataset_without_eval_split

# Custom split ratio
python train.py \
    --base_model gpt2 \
    --dataset your_dataset \
    --eval_split_ratio 0.1  # 10% for evaluation
```

The trainer checks for common eval split names: `validation`, `valid`, `val`, `test`, `dev`. If none exist, it creates a split from the training data.

## How the Surrogate Loss Works

The surrogate-assisted loss combines two components:

1. **Standard Cross-Entropy Loss**: 
   $$\mathcal{L}_{CE} = -\sum_{i} \log p(y_i | x_{<i})$$

2. **Surrogate-Guided Loss**:
   - The surrogate model computes probabilities for each token position
   - Top-k tokens (candidate pool, e.g., k=30) with highest probability are selected
   - **Probability threshold filtering**: Tokens below the threshold (e.g., 0.02) are masked out
   - The actual target token is **masked out** from consideration
   - Remaining tokens are weighted by softmax of negative perplexity
   - The base model is encouraged to also assign probability to these tokens

The combined loss:
$$\mathcal{L} = \mathcal{L}_{CE} + \lambda(t) \cdot \mathcal{L}_{surrogate}$$

Where:
$$\mathcal{L}_{surrogate} = \sum_{i} \sum_{j \in \text{filtered-top-k}} w_{i,j} \cdot (-\log p_{base}(t_j | x_{<i}))$$

And $w_{i,j}$ is the softmax-normalized weight based on surrogate perplexity.

### Probability Threshold Filtering

The `probability_threshold` parameter filters out low-confidence tokens from the surrogate's suggestions:

```bash
python train.py \
    --surrogate_k 30 \
    --probability_threshold 0.02
```

This means:
1. Select top-30 tokens by probability from the surrogate
2. Mask out any tokens with probability < 0.02
3. Weight the remaining tokens by their perplexity

Since perplexity = 1/probability, the threshold check is: `perp > 1/threshold → masked`.

For a threshold of 0.02, tokens with perplexity > 50 are masked out. This ensures only tokens the surrogate is reasonably confident about contribute to the loss.

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
3. **Attention Masking**: Input tokens not in the intersection are masked in the surrogate's attention mechanism (attention weight = 0), so the surrogate model ignores them during forward pass
4. **Target Token Handling**: When a target token doesn't exist in the vocabulary intersection, the entire surrogate contribution for that position is zeroed out (perplexity row set to infinity → softmax weight = 0). This prevents the surrogate from providing misleading guidance for tokens it cannot reason about.

### Intersection Attention Mask

The `compute_intersection_attention_mask` function in `losses.py` creates a binary attention mask:

```python
from losses import compute_intersection_attention_mask

# Create mask: 1 = token in intersection, 0 = not in intersection
attention_mask = compute_intersection_attention_mask(
    input_ids=input_ids,
    lookup_base_to_surrogate=lookup_table
)

# Pass to surrogate model
surrogate_outputs = surrogate_model(
    input_ids=translated_input_ids,
    attention_mask=attention_mask
)
```

### Target Token Zeroing

When a target label token doesn't exist in the surrogate's vocabulary:
- The entire perplexity vector for that position is set to `inf`
- The `all_inf_mask` detection in the loss function identifies these positions
- The softmax weighting produces zero weights for these positions
- The surrogate loss contribution for that position becomes zero

This ensures the model only learns from surrogate guidance when the surrogate can meaningfully reason about the target token.

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

5. **Training from Scratch**: When using `--init_from_scratch`:
   - You may need more training steps
   - Consider larger warmup ratio
   - Monitor training closely for instabilities

6. **TPU Training**:
   - Use `bf16` mixed precision (native support)
   - Set `tpu_cores` appropriately for your TPU type
   - Avoid excessive logging (slow on TPU)

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

### torchrun Issues
- Ensure NCCL is properly installed
- Check network connectivity between nodes
- Verify GPU visibility with `CUDA_VISIBLE_DEVICES`

### TPU Issues
- Install PyTorch XLA: `pip install torch-xla`
- Use `bf16` precision for best performance
- Reduce `num_workers` to 0
- For TPU Pods, ensure environment variables are set correctly on ALL hosts
- Check that all hosts can communicate (network connectivity)
- Verify TPU_PROCESS_ADDRESSES includes all hosts with correct ports

### TPU Pod Debugging
```bash
# Check XLA environment
python -c "import torch_xla; print(torch_xla.__version__)"

# Verify PJRT is enabled
echo $PJRT_DEVICE  # Should print "TPU"

# Check process configuration
echo "Process $TPU_PROCESS_ID of $TPU_PROCESS_COUNT"

# Test XLA device initialization
python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```

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
