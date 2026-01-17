#!/usr/bin/env python3
"""
Surrogate-Assisted Language Model Training Script

This script implements a training approach where a surrogate model guides the 
primary model's learning by providing token-level perplexity signals.

Supports: CUDA GPUs, Apple MPS, TPUs (via PyTorch XLA), and CPU.

Usage:
    python train.py --config config.yaml
    python train.py --base_model gpt2 --surrogate_model Qwen/Qwen3-0.6B --dataset wikitext --dataset_config wikitext-2-raw-v1
    
    # For TPU (single core)
    python train.py --config config.yaml --device tpu
    
    # For TPU (multi-core with xmp.spawn)
    python train.py --config config.yaml --device tpu --tpu_cores 8
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

# Optional: CUDA AMP
try:
    from torch.amp import autocast, GradScaler
    HAS_CUDA_AMP = True
except ImportError:
    try:
        # Fallback for older PyTorch versions
        from torch.cuda.amp import GradScaler, autocast
        HAS_CUDA_AMP = True
    except ImportError:
        HAS_CUDA_AMP = False
        GradScaler = None
        autocast = None

# Optional: PyTorch XLA for TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    HAS_XLA = True
except ImportError:
    HAS_XLA = False
    xm = None
    pl = None
    xmp = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# lm-evaluation-harness for benchmarking
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import simple_evaluate
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    lm_eval = None
    HFLM = None
    simple_evaluate = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    name_or_path: str = "gpt2"
    dtype: str = "float16"  # float16, bfloat16, float32
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    
    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.float16)


@dataclass
class SurrogateConfig:
    """Configuration for the surrogate model."""
    name_or_path: str = "Qwen/Qwen3-0.6B"
    dtype: str = "float16"
    # Probability threshold for token selection (replaces fixed k)
    # Selects all surrogate tokens with probability > prob_threshold
    prob_threshold: float = 0.03  # e.g., 0.03 selects tokens with >3% probability
    # Maximum tokens to consider per position (for memory efficiency)
    max_tokens: int = 100
    enabled: bool = True
    trust_remote_code: bool = False
    loss_weight_initial: float = 1.0  # Initial weight for surrogate loss
    loss_weight_final: float = 0.0    # Final weight for surrogate loss (after cosine decay)
    
    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.float16)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    max_seq_length: int = 1024
    preprocessing_num_workers: int = 4
    train_file: Optional[str] = None  # For custom datasets
    eval_file: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str = "./outputs"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    
    # Device selection: "auto", "cuda", "mps", "tpu", "cpu"
    device: str = "auto"
    
    # TPU-specific options
    tpu_cores: int = 1  # Number of TPU cores (1 for single, 8 for v3-8, etc.)
    tpu_metrics_debug: bool = False  # Print TPU metrics for debugging
    
    # Precision
    mixed_precision: str = "fp16"  # fp16, bf16, no
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Auxiliary loss
    use_z_loss: bool = False
    z_loss_multiplier: float = 1e-4
    
    # Distributed (for DDP on GPU)
    local_rank: int = -1
    
    # Misc
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    # W&B
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for lm-evaluation-harness benchmarks."""
    enabled: bool = True
    eval_interval: int = 1000  # Run benchmarks every N steps
    
    # Benchmark tasks to run (from lm-evaluation-harness)
    # Common options: "hellaswag", "arc_easy", "arc_challenge", "winogrande", 
    # "piqa", "boolq", "lambada_openai", "mmlu", "truthfulqa_mc"
    tasks: List[str] = field(default_factory=lambda: ["hellaswag", "arc_easy", "piqa"])
    
    # Number of few-shot examples (0 for zero-shot)
    num_fewshot: int = 0
    
    # Batch size for evaluation
    batch_size: int = 8
    
    # Limit number of examples per task (None for all)
    limit: Optional[int] = None
    
    # Log individual task scores
    log_individual_tasks: bool = True
    
    # Log aggregated score (mean across tasks)
    log_aggregate_score: bool = True


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load config from YAML. Install with: pip install pyyaml")
        
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**raw_config.get("model", {})),
            surrogate=SurrogateConfig(**raw_config.get("surrogate", {})),
            data=DataConfig(**raw_config.get("data", {})),
            training=TrainingConfig(**raw_config.get("training", {})),
            evaluation=EvaluationConfig(**raw_config.get("evaluation", {})),
        )
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create configuration from command line arguments."""
        config = cls()
        
        # Model config
        if args.base_model:
            config.model.name_or_path = args.base_model
        if args.model_dtype:
            config.model.dtype = args.model_dtype
        if args.gradient_checkpointing:
            config.model.gradient_checkpointing = True
        if args.trust_remote_code:
            config.model.trust_remote_code = True
            
        # Surrogate config
        if args.surrogate_model:
            config.surrogate.name_or_path = args.surrogate_model
        if args.surrogate_prob_threshold:
            config.surrogate.prob_threshold = args.surrogate_prob_threshold
        if args.surrogate_max_tokens:
            config.surrogate.max_tokens = args.surrogate_max_tokens
        if args.no_surrogate:
            config.surrogate.enabled = False
        if args.surrogate_dtype:
            config.surrogate.dtype = args.surrogate_dtype
            
        # Data config
        if args.dataset:
            config.data.dataset_name = args.dataset
        if args.dataset_config:
            config.data.dataset_config = args.dataset_config
        if args.text_column:
            config.data.text_column = args.text_column
        if args.max_seq_length:
            config.data.max_seq_length = args.max_seq_length
        if args.train_file:
            config.data.train_file = args.train_file
        if args.eval_file:
            config.data.eval_file = args.eval_file
            
        # Training config
        if args.output_dir:
            config.training.output_dir = args.output_dir
        if args.num_epochs:
            config.training.num_epochs = args.num_epochs
        if args.batch_size:
            config.training.per_device_train_batch_size = args.batch_size
        if args.gradient_accumulation_steps:
            config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
        if args.weight_decay:
            config.training.weight_decay = args.weight_decay
        if args.warmup_ratio:
            config.training.warmup_ratio = args.warmup_ratio
        if args.max_grad_norm:
            config.training.max_grad_norm = args.max_grad_norm
        if args.seed:
            config.training.seed = args.seed
        if args.mixed_precision:
            config.training.mixed_precision = args.mixed_precision
        if args.logging_steps:
            config.training.logging_steps = args.logging_steps
        if args.eval_steps:
            config.training.eval_steps = args.eval_steps
        if args.save_steps:
            config.training.save_steps = args.save_steps
        if args.wandb_project:
            config.training.wandb_project = args.wandb_project
        if args.wandb_run_name:
            config.training.wandb_run_name = args.wandb_run_name
        if args.resume_from_checkpoint:
            config.training.resume_from_checkpoint = args.resume_from_checkpoint
        if args.use_z_loss:
            config.training.use_z_loss = True
        if args.local_rank is not None:
            config.training.local_rank = args.local_rank
        if hasattr(args, 'device') and args.device:
            config.training.device = args.device
        if hasattr(args, 'tpu_cores') and args.tpu_cores:
            config.training.tpu_cores = args.tpu_cores
            
        return config
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            "model": vars(self.model),
            "surrogate": vars(self.surrogate),
            "data": vars(self.data),
            "training": vars(self.training),
            "evaluation": vars(self.evaluation),
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)


# =============================================================================
# Loss Functions
# =============================================================================

def surrogate_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    perp_values: Optional[torch.Tensor] = None,
    perp_indices: Optional[torch.Tensor] = None,
    lookup_surrogate_to_self: Optional[torch.Tensor] = None,
    surrogate_weight: float = 1.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Cross entropy loss with optional surrogate-guided auxiliary term.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size) or (batch_size * seq_len, vocab_size)
        labels: Target labels of shape (batch_size, seq_len) or (batch_size * seq_len,)
        perp_values: Perplexity values from surrogate model (batch_size, seq_len, max_tokens)
            Invalid entries are marked with inf.
        perp_indices: Token indices from surrogate model (batch_size, seq_len, max_tokens)
            Invalid entries are marked with -1.
        lookup_surrogate_to_self: Lookup table mapping surrogate vocab to base model vocab
        surrogate_weight: Weight for the surrogate loss term (decays over training)
        ignore_index: Index to ignore in loss computation
        reduction: "mean", "sum", or "none"
        compute_z_loss: Whether to compute auxiliary z-loss
        z_loss_multiplier: Multiplier for z-loss
        
    Returns:
        Tuple of (loss, z_loss) where z_loss may be None
    """
    device = logits.device
    
    if perp_indices is not None and lookup_surrogate_to_self is not None and surrogate_weight > 0:
        # Surrogate-guided loss computation
        if logits.dim() == 2:
            # Reshape from (batch*seq, vocab) to (batch, seq, vocab)
            # We need to infer the shape from perp_indices
            batch_size = perp_indices.shape[0]
            seq_len = perp_indices.shape[1]
            vocab_size = logits.shape[-1]
            logits = logits.view(batch_size, seq_len, vocab_size)
            labels = labels.view(batch_size, seq_len)
        
        batch_size, seq_len, vocab_size = logits.shape
        max_tokens = perp_indices.shape[-1]
        
        # Handle invalid indices (-1) by clamping first, then masking
        valid_indices_mask = perp_indices >= 0
        safe_perp_indices = perp_indices.clamp(min=0)
        
        # Translate surrogate indices to base model indices
        translated_perp_indices = lookup_surrogate_to_self[safe_perp_indices]
        translated_perp_indices[~valid_indices_mask] = -100  # Mark invalid
        
        # Handle invalid translations (-100 or out of bounds)
        valid_translation_mask = (translated_perp_indices >= 0) & (translated_perp_indices < vocab_size)
        safe_translated_indices = translated_perp_indices.clamp(0, vocab_size - 1)
        
        # Compute log probabilities over the FULL vocabulary first, then gather
        # This is the correct way - we need the actual model probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        gathered_log_probs = torch.gather(log_probs, dim=2, index=safe_translated_indices)
        gathered_nll = -gathered_log_probs  # Negative log likelihood
        
        # Combine validity masks:
        # 1. perp_values not inf (token was above threshold)
        # 2. perp_indices not -1 (valid index)
        # 3. Translation succeeded
        token_valid_mask = ~torch.isinf(perp_values)
        combined_mask = token_valid_mask & valid_indices_mask & valid_translation_mask
        
        # Count valid entries for normalization
        num_valid_surrogate_entries = combined_mask.sum().item()
        
        # Compute softmax weights from perplexity (lower perp = higher weight)
        # Mask invalid entries before softmax
        masked_perp_values = perp_values.clone()
        masked_perp_values[~combined_mask] = float('inf')
        
        # Handle edge case: if all values in a row are inf, softmax will produce NaN
        # Replace rows where all values are inf with zeros (no surrogate contribution)
        row_all_inf = torch.isinf(masked_perp_values).all(dim=-1, keepdim=True)
        masked_perp_values = masked_perp_values.masked_fill(row_all_inf.expand_as(masked_perp_values), 0.0)
        
        softmax_weights = F.softmax(-masked_perp_values, dim=-1)
        softmax_weights = softmax_weights * combined_mask.float()
        
        # Zero out NaN weights (safety check)
        softmax_weights = torch.nan_to_num(softmax_weights, nan=0.0)
        
        # Weighted surrogate loss (scaled by surrogate_weight)
        # Also zero out NaN in gathered_nll (from invalid indices)
        gathered_nll_safe = torch.nan_to_num(gathered_nll, nan=0.0, posinf=0.0, neginf=0.0)
        weighted_nll = gathered_nll_safe * softmax_weights
        surrogate_loss = surrogate_weight * weighted_nll.sum()
        
        # Standard cross-entropy loss
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        ce_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction='sum')
        
        # Combine losses
        num_valid_labels = (labels != ignore_index).sum().item()
        
        # Normalization: CE loss normalized by labels, surrogate loss normalized by surrogate entries
        # Then combined (surrogate already scaled by surrogate_weight)
        if reduction == "mean":
            ce_loss_normalized = ce_loss / (num_valid_labels + 1e-8)
            surrogate_loss_normalized = surrogate_loss / (num_valid_surrogate_entries + 1e-8)
            loss = ce_loss_normalized + surrogate_loss_normalized
        elif reduction == "sum":
            loss = ce_loss + surrogate_loss
        else:  # reduction == "none"
            # Return per-token loss (reshape surrogate loss appropriately)
            ce_loss_per_token = F.cross_entropy(
                logits_flat, labels_flat, ignore_index=ignore_index, reduction='none'
            ).view(batch_size, seq_len)
            surr_loss_per_token = surrogate_weight * weighted_nll.sum(dim=-1)
            loss = ce_loss_per_token + surr_loss_per_token
        
    else:
        # Standard cross-entropy loss
        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    
    # Compute z-loss if requested
    z_loss = None
    if compute_z_loss:
        if logits.dim() == 2:
            z_squared = logits.logsumexp(-1).pow(2)
        else:
            z_squared = logits.view(-1, logits.size(-1)).logsumexp(-1).pow(2)
        
        if labels.dim() > 1:
            label_mask = (labels.view(-1) != ignore_index).float()
        else:
            label_mask = (labels != ignore_index).float()
            
        if reduction == "mean":
            z_loss = z_loss_multiplier * (z_squared * label_mask).sum() / (label_mask.sum() + 1e-8)
        elif reduction == "sum":
            z_loss = z_loss_multiplier * (z_squared * label_mask).sum()
        else:
            z_loss = z_loss_multiplier * z_squared
    
    return loss, z_loss


# =============================================================================
# Metrics and FLOPS Computation
# =============================================================================

@dataclass
class SpeedMetrics:
    """Track training speed and throughput metrics."""
    start_time: float = 0.0
    tokens_seen: int = 0
    batches_seen: int = 0
    total_tokens: int = 0  # Cumulative across all training
    
    def reset(self) -> None:
        """Reset metrics for a new logging interval."""
        self.start_time = time.time()
        self.tokens_seen = 0
        self.batches_seen = 0
    
    def update(self, batch_tokens: int) -> None:
        """Update metrics with a new batch."""
        self.tokens_seen += batch_tokens
        self.batches_seen += 1
        self.total_tokens += batch_tokens
    
    def get_metrics(self, world_size: int = 1) -> Dict[str, float]:
        """Compute throughput metrics."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            elapsed = 1e-8
        
        return {
            "tokens_per_second": self.tokens_seen / elapsed,
            "tokens_per_second_global": (self.tokens_seen * world_size) / elapsed,
            "batches_per_second": self.batches_seen / elapsed,
            "total_tokens_trained": self.total_tokens * world_size,
        }


def estimate_model_flops(
    model: PreTrainedModel,
    batch_size: int,
    seq_length: int,
) -> Tuple[int, int]:
    """
    Estimate FLOPs for a transformer model forward and backward pass.
    
    Based on: https://arxiv.org/abs/2001.08361 (Scaling Laws for Neural Language Models)
    Approximate FLOPs per token ≈ 6 * num_params (forward + backward)
    Forward only ≈ 2 * num_params per token
    
    Returns:
        Tuple of (flops_per_token, total_flops_per_batch)
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Approximate FLOPs: 2 * params for forward, 4 * params for backward (2x forward)
    # Total = 6 * params per token
    flops_per_token_forward = 2 * num_params
    flops_per_token_backward = 4 * num_params
    flops_per_token_total = flops_per_token_forward + flops_per_token_backward
    
    # Total FLOPs per batch
    total_tokens = batch_size * seq_length
    total_flops = flops_per_token_total * total_tokens
    
    return flops_per_token_total, total_flops


# =============================================================================
# Token Vocabulary Alignment
# =============================================================================

class VocabularyAligner:
    """Handles vocabulary alignment between base and surrogate models."""
    
    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizer,
        surrogate_tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        self.base_tokenizer = base_tokenizer
        self.surrogate_tokenizer = surrogate_tokenizer
        self.device = device
        
        # Build lookup tables
        self._build_lookup_tables()
        
    def _build_lookup_tables(self) -> None:
        """Build bidirectional lookup tables between vocabularies."""
        base_vocab = self.base_tokenizer.get_vocab()
        surrogate_vocab = self.surrogate_tokenizer.get_vocab()
        
        base_tokens = set(base_vocab.keys())
        surrogate_tokens = set(surrogate_vocab.keys())
        
        # Find intersection
        intersection = base_tokens.intersection(surrogate_tokens)
        logger.info(f"Vocabulary intersection size: {len(intersection)} tokens")
        logger.info(f"Base vocab size: {len(base_tokens)}, Surrogate vocab size: {len(surrogate_tokens)}")
        
        # Create lookup tables
        base_vocab_size = len(base_vocab)
        surrogate_vocab_size = len(surrogate_vocab)
        
        # Build index arrays on CPU first (much faster than individual GPU assignments)
        base_ids = []
        surrogate_ids = []
        for token in intersection:
            base_ids.append(base_vocab[token])
            surrogate_ids.append(surrogate_vocab[token])
        
        base_ids_tensor = torch.tensor(base_ids, dtype=torch.long)
        surrogate_ids_tensor = torch.tensor(surrogate_ids, dtype=torch.long)
        
        # Create lookup tables on CPU, then move to device
        # Base -> Surrogate
        self.lookup_base_to_surrogate = torch.full(
            (base_vocab_size,), fill_value=-100, dtype=torch.long
        )
        self.lookup_base_to_surrogate[base_ids_tensor] = surrogate_ids_tensor
        self.lookup_base_to_surrogate = self.lookup_base_to_surrogate.to(self.device)
        
        # Surrogate -> Base
        self.lookup_surrogate_to_base = torch.full(
            (surrogate_vocab_size,), fill_value=-100, dtype=torch.long
        )
        self.lookup_surrogate_to_base[surrogate_ids_tensor] = base_ids_tensor
        self.lookup_surrogate_to_base = self.lookup_surrogate_to_base.to(self.device)
        
        # Store permitted tokens
        self.base_permitted_ids = base_ids_tensor.to(self.device)
        self.surrogate_permitted_ids = surrogate_ids_tensor.to(self.device)
        
        logger.info("Lookup tables built successfully.")
        
    def translate_base_to_surrogate(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Translate base model token IDs to surrogate model token IDs."""
        return self.lookup_base_to_surrogate[token_ids.clamp(0, len(self.lookup_base_to_surrogate) - 1)]
    
    def translate_surrogate_to_base(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Translate surrogate model token IDs to base model token IDs."""
        return self.lookup_surrogate_to_base[token_ids.clamp(0, len(self.lookup_surrogate_to_base) - 1)]


# =============================================================================
# Dataset
# =============================================================================

class TextDataset(Dataset):
    """Dataset for language model training."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        logger.info("Tokenizing dataset...")
        self.examples = []
        for text in tqdm(texts, desc="Tokenizing"):
            if not text.strip():
                continue
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            if len(encoded["input_ids"]) > 1:
                self.examples.append(encoded["input_ids"])
        
        logger.info(f"Created dataset with {len(self.examples)} examples")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor(self.examples[idx], dtype=torch.long)}


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Collate function with padding."""
    input_ids = [item["input_ids"] for item in batch]
    
    # Pad to max length in batch
    max_len = min(max(len(ids) for ids in input_ids), max_length)
    
    padded_input_ids = []
    attention_masks = []
    
    for ids in input_ids:
        if len(ids) < max_len:
            padding_length = max_len - len(ids)
            padded_ids = torch.cat([ids, torch.full((padding_length,), pad_token_id, dtype=torch.long)])
            mask = torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(padding_length, dtype=torch.long)])
        else:
            padded_ids = ids[:max_len]
            mask = torch.ones(max_len, dtype=torch.long)
        
        padded_input_ids.append(padded_ids)
        attention_masks.append(mask)
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(attention_masks),
    }


class CollateFunction:
    """Picklable collate function wrapper for multiprocessing DataLoader."""
    
    def __init__(self, pad_token_id: int, max_length: int):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return collate_fn(batch, self.pad_token_id, self.max_length)


def load_training_data(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Load training and validation datasets."""
    
    if config.train_file:
        # Load from local files
        logger.info(f"Loading data from local files: {config.train_file}")
        
        with open(config.train_file, "r") as f:
            if config.train_file.endswith(".json"):
                data = json.load(f)
                train_texts = [item.get(config.text_column, item) for item in data]
            else:
                train_texts = f.readlines()
        
        eval_texts = None
        if config.eval_file:
            with open(config.eval_file, "r") as f:
                if config.eval_file.endswith(".json"):
                    data = json.load(f)
                    eval_texts = [item.get(config.text_column, item) for item in data]
                else:
                    eval_texts = f.readlines()
    else:
        # Load from HuggingFace datasets
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        logger.info(f"Loading dataset: {config.dataset_name} ({config.dataset_config})")
        
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            trust_remote_code=True,
        )
        
        train_texts = dataset[config.dataset_split][config.text_column]
        eval_texts = None
        if config.eval_split in dataset:
            eval_texts = dataset[config.eval_split][config.text_column]
    
    train_dataset = TextDataset(train_texts, tokenizer, config.max_seq_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, config.max_seq_length) if eval_texts else None
    
    return train_dataset, eval_dataset


# =============================================================================
# Trainer
# =============================================================================

class SurrogateTrainer:
    """
    Trainer class implementing surrogate-assisted language model training.
    """
    
    def __init__(
        self,
        config: Config,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        surrogate_model: Optional[PreTrainedModel] = None,
        surrogate_tokenizer: Optional[PreTrainedTokenizer] = None,
        tpu_rank: Optional[int] = None,  # For TPU multi-core training
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.surrogate_model = surrogate_model
        self.surrogate_tokenizer = surrogate_tokenizer
        self.tpu_rank = tpu_rank
        
        # Setup device (TPU, GPU, MPS, or CPU)
        self.device = self._setup_device()
        self.is_tpu = self.device.type == 'xla' if HAS_XLA else False
        
        # Setup distributed training (GPU DDP or TPU)
        self.is_distributed = config.training.local_rank != -1 or (self.is_tpu and config.training.tpu_cores > 1)
        self.world_size = 1
        self.global_rank = 0
        self.local_rank = 0
        
        if self.is_tpu and config.training.tpu_cores > 1:
            self._setup_tpu_distributed()
        elif config.training.local_rank != -1:
            self._setup_distributed()
        
        # Move models to device
        logger.info(f"Moving models to device: {self.device}")
        self.model = self.model.to(self.device)
        if self.surrogate_model is not None:
            self.surrogate_model = self.surrogate_model.to(self.device)
            self.surrogate_model.eval()
        
        # Setup vocabulary alignment
        self.vocab_aligner = None
        if self.surrogate_model is not None and self.surrogate_tokenizer is not None:
            logger.info("Building vocabulary alignment...")
            self.vocab_aligner = VocabularyAligner(
                self.tokenizer,
                self.surrogate_tokenizer,
                self.device,
            )
            logger.info("Vocabulary alignment complete.")
        
        # Setup data loaders
        logger.info("Setting up data loaders...")
        self._setup_dataloaders()
        logger.info("Data loaders ready.")
        
        # Setup optimizer and scheduler
        logger.info("Setting up optimizer and scheduler...")
        self._setup_optimizer()
        
        # Setup mixed precision (not used for TPU - TPU handles precision internally)
        self._setup_mixed_precision()
        logger.info("Trainer initialization complete.")
        
        # Distributed model wrapper (GPU DDP only, not needed for TPU)
        if config.training.local_rank != -1 and not self.is_tpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        
        # Speed and FLOPS metrics
        self.speed_metrics = SpeedMetrics()
        self.flops_per_token, self.flops_per_batch = estimate_model_flops(
            self.model.module if hasattr(self.model, 'module') else self.model,
            config.training.per_device_train_batch_size,
            config.data.max_seq_length,
        )
        logger.info(f"Estimated FLOPs per token: {self.flops_per_token:,}")
        logger.info(f"Estimated FLOPs per batch: {self.flops_per_batch:,}")
        
        # Setup logging
        self._setup_logging()
        
    def _setup_device(self) -> torch.device:
        """Setup compute device based on configuration and availability."""
        device_type = self.config.training.device.lower()
        
        if device_type == "tpu":
            if not HAS_XLA:
                raise RuntimeError("PyTorch XLA is required for TPU. Install with: pip install torch-xla")
            device = xm.xla_device()
            logger.info(f"Using TPU device: {device}")
            return device
        
        elif device_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if self.config.training.local_rank != -1:
                return torch.device("cuda", self.config.training.local_rank)
            return torch.device("cuda")
        
        elif device_type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            return torch.device("mps")
        
        elif device_type == "cpu":
            return torch.device("cpu")
        
        elif device_type == "auto":
            # Auto-detect best available device
            if HAS_XLA:
                try:
                    # Check if we're actually running on TPU
                    device = xm.xla_device()
                    # This will fail if not on TPU
                    _ = torch.zeros(1, device=device)
                    logger.info(f"Auto-detected TPU device: {device}")
                    return device
                except Exception:
                    pass
            
            if torch.cuda.is_available():
                if self.config.training.local_rank != -1:
                    return torch.device("cuda", self.config.training.local_rank)
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        
        else:
            raise ValueError(f"Unknown device type: {device_type}")
    
    def _setup_tpu_distributed(self) -> None:
        """Setup distributed training for TPU."""
        if not HAS_XLA:
            raise RuntimeError("PyTorch XLA required for TPU distributed training")
        
        self.world_size = xm.xrt_world_size()
        self.global_rank = xm.get_ordinal()
        self.local_rank = xm.get_local_ordinal()
        logger.info(f"TPU distributed training: rank {self.global_rank}/{self.world_size}")
    
    def _setup_distributed(self) -> None:
        """Setup distributed training for GPU (DDP)."""
        dist.init_process_group(backend="nccl")
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.local_rank = self.config.training.local_rank
        torch.cuda.set_device(self.local_rank)
        logger.info(f"GPU distributed training: rank {self.global_rank}/{self.world_size}")
    
    def _setup_dataloaders(self) -> None:
        """Setup training and evaluation data loaders."""
        train_sampler = None
        
        # For TPU distributed training
        if self.is_tpu and self.config.training.tpu_cores > 1:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True,
            )
        # For GPU distributed training
        elif self.is_distributed and not self.is_tpu:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        # Don't use pin_memory for TPU
        pin_memory = not self.is_tpu
        
        # For TPU, use fewer workers to avoid issues
        num_workers = 0 if self.is_tpu else self.config.data.preprocessing_num_workers
        
        # Create picklable collate function for multiprocessing
        collate = CollateFunction(pad_token_id, self.config.data.max_seq_length)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.per_device_train_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        
        # Wrap with TPU parallel loader for efficient data transfer
        if self.is_tpu and HAS_XLA:
            self.train_loader = pl.MpDeviceLoader(self.train_loader, self.device)
        
        self.eval_loader = None
        if self.eval_dataset is not None:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.training.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            # Wrap eval loader for TPU
            if self.is_tpu and HAS_XLA:
                self.eval_loader = pl.MpDeviceLoader(eval_loader, self.device)
            else:
                self.eval_loader = eval_loader
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Separate parameters with and without weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # Calculate total training steps
        num_update_steps_per_epoch = len(self.train_loader) // self.config.training.gradient_accumulation_steps
        self.total_training_steps = num_update_steps_per_epoch * self.config.training.num_epochs
        
        # Warmup steps
        if self.config.training.warmup_steps is not None:
            warmup_steps = self.config.training.warmup_steps
        else:
            warmup_steps = int(self.total_training_steps * self.config.training.warmup_ratio)
        
        self.scheduler = get_scheduler(
            name=self.config.training.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_training_steps,
        )
        
        logger.info(f"Total training steps: {self.total_training_steps}, Warmup steps: {warmup_steps}")
    
    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training."""
        self.scaler = None
        self.autocast_dtype = torch.float32
        self.use_autocast = False
        
        # TPU handles precision internally via XLA, no GradScaler needed
        if self.is_tpu:
            if self.config.training.mixed_precision == "bf16":
                # TPU v3+ supports bfloat16 natively
                self.autocast_dtype = torch.bfloat16
                self.use_autocast = True
                logger.info("TPU: Using bfloat16 precision")
            else:
                # TPU defaults to float32, XLA optimizes internally
                logger.info("TPU: Using float32 precision (XLA will optimize)")
            return
        
        # CUDA mixed precision
        if self.device.type == "cuda":
            if self.config.training.mixed_precision == "fp16" and HAS_CUDA_AMP:
                # Only use GradScaler for fp16 on CUDA
                # GradScaler auto-detects the device
                self.scaler = GradScaler()
                self.autocast_dtype = torch.float16
                self.use_autocast = True
                logger.info("CUDA: Using fp16 mixed precision with GradScaler")
            elif self.config.training.mixed_precision == "bf16":
                # bfloat16 doesn't need GradScaler (no loss scaling needed)
                self.autocast_dtype = torch.bfloat16
                self.use_autocast = True
                logger.info("CUDA: Using bf16 mixed precision (no GradScaler)")
            else:
                logger.info("CUDA: Using float32 precision")
        
        # MPS mixed precision (limited support, no GradScaler)
        elif self.device.type == "mps":
            if self.config.training.mixed_precision == "fp16":
                # MPS supports fp16 autocast but not GradScaler
                self.autocast_dtype = torch.float16
                self.use_autocast = True
                logger.info("MPS: Using fp16 autocast (no GradScaler)")
            elif self.config.training.mixed_precision == "bf16":
                self.autocast_dtype = torch.bfloat16
                self.use_autocast = True
                logger.info("MPS: Using bf16 autocast (no GradScaler)")
            else:
                logger.info("MPS: Using float32 precision")
        
        # CPU - no mixed precision benefits
        elif self.device.type == "cpu":
            if self.config.training.mixed_precision not in ("no", "fp32", None):
                logger.warning("CPU: Mixed precision has limited benefit, using float32")
            self.use_autocast = False
            self.autocast_dtype = torch.float32
    
    def _setup_logging(self) -> None:
        """Setup logging with Weights & Biases."""
        if self.global_rank == 0 and HAS_WANDB and self.config.training.wandb_project:
            wandb.init(
                project=self.config.training.wandb_project,
                name=self.config.training.wandb_run_name,
                entity=self.config.training.wandb_entity,
                config={
                    "model": vars(self.config.model),
                    "surrogate": vars(self.config.surrogate),
                    "data": vars(self.config.data),
                    "training": vars(self.config.training),
                },
            )
    
    def get_labels(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create labels from input_ids (shifted left by 1)."""
        labels = batch["input_ids"].clone()
        
        # Mask padding tokens
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        
        # Shift for next-token prediction
        labels = labels[..., 1:].contiguous()
        
        return labels
    
    def get_surrogate_loss_weight(self) -> float:
        """
        Compute the current surrogate loss weight using cosine decay (no warmup).
        
        The weight decays from loss_weight_initial to loss_weight_final over
        the course of training following a cosine schedule:
        
        weight = final + 0.5 * (initial - final) * (1 + cos(pi * step / total_steps))
        
        Returns:
            Current surrogate loss weight
        """
        if not self.config.surrogate.enabled:
            return 0.0
        
        initial = self.config.surrogate.loss_weight_initial
        final = self.config.surrogate.loss_weight_final
        
        if self.total_training_steps == 0:
            return initial
        
        # Cosine decay without warmup
        progress = min(self.global_step / self.total_training_steps, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        weight = final + (initial - final) * cosine_decay
        return weight
    
    def compute_surrogate_guidance(
        self,
        batch: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute probability threshold-based guidance from surrogate model.
        
        Instead of selecting a fixed top-k tokens, this method selects all tokens
        whose probability exceeds the configured threshold.
        
        Returns:
            Tuple of (perp_values, perp_indices) or (None, None) if surrogate is disabled
            - perp_values: shape (batch_size, seq_len-1, max_tokens), inf for invalid entries
            - perp_indices: shape (batch_size, seq_len-1, max_tokens), -1 for invalid entries
        """
        if self.surrogate_model is None or not self.config.surrogate.enabled:
            return None, None
        
        prob_threshold = self.config.surrogate.prob_threshold
        max_tokens = self.config.surrogate.max_tokens
        
        with torch.no_grad():
            input_ids = batch["input_ids"]  # (batch_size, seq_len)
            attention_mask = batch.get("attention_mask")
            batch_size, seq_len = input_ids.shape
            
            # === DIRECT GPU TOKEN TRANSLATION (no decode/re-encode) ===
            # Translate base model token IDs to surrogate token IDs directly
            surrogate_input_ids = self.vocab_aligner.lookup_base_to_surrogate[input_ids]
            
            # Create mask for untranslatable tokens (those not in vocabulary intersection)
            untranslatable_mask = (surrogate_input_ids == -100)
            
            # Replace untranslatable tokens with surrogate's pad token for forward pass
            surrogate_pad_id = self.surrogate_tokenizer.pad_token_id
            if surrogate_pad_id is None:
                surrogate_pad_id = self.surrogate_tokenizer.eos_token_id
            
            surrogate_input_ids_safe = surrogate_input_ids.clone()
            surrogate_input_ids_safe[untranslatable_mask] = surrogate_pad_id
            
            # Create attention mask: mask out untranslatable tokens
            if attention_mask is not None:
                surrogate_attention_mask = attention_mask.clone()
                surrogate_attention_mask[untranslatable_mask] = 0
            else:
                surrogate_attention_mask = (~untranslatable_mask).long()
            
            # Forward pass through surrogate model (all on GPU)
            surrogate_outputs = self.surrogate_model(
                input_ids=surrogate_input_ids_safe,
                attention_mask=surrogate_attention_mask,
            )
            # Shift logits to align with labels (next-token prediction)
            # surrogate_logits: (batch_size, seq_len-1, vocab_size)
            surrogate_logits = surrogate_outputs.logits[..., :-1, :].contiguous()
            
            surr_vocab_size = surrogate_logits.shape[-1]
            label_seq_len = labels.shape[1]  # seq_len - 1 due to shift in get_labels
            
            # Compute probabilities from logits
            surrogate_probs = F.softmax(surrogate_logits, dim=-1)
            # Compute perplexity (inverse probability) - lower perp = higher confidence
            surrogate_perp = torch.reciprocal(surrogate_probs + 1e-8)
            # Shape: (batch_size, seq_len-1, surr_vocab_size)
            
            # === MASKING === 
            # 1. Mask out tokens not in vocabulary intersection (can't translate back)
            vocab_mask = ~torch.isin(
                torch.arange(surr_vocab_size, device=self.device),
                self.vocab_aligner.surrogate_permitted_ids,
            )
            surrogate_probs[:, :, vocab_mask] = 0.0
            surrogate_perp[:, :, vocab_mask] = float('inf')
            
            # 2. Mask positions corresponding to labels=-100 (padding/ignored)
            invalid_label_positions = (labels == -100).unsqueeze(-1)
            surrogate_probs = surrogate_probs.masked_fill(invalid_label_positions, 0.0)
            surrogate_perp = surrogate_perp.masked_fill(invalid_label_positions, float('inf'))
            
            # 3. Mask positions where the INPUT token was untranslatable
            #    (shifted by 1 to align with logits/labels which predict next token)
            untranslatable_positions = untranslatable_mask[:, 1:].unsqueeze(-1)
            surrogate_probs = surrogate_probs.masked_fill(untranslatable_positions, 0.0)
            surrogate_perp = surrogate_perp.masked_fill(untranslatable_positions, float('inf'))
            
            # 4. Mask out the actual target token (we don't want it in selection)
            translated_labels = self.vocab_aligner.translate_base_to_surrogate(labels)
            valid_label_mask = translated_labels != -100
            
            batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, label_seq_len)
            seq_indices = torch.arange(label_seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            
            # Use 0 for invalid labels to avoid indexing errors (they're already masked anyway)
            safe_translated_labels = translated_labels.clone()
            safe_translated_labels[~valid_label_mask] = 0
            surrogate_probs[batch_indices, seq_indices, safe_translated_labels] = 0.0
            surrogate_perp[batch_indices, seq_indices, safe_translated_labels] = float('inf')
            
            # === PROBABILITY THRESHOLD SELECTION ===
            # Mask tokens below the probability threshold
            below_threshold_mask = surrogate_probs <= prob_threshold
            surrogate_perp[below_threshold_mask] = float('inf')
            
            # Select up to max_tokens with lowest perplexity (highest probability above threshold)
            topk_result = torch.topk(surrogate_perp, k=max_tokens, largest=False, sorted=True, dim=-1)
            
            perp_values = topk_result.values   # (batch_size, seq_len-1, max_tokens)
            perp_indices = topk_result.indices  # (batch_size, seq_len-1, max_tokens) - in surrogate vocab
            
            # Mark indices as invalid (-1) where perplexity is inf (below threshold or invalid)
            invalid_mask = torch.isinf(perp_values)
            perp_indices = perp_indices.masked_fill(invalid_mask, -1)
            
        return perp_values, perp_indices
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Execute a single training step."""
        # For TPU with MpDeviceLoader, data is already on device
        if not self.is_tpu:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get labels
        labels = self.get_labels(batch)
        
        # Compute surrogate guidance
        perp_values, perp_indices = self.compute_surrogate_guidance(batch, labels)
        
        # Forward pass with appropriate autocast context
        if self.use_autocast and not self.is_tpu:
            # CUDA/MPS autocast - use torch.amp.autocast
            autocast_context = torch.amp.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
        elif self.is_tpu and self.autocast_dtype == torch.bfloat16:
            # TPU bfloat16 - use autocast if available in torch_xla
            autocast_context = torch.amp.autocast('xla', dtype=torch.bfloat16, enabled=True)
        else:
            # No autocast
            from contextlib import nullcontext
            autocast_context = nullcontext()
        
        with autocast_context:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            
            # Shift logits for next-token prediction
            logits = outputs.logits[..., :-1, :].contiguous()
            
            # Flatten for loss computation
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            
            # Get current surrogate loss weight (cosine decay)
            surrogate_weight = self.get_surrogate_loss_weight()
            
            # Compute loss
            lookup_table = None
            if self.vocab_aligner is not None:
                lookup_table = self.vocab_aligner.lookup_surrogate_to_base
            
            ce_loss, z_loss = surrogate_cross_entropy_loss(
                logits=logits_flat if perp_indices is None else logits,
                labels=labels_flat if perp_indices is None else labels,
                perp_values=perp_values,
                perp_indices=perp_indices,
                lookup_surrogate_to_self=lookup_table,
                surrogate_weight=surrogate_weight,
                ignore_index=-100,
                reduction="mean",
                compute_z_loss=self.config.training.use_z_loss,
                z_loss_multiplier=self.config.training.z_loss_multiplier,
            )
            
            loss = ce_loss
            if z_loss is not None:
                loss = loss + z_loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        metrics = {
            "loss": ce_loss.item(),
            "perplexity": math.exp(min(ce_loss.item(), 20)),
            "surrogate_weight": surrogate_weight,
        }
        if z_loss is not None:
            metrics["z_loss"] = z_loss.item()
        
        # Track number of auxiliary tokens selected by surrogate
        if perp_indices is not None and perp_values is not None:
            # Count valid tokens (not -1 index and not inf perplexity)
            valid_mask = (perp_indices >= 0) & (~torch.isinf(perp_values))
            num_aux_tokens = valid_mask.sum().item()
            # Also compute average per position (for positions that have at least one token)
            num_positions = perp_indices.shape[0] * perp_indices.shape[1]
            positions_with_tokens = valid_mask.any(dim=-1).sum().item()
            
            metrics["aux_tokens_total"] = num_aux_tokens
            metrics["aux_tokens_per_position"] = num_aux_tokens / (num_positions + 1e-8)
            metrics["positions_with_aux_tokens"] = positions_with_tokens
            metrics["positions_with_aux_tokens_pct"] = 100.0 * positions_with_tokens / (num_positions + 1e-8)
        
        return metrics
    
    def optimizer_step(self) -> Dict[str, float]:
        """Execute optimizer step with gradient clipping."""
        metrics = {}
        
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        if self.is_tpu:
            # TPU gradient clipping
            grad_norm = xm.optimizer_step(self.optimizer, optimizer_args={'max_grad_norm': self.config.training.max_grad_norm})
            # xm.optimizer_step doesn't return grad_norm, compute it separately if needed
            xm.mark_step()
            metrics["grad_norm"] = 0.0  # TPU doesn't easily expose this
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            )
            metrics["grad_norm"] = grad_norm.item()
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation loop."""
        if self.eval_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        # Set up autocast context
        if self.use_autocast and not self.is_tpu:
            autocast_context = torch.amp.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
        elif self.is_tpu and self.autocast_dtype == torch.bfloat16:
            autocast_context = torch.amp.autocast('xla', dtype=torch.bfloat16, enabled=True)
        else:
            from contextlib import nullcontext
            autocast_context = nullcontext()
        
        for batch in tqdm(self.eval_loader, desc="Evaluating", disable=self.global_rank != 0):
            # For TPU with MpDeviceLoader, data is already on device
            if not self.is_tpu:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = self.get_labels(batch)
            
            with autocast_context:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
                
                logits = outputs.logits[..., :-1, :].contiguous()
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction="sum")
            
            num_tokens = (labels_flat != -100).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
            
            # Mark step for TPU
            if self.is_tpu:
                xm.mark_step()
        
        # Aggregate across ranks
        if self.is_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_tokens_tensor = torch.tensor(total_tokens, device=self.device)
            
            if self.is_tpu:
                # TPU all-reduce
                total_loss_tensor = xm.all_reduce(xm.REDUCE_SUM, total_loss_tensor)
                total_tokens_tensor = xm.all_reduce(xm.REDUCE_SUM, total_tokens_tensor)
            else:
                # GPU all-reduce
                dist.all_reduce(total_loss_tensor)
                dist.all_reduce(total_tokens_tensor)
            
            total_loss = total_loss_tensor.item()
            total_tokens = total_tokens_tensor.item()
        
        avg_loss = total_loss / (total_tokens + 1e-8)
        perplexity = math.exp(min(avg_loss, 20))
        
        self.model.train()
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }
    
    def run_benchmarks(self) -> Dict[str, float]:
        """
        Run lm-evaluation-harness benchmarks and return scores.
        
        Returns:
            Dictionary of benchmark scores
        """
        if not self.config.evaluation.enabled:
            return {}
        
        if not HAS_LM_EVAL:
            logger.warning(
                "lm-evaluation-harness not installed. "
                "Install with: pip install lm-eval"
            )
            return {}
        
        if self.global_rank != 0:
            # Only run benchmarks on main process
            return {}
        
        logger.info(f"Running benchmarks: {self.config.evaluation.tasks}")
        
        self.model.eval()
        
        try:
            # Get the underlying model (unwrap DDP if necessary)
            model_to_eval = self.model.module if hasattr(self.model, "module") else self.model
            
            # Create lm-eval model wrapper
            lm_model = HFLM(
                pretrained=model_to_eval,
                tokenizer=self.tokenizer,
                batch_size=self.config.evaluation.batch_size,
                device=str(self.device),
            )
            
            # Run evaluation
            results = simple_evaluate(
                model=lm_model,
                tasks=self.config.evaluation.tasks,
                num_fewshot=self.config.evaluation.num_fewshot,
                limit=self.config.evaluation.limit,
                log_samples=False,
            )
            
            # Extract metrics
            benchmark_metrics = {}
            task_scores = []
            
            for task_name, task_results in results.get("results", {}).items():
                # Get the main accuracy/score metric
                if "acc" in task_results:
                    score = task_results["acc"]
                    metric_name = "acc"
                elif "acc_norm" in task_results:
                    score = task_results["acc_norm"]
                    metric_name = "acc_norm"
                elif "exact_match" in task_results:
                    score = task_results["exact_match"]
                    metric_name = "exact_match"
                else:
                    # Take first numeric metric
                    for k, v in task_results.items():
                        if isinstance(v, (int, float)) and not k.endswith("_stderr"):
                            score = v
                            metric_name = k
                            break
                    else:
                        continue
                
                task_scores.append(score)
                
                if self.config.evaluation.log_individual_tasks:
                    benchmark_metrics[f"benchmark/{task_name}/{metric_name}"] = score
                    
                    # Also log stderr if available
                    stderr_key = f"{metric_name}_stderr"
                    if stderr_key in task_results:
                        benchmark_metrics[f"benchmark/{task_name}/{stderr_key}"] = task_results[stderr_key]
            
            # Compute aggregate score
            if self.config.evaluation.log_aggregate_score and task_scores:
                benchmark_metrics["benchmark/average_score"] = sum(task_scores) / len(task_scores)
            
            logger.info(f"Benchmark results: {benchmark_metrics}")
            
        except Exception as e:
            logger.error(f"Error running benchmarks: {e}")
            benchmark_metrics = {}
        
        self.model.train()
        
        return benchmark_metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        if self.global_rank != 0:
            return
        
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }
        if self.scaler is not None:
            state["scaler_state"] = self.scaler.state_dict()
        
        torch.save(state, os.path.join(path, "training_state.pt"))
        self.config.save(os.path.join(path, "config.json"))
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        state_path = os.path.join(path, "training_state.pt")
        if not os.path.exists(state_path):
            logger.warning(f"No training state found at {state_path}")
            return
        
        state = torch.load(state_path, map_location=self.device)
        
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_eval_loss = state["best_eval_loss"]
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        
        if self.scaler is not None and "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])
        
        # Restore RNG states
        rng_state = state["rng_state"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state["cuda"] is not None:
            torch.cuda.set_rng_state(rng_state["cuda"])
        
        logger.info(f"Checkpoint loaded from {path}, resuming from step {self.global_step}")
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"  Num epochs: {self.config.training.num_epochs}")
        logger.info(f"  Batch size per device: {self.config.training.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps: {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps: {self.total_training_steps}")
        logger.info(f"  Surrogate model enabled: {self.config.surrogate.enabled}")
        if self.config.surrogate.enabled:
            logger.info(f"  Surrogate model: {self.config.surrogate.name_or_path}")
            logger.info(f"  Surrogate prob_threshold: {self.config.surrogate.prob_threshold}")
        if self.config.evaluation.enabled:
            logger.info(f"  Benchmark tasks: {self.config.evaluation.tasks}")
            logger.info(f"  Benchmark eval interval: {self.config.evaluation.eval_interval}")
        
        # Resume from checkpoint if specified
        if self.config.training.resume_from_checkpoint:
            self.load_checkpoint(self.config.training.resume_from_checkpoint)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Initialize speed metrics
        self.speed_metrics.reset()
        running_loss = 0.0
        running_grad_norm = 0.0
        running_steps = 0
        running_aux_tokens = 0.0
        running_aux_positions = 0.0
        running_aux_positions_pct = 0.0
        
        # Compute batch tokens
        batch_tokens = (
            self.config.training.per_device_train_batch_size 
            * self.config.data.max_seq_length
        )
        global_batch_tokens = batch_tokens * self.world_size * self.config.training.gradient_accumulation_steps
        
        for epoch in range(self.epoch, self.config.training.num_epochs):
            self.epoch = epoch
            
            if self.is_distributed and hasattr(self.train_loader, 'sampler'):
                self.train_loader.sampler.set_epoch(epoch)
            
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}",
                disable=self.global_rank != 0,
            )
            
            for step, batch in enumerate(progress_bar):
                # Training step
                metrics = self.train_step(batch)
                running_loss += metrics["loss"]
                running_steps += 1
                
                # Accumulate auxiliary token metrics
                if "aux_tokens_per_position" in metrics:
                    running_aux_tokens += metrics["aux_tokens_per_position"]
                    running_aux_positions += metrics["positions_with_aux_tokens"]
                    running_aux_positions_pct += metrics["positions_with_aux_tokens_pct"]
                
                # Update speed metrics
                self.speed_metrics.update(batch_tokens)
                
                # Optimizer step (with gradient accumulation)
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    optim_metrics = self.optimizer_step()
                    self.global_step += 1
                    running_grad_norm += optim_metrics["grad_norm"]
                    
                    # Logging
                    if self.global_step % self.config.training.logging_steps == 0:
                        avg_loss = running_loss / running_steps
                        avg_grad_norm = running_grad_norm / (self.global_step % self.config.training.logging_steps or self.config.training.logging_steps)
                        
                        # Get speed metrics
                        speed = self.speed_metrics.get_metrics(self.world_size)
                        
                        # Compute FLOPS metrics
                        flops_per_second = speed["tokens_per_second_global"] * self.flops_per_token
                        flops_per_second_per_device = speed["tokens_per_second"] * self.flops_per_token
                        total_flops = self.speed_metrics.total_tokens * self.world_size * self.flops_per_token
                        
                        log_metrics = {
                            # Training metrics
                            "train/loss": avg_loss,
                            "train/perplexity": math.exp(min(avg_loss, 20)),
                            "train/learning_rate": optim_metrics["learning_rate"],
                            "train/grad_norm": avg_grad_norm,
                            "train/surrogate_weight": metrics.get("surrogate_weight", 0.0),
                            
                            # Surrogate/auxiliary token metrics
                            "surrogate/aux_tokens_per_position": running_aux_tokens / (running_steps + 1e-8),
                            "surrogate/positions_with_aux_tokens_pct": running_aux_positions_pct / (running_steps + 1e-8),
                            
                            # Throughput metrics
                            "throughput/tokens_per_second": speed["tokens_per_second_global"],
                            "throughput/tokens_per_second_per_device": speed["tokens_per_second"],
                            "throughput/batches_per_second": speed["batches_per_second"],
                            "throughput/total_tokens_trained": speed["total_tokens_trained"],
                            
                            # FLOPS metrics
                            "flops/total_tflops": total_flops / 1e12,
                            "flops/tflops_per_second": flops_per_second / 1e12,
                            "flops/tflops_per_second_per_device": flops_per_second_per_device / 1e12,
                            
                            # Progress
                            "progress/global_step": self.global_step,
                            "progress/epoch": epoch,
                            "progress/percent_complete": (self.global_step / self.total_training_steps) * 100,
                        }
                        
                        if self.global_rank == 0:
                            progress_bar.set_postfix(
                                loss=f"{avg_loss:.4f}", 
                                ppl=f"{math.exp(min(avg_loss, 20)):.2f}",
                                tok_s=f"{speed['tokens_per_second_global']:.0f}"
                            )
                            
                            if HAS_WANDB and wandb.run is not None:
                                wandb.log(log_metrics, step=self.global_step)
                        
                        # Reset running metrics
                        running_loss = 0.0
                        running_grad_norm = 0.0
                        running_steps = 0
                        running_aux_tokens = 0.0
                        running_aux_positions = 0.0
                        running_aux_positions_pct = 0.0
                        self.speed_metrics.reset()
                    
                    # Evaluation (loss-based)
                    if self.global_step % self.config.training.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        
                        if eval_metrics and self.global_rank == 0:
                            logger.info(f"Step {self.global_step}: {eval_metrics}")
                            
                            if HAS_WANDB and wandb.run is not None:
                                wandb.log(eval_metrics, step=self.global_step)
                            
                            # Save best model
                            if eval_metrics["eval_loss"] < self.best_eval_loss:
                                self.best_eval_loss = eval_metrics["eval_loss"]
                                best_path = os.path.join(self.config.training.output_dir, "best_model")
                                self.save_checkpoint(best_path)
                    
                    # Benchmark evaluation (lm-eval-harness)
                    if (self.config.evaluation.enabled and 
                        self.global_step % self.config.evaluation.eval_interval == 0):
                        benchmark_metrics = self.run_benchmarks()
                        
                        if benchmark_metrics and self.global_rank == 0:
                            logger.info(f"Step {self.global_step} benchmarks: {benchmark_metrics}")
                            
                            if HAS_WANDB and wandb.run is not None:
                                wandb.log(benchmark_metrics, step=self.global_step)
                    
                    # Save checkpoint
                    if self.global_step % self.config.training.save_steps == 0:
                        checkpoint_path = os.path.join(
                            self.config.training.output_dir,
                            f"checkpoint-{self.global_step}",
                        )
                        self.save_checkpoint(checkpoint_path)
                        
                        # Remove old checkpoints
                        self._cleanup_checkpoints()
                    
                    # Check if training is complete
                    if self.global_step >= self.total_training_steps:
                        break
            
            if self.global_step >= self.total_training_steps:
                break
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_metrics = self.evaluate()
        if eval_metrics and self.global_rank == 0:
            logger.info(f"Final evaluation: {eval_metrics}")
            if HAS_WANDB and wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)
        
        # Final benchmarks
        if self.config.evaluation.enabled:
            logger.info("Running final benchmarks...")
            benchmark_metrics = self.run_benchmarks()
            if benchmark_metrics and self.global_rank == 0:
                logger.info(f"Final benchmarks: {benchmark_metrics}")
                if HAS_WANDB and wandb.run is not None:
                    wandb.log(benchmark_metrics, step=self.global_step)
        
        final_path = os.path.join(self.config.training.output_dir, "final_model")
        self.save_checkpoint(final_path)
        
        logger.info("Training complete!")
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints to stay within save_total_limit."""
        if self.global_rank != 0 or self.config.training.save_total_limit <= 0:
            return
        
        checkpoint_dirs = []
        for name in os.listdir(self.config.training.output_dir):
            if name.startswith("checkpoint-"):
                path = os.path.join(self.config.training.output_dir, name)
                if os.path.isdir(path):
                    step = int(name.split("-")[1])
                    checkpoint_dirs.append((step, path))
        
        checkpoint_dirs.sort(key=lambda x: x[0])
        
        while len(checkpoint_dirs) > self.config.training.save_total_limit:
            _, oldest_path = checkpoint_dirs.pop(0)
            logger.info(f"Removing old checkpoint: {oldest_path}")
            import shutil
            shutil.rmtree(oldest_path)
    
    def close(self) -> None:
        """Cleanup resources."""
        # GPU distributed cleanup
        if self.is_distributed and not self.is_tpu:
            dist.destroy_process_group()
        
        # TPU cleanup - rendezvous to ensure all cores finish
        if self.is_tpu and HAS_XLA:
            xm.rendezvous("training_complete")
        
        if HAS_WANDB and wandb.run is not None and self.global_rank == 0:
            wandb.finish()


# =============================================================================
# Main
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Surrogate-Assisted Language Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, help="Base model name or path")
    parser.add_argument("--model_dtype", type=str, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    
    # Surrogate arguments
    parser.add_argument("--surrogate_model", type=str, help="Surrogate model name or path")
    parser.add_argument("--surrogate_prob_threshold", type=float, help="Probability threshold for token selection (e.g., 0.03)")
    parser.add_argument("--surrogate_max_tokens", type=int, help="Maximum tokens to consider per position")
    parser.add_argument("--surrogate_dtype", type=str, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--no_surrogate", action="store_true", help="Disable surrogate model")
    parser.add_argument("--surrogate_loss_weight_initial", type=float, help="Initial surrogate loss weight")
    parser.add_argument("--surrogate_loss_weight_final", type=float, help="Final surrogate loss weight after decay")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, help="Dataset configuration")
    parser.add_argument("--train_file", type=str, help="Path to custom training file")
    parser.add_argument("--eval_file", type=str, help="Path to custom evaluation file")
    parser.add_argument("--text_column", type=str, help="Column name containing text")
    parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "bf16", "no"])
    parser.add_argument("--use_z_loss", action="store_true")
    
    # Logging arguments
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    
    # Checkpoint arguments
    parser.add_argument("--resume_from_checkpoint", type=str)
    
    # Distributed training
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # Device selection
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "tpu", "cpu"],
                        default="auto", help="Device to use for training")
    parser.add_argument("--tpu_cores", type=int, default=1, 
                        help="Number of TPU cores (1 for single, 8 for v3-8)")
    
    return parser.parse_args()


def _train_fn(rank: int, config: Config, base_tokenizer, surrogate_tokenizer, train_dataset, eval_dataset):
    """Training function for TPU multi-core training via xmp.spawn."""
    # Set seed with rank offset for different data ordering
    set_seed(config.training.seed + rank)
    
    # Initialize models (each core loads its own copy)
    logger.info(f"[Rank {rank}] Loading base model: {config.model.name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=config.model.get_torch_dtype(),
        trust_remote_code=config.model.trust_remote_code,
    )
    
    if config.model.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    
    surrogate_model = None
    if config.surrogate.enabled:
        logger.info(f"[Rank {rank}] Loading surrogate model: {config.surrogate.name_or_path}")
        surrogate_model = AutoModelForCausalLM.from_pretrained(
            config.surrogate.name_or_path,
            torch_dtype=config.surrogate.get_torch_dtype(),
            trust_remote_code=config.surrogate.trust_remote_code,
        )
        surrogate_model.eval()
        for param in surrogate_model.parameters():
            param.requires_grad = False
    
    # Initialize trainer
    trainer = SurrogateTrainer(
        config=config,
        model=base_model,
        tokenizer=base_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        surrogate_model=surrogate_model,
        surrogate_tokenizer=surrogate_tokenizer,
        tpu_rank=rank,
    )
    
    try:
        trainer.train()
    finally:
        trainer.close()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
        # Override with command line arguments
        if args.base_model:
            config.model.name_or_path = args.base_model
        if hasattr(args, 'device') and args.device:
            config.training.device = args.device
        if hasattr(args, 'tpu_cores') and args.tpu_cores:
            config.training.tpu_cores = args.tpu_cores
    else:
        config = Config.from_args(args)
    
    # Set random seed
    set_seed(config.training.seed)
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Initialize tokenizers (shared across TPU cores)
    logger.info(f"Loading base tokenizer: {config.model.name_or_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    surrogate_tokenizer = None
    if config.surrogate.enabled:
        logger.info(f"Loading surrogate tokenizer: {config.surrogate.name_or_path}")
        surrogate_tokenizer = AutoTokenizer.from_pretrained(
            config.surrogate.name_or_path,
            padding_side="left",
            trust_remote_code=config.surrogate.trust_remote_code,
        )
        if surrogate_tokenizer.pad_token is None:
            surrogate_tokenizer.pad_token = surrogate_tokenizer.eos_token
    
    # Load datasets (shared across TPU cores)
    logger.info("Loading datasets...")
    train_dataset, eval_dataset = load_training_data(config.data, base_tokenizer)
    
    # Check if TPU multi-core training
    if config.training.device == "tpu" and config.training.tpu_cores > 1:
        if not HAS_XLA:
            raise RuntimeError("PyTorch XLA required for TPU training. Install with: pip install torch-xla")
        
        logger.info(f"Starting TPU multi-core training with {config.training.tpu_cores} cores")
        
        # Use xmp.spawn for multi-core TPU training
        xmp.spawn(
            _train_fn,
            args=(config, base_tokenizer, surrogate_tokenizer, train_dataset, eval_dataset),
            nprocs=config.training.tpu_cores,
            start_method='fork',
        )
    else:
        # Single device training (GPU, single TPU core, MPS, or CPU)
        
        # Determine the dtype for the base model
        # When using fp16 mixed precision on CUDA with GradScaler, model weights must be FP32
        # GradScaler handles the FP16 computation via autocast, but gradients must be FP32
        base_model_dtype = config.model.get_torch_dtype()
        if (config.training.device in ["auto", "cuda"] and 
            torch.cuda.is_available() and 
            config.training.mixed_precision == "fp16"):
            base_model_dtype = torch.float32
            logger.info("Using FP32 for base model weights (fp16 mixed precision via autocast + GradScaler)")
        
        logger.info(f"Loading base model: {config.model.name_or_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            torch_dtype=base_model_dtype,
            trust_remote_code=config.model.trust_remote_code,
        )
        
        if config.model.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
        
        surrogate_model = None
        if config.surrogate.enabled:
            # Surrogate model can stay in FP16 since it's inference-only (no gradients)
            logger.info(f"Loading surrogate model: {config.surrogate.name_or_path}")
            surrogate_model = AutoModelForCausalLM.from_pretrained(
                config.surrogate.name_or_path,
                torch_dtype=config.surrogate.get_torch_dtype(),
                trust_remote_code=config.surrogate.trust_remote_code,
            )
            surrogate_model.eval()
            for param in surrogate_model.parameters():
                param.requires_grad = False
        
        # Initialize trainer
        trainer = SurrogateTrainer(
            config=config,
            model=base_model,
            tokenizer=base_tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            surrogate_model=surrogate_model,
            surrogate_tokenizer=surrogate_tokenizer,
        )
        
        try:
            trainer.train()
        finally:
            trainer.close()


if __name__ == "__main__":
    main()
