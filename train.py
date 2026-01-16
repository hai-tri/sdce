#!/usr/bin/env python3
"""
Surrogate-Assisted Language Model Training Script

This script implements a training approach where a surrogate model guides the 
primary model's learning by providing token-level perplexity signals.

Usage:
    python train.py --config config.yaml
    python train.py --base_model gpt2 --surrogate_model Qwen/Qwen3-0.6B --dataset wikitext --dataset_config wikitext-2-raw-v1
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
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

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
    k: int = 6  # Number of top-k tokens to consider from surrogate
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
    
    # Distributed
    local_rank: int = -1
    
    # Misc
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    # W&B
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
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
        if args.surrogate_k:
            config.surrogate.k = args.surrogate_k
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
            
        return config
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            "model": vars(self.model),
            "surrogate": vars(self.surrogate),
            "data": vars(self.data),
            "training": vars(self.training),
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
        perp_values: Perplexity values from surrogate model (batch_size, seq_len, k)
        perp_indices: Token indices from surrogate model (batch_size, seq_len, k)
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
        k = perp_indices.shape[-1]
        
        # Translate surrogate indices to base model indices
        translated_perp_indices = lookup_surrogate_to_self[perp_indices]
        
        # Handle invalid translations (-100 or out of bounds)
        valid_translation_mask = (translated_perp_indices >= 0) & (translated_perp_indices < vocab_size)
        translated_perp_indices = translated_perp_indices.clamp(0, vocab_size - 1)
        
        # Gather logits for the top-k tokens
        gathered_logits = torch.gather(logits, dim=2, index=translated_perp_indices)
        gathered_logit_probs = F.softmax(gathered_logits, dim=-1)
        gathered_nll = -torch.log(gathered_logit_probs + 1e-10)
        
        # Mask out invalid entries (where perp_values is inf or translation failed)
        isinf_mask = torch.isinf(perp_values).all(dim=-1)  # (batch, seq)
        valid_row_mask = (~isinf_mask).unsqueeze(-1) & valid_translation_mask  # (batch, seq, k)
        
        # Count valid entries for normalization
        num_valid_surrogate_entries = valid_row_mask.sum().item()
        
        # Compute softmax weights from perplexity (lower perp = higher weight)
        # Mask invalid entries before softmax
        masked_perp_values = perp_values.clone()
        masked_perp_values[~valid_row_mask] = float('inf')
        softmax_weights = F.softmax(-masked_perp_values, dim=-1)
        softmax_weights = softmax_weights * valid_row_mask.float()
        
        # Weighted surrogate loss (scaled by surrogate_weight)
        weighted_nll = gathered_nll * softmax_weights
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
        
        # Base -> Surrogate
        self.lookup_base_to_surrogate = torch.full(
            (base_vocab_size,), fill_value=-100, dtype=torch.long, device=self.device
        )
        
        # Surrogate -> Base
        self.lookup_surrogate_to_base = torch.full(
            (surrogate_vocab_size,), fill_value=-100, dtype=torch.long, device=self.device
        )
        
        # Populate lookup tables
        for token in intersection:
            base_id = base_vocab[token]
            surrogate_id = surrogate_vocab[token]
            self.lookup_base_to_surrogate[base_id] = surrogate_id
            self.lookup_surrogate_to_base[surrogate_id] = base_id
        
        # Store permitted tokens
        self.base_permitted_ids = torch.tensor(
            [base_vocab[t] for t in intersection], device=self.device
        )
        self.surrogate_permitted_ids = torch.tensor(
            [surrogate_vocab[t] for t in intersection], device=self.device
        )
        
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
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.surrogate_model = surrogate_model
        self.surrogate_tokenizer = surrogate_tokenizer
        
        # Setup device
        self.device = self._setup_device()
        
        # Setup distributed training
        self.is_distributed = config.training.local_rank != -1
        self.world_size = 1
        self.global_rank = 0
        self.local_rank = 0
        
        if self.is_distributed:
            self._setup_distributed()
        
        # Move models to device
        self.model = self.model.to(self.device)
        if self.surrogate_model is not None:
            self.surrogate_model = self.surrogate_model.to(self.device)
            self.surrogate_model.eval()
        
        # Setup vocabulary alignment
        self.vocab_aligner = None
        if self.surrogate_model is not None and self.surrogate_tokenizer is not None:
            self.vocab_aligner = VocabularyAligner(
                self.tokenizer,
                self.surrogate_tokenizer,
                self.device,
            )
        
        # Setup data loaders
        self._setup_dataloaders()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup mixed precision
        self._setup_mixed_precision()
        
        # Distributed model wrapper
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        
        # Setup logging
        self._setup_logging()
        
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.cuda.is_available():
            if self.config.training.local_rank != -1:
                return torch.device("cuda", self.config.training.local_rank)
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _setup_distributed(self) -> None:
        """Setup distributed training."""
        dist.init_process_group(backend="nccl")
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.local_rank = self.config.training.local_rank
        torch.cuda.set_device(self.local_rank)
        logger.info(f"Distributed training: rank {self.global_rank}/{self.world_size}")
    
    def _setup_dataloaders(self) -> None:
        """Setup training and evaluation data loaders."""
        train_sampler = None
        if self.is_distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.per_device_train_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=lambda b: collate_fn(b, pad_token_id, self.config.data.max_seq_length),
            num_workers=self.config.data.preprocessing_num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        self.eval_loader = None
        if self.eval_dataset is not None:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.training.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_fn(b, pad_token_id, self.config.data.max_seq_length),
                num_workers=self.config.data.preprocessing_num_workers,
                pin_memory=True,
            )
    
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
        
        if self.config.training.mixed_precision == "fp16":
            self.scaler = GradScaler()
            self.autocast_dtype = torch.float16
        elif self.config.training.mixed_precision == "bf16":
            self.autocast_dtype = torch.bfloat16
    
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
        Compute top-k perplexity tokens from surrogate model.
        
        Returns:
            Tuple of (perp_values, perp_indices) or (None, None) if surrogate is disabled
        """
        if self.surrogate_model is None or not self.config.surrogate.enabled:
            return None, None
        
        k = self.config.surrogate.k
        
        with torch.no_grad():
            # Decode batch to strings using base tokenizer
            input_ids = batch["input_ids"]
            batch_strings = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True)
            
            # Re-encode with surrogate tokenizer
            surrogate_encodings = self.surrogate_tokenizer(
                batch_strings,
                padding=True,
                truncation=True,
                max_length=self.config.data.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            
            # Get surrogate model logits
            surrogate_outputs = self.surrogate_model(**surrogate_encodings)
            surrogate_logits = surrogate_outputs.logits[..., :-1, :].contiguous()
            
            batch_size, surr_seq_len, surr_vocab_size = surrogate_logits.shape
            
            # Compute perplexity (inverse probability)
            surrogate_probs = F.softmax(surrogate_logits, dim=-1)
            surrogate_perp = torch.reciprocal(surrogate_probs + 1e-8)
            
            # Align sequence lengths
            base_seq_len = labels.shape[1]
            min_seq_len = min(base_seq_len, surr_seq_len)
            surrogate_perp = surrogate_perp[:, :min_seq_len, :]
            
            # Mask out tokens not in vocabulary intersection
            vocab_mask = ~torch.isin(
                torch.arange(surr_vocab_size, device=self.device),
                self.vocab_aligner.surrogate_permitted_ids,
            )
            surrogate_perp[:, :, vocab_mask] = float('inf')
            
            # Mask positions corresponding to labels=-100
            aligned_labels = labels[:, :min_seq_len]
            invalid_positions = (aligned_labels == -100).unsqueeze(-1)
            surrogate_perp[invalid_positions.expand_as(surrogate_perp)] = float('inf')
            
            # Get top-k lowest perplexity tokens (exclude the actual label)
            translated_labels = self.vocab_aligner.translate_base_to_surrogate(aligned_labels)
            valid_mask = translated_labels != -100
            
            # Set perplexity of actual labels to inf so they're not selected
            batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, min_seq_len)
            seq_indices = torch.arange(min_seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            
            safe_translated_labels = translated_labels.clone()
            safe_translated_labels[~valid_mask] = 0
            surrogate_perp[batch_indices, seq_indices, safe_translated_labels] = float('inf')
            
            # Top-k selection
            topk_result = torch.topk(surrogate_perp, k=k, largest=False, sorted=True, dim=-1)
            perp_values = topk_result.values  # (batch, seq, k)
            perp_indices = topk_result.indices  # (batch, seq, k)
            
            # Pad to match base model sequence length
            if min_seq_len < base_seq_len:
                pad_len = base_seq_len - min_seq_len
                perp_values = F.pad(perp_values, (0, 0, 0, pad_len), value=float('inf'))
                perp_indices = F.pad(perp_indices, (0, 0, 0, pad_len), value=0)
            
        return perp_values, perp_indices
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Execute a single training step."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get labels
        labels = self.get_labels(batch)
        
        # Compute surrogate guidance
        perp_values, perp_indices = self.compute_surrogate_guidance(batch, labels)
        
        # Forward pass
        with autocast(device_type=self.device.type, dtype=self.autocast_dtype):
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
        
        return metrics
    
    def optimizer_step(self) -> Dict[str, float]:
        """Execute optimizer step with gradient clipping."""
        metrics = {}
        
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
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
        
        for batch in tqdm(self.eval_loader, desc="Evaluating", disable=self.global_rank != 0):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = self.get_labels(batch)
            
            with autocast(device_type=self.device.type, dtype=self.autocast_dtype):
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
        
        # Aggregate across ranks
        if self.is_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_tokens_tensor = torch.tensor(total_tokens, device=self.device)
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
            logger.info(f"  Surrogate k: {self.config.surrogate.k}")
        
        # Resume from checkpoint if specified
        if self.config.training.resume_from_checkpoint:
            self.load_checkpoint(self.config.training.resume_from_checkpoint)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Training metrics
        running_loss = 0.0
        running_steps = 0
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config.training.num_epochs):
            self.epoch = epoch
            
            if self.is_distributed:
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
                
                # Optimizer step (with gradient accumulation)
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    optim_metrics = self.optimizer_step()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.training.logging_steps == 0:
                        avg_loss = running_loss / running_steps
                        elapsed = time.time() - start_time
                        tokens_per_sec = (
                            running_steps
                            * self.config.training.per_device_train_batch_size
                            * self.config.data.max_seq_length
                            * self.world_size
                        ) / elapsed
                        
                        log_metrics = {
                            "train/loss": avg_loss,
                            "train/perplexity": math.exp(min(avg_loss, 20)),
                            "train/learning_rate": optim_metrics["learning_rate"],
                            "train/grad_norm": optim_metrics["grad_norm"],
                            "train/tokens_per_second": tokens_per_sec,
                            "train/global_step": self.global_step,
                        }
                        
                        if self.global_rank == 0:
                            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", ppl=f"{math.exp(min(avg_loss, 20)):.2f}")
                            
                            if HAS_WANDB and wandb.run is not None:
                                wandb.log(log_metrics, step=self.global_step)
                        
                        running_loss = 0.0
                        running_steps = 0
                        start_time = time.time()
                    
                    # Evaluation
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
        
        # Final evaluation and save
        eval_metrics = self.evaluate()
        if eval_metrics and self.global_rank == 0:
            logger.info(f"Final evaluation: {eval_metrics}")
        
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
        if self.is_distributed:
            dist.destroy_process_group()
        
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
    parser.add_argument("--surrogate_k", type=int, help="Number of top-k tokens from surrogate")
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
        # Override with command line arguments
        if args.base_model:
            config.model.name_or_path = args.base_model
        # ... (add other overrides as needed)
    else:
        config = Config.from_args(args)
    
    # Set random seed
    set_seed(config.training.seed)
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Initialize tokenizers
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
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, eval_dataset = load_training_data(config.data, base_tokenizer)
    
    # Initialize models
    logger.info(f"Loading base model: {config.model.name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=config.model.get_torch_dtype(),
        trust_remote_code=config.model.trust_remote_code,
    )
    
    if config.model.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    
    surrogate_model = None
    if config.surrogate.enabled:
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
