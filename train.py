"""
Surrogate-Assisted Language Model Training Script

Trains a language model from scratch with optional surrogate model guidance.
Supports configurable storage directories for models, datasets, and checkpoints.
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_scheduler,
)
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import numpy as np

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from losses import compute_combined_loss


# =============================================================================
# Storage Configuration
# =============================================================================

class StorageManager:
    """Manages storage directories for models, datasets, and outputs."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize storage manager from config.
        
        Args:
            config: Configuration dictionary with 'storage' section
        """
        storage_cfg = config.get('storage', {})
        
        # Base directory (default to ./storage)
        self.base_dir = Path(storage_cfg.get('base_dir', './storage')).resolve()
        
        # Subdirectories
        self.models_dir = self._resolve_path(storage_cfg.get('models_dir', 'models'))
        self.datasets_dir = self._resolve_path(storage_cfg.get('datasets_dir', 'datasets'))
        self.checkpoints_dir = self._resolve_path(storage_cfg.get('checkpoints_dir', 'checkpoints'))
        self.logs_dir = self._resolve_path(storage_cfg.get('logs_dir', 'logs'))
        
        # Options
        self.use_hf_cache = storage_cfg.get('use_hf_cache', False)
        self.set_hf_env_vars = storage_cfg.get('set_hf_env_vars', True)
        
        # Create directories
        self._create_directories()
        
        # Set environment variables if requested
        if self.set_hf_env_vars and not self.use_hf_cache:
            self._set_environment_variables()
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path, making it relative to base_dir if not absolute."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_dir / p
    
    def _create_directories(self):
        """Create all storage directories."""
        for dir_path in [self.base_dir, self.models_dir, self.datasets_dir, 
                         self.checkpoints_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Storage directory: {dir_path}")
    
    def _set_environment_variables(self):
        """Set HuggingFace environment variables for caching."""
        # HF_HOME is the main cache directory (controls all HF caching in v5+)
        os.environ['HF_HOME'] = str(self.models_dir)
        
        # HF_DATASETS_CACHE for datasets library
        os.environ['HF_DATASETS_CACHE'] = str(self.datasets_dir)
        
        # HUGGINGFACE_HUB_CACHE for hub downloads
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(self.models_dir / 'hub')
        
        print(f"Set HF_HOME={os.environ['HF_HOME']}")
        print(f"Set HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")
        print(f"Set HUGGINGFACE_HUB_CACHE={os.environ['HUGGINGFACE_HUB_CACHE']}")
    
    def get_model_cache_dir(self) -> Path:
        """Get the directory for caching models."""
        if self.use_hf_cache:
            return None  # Use default HF cache
        return self.models_dir
    
    def get_dataset_cache_dir(self) -> Path:
        """Get the directory for caching datasets."""
        if self.use_hf_cache:
            return None  # Use default HF cache
        return self.datasets_dir
    
    def get_checkpoint_dir(self, run_name: str = None) -> Path:
        """Get the directory for saving checkpoints."""
        if run_name:
            return self.checkpoints_dir / run_name
        return self.checkpoints_dir
    
    def get_log_dir(self, run_name: str = None) -> Path:
        """Get the directory for saving logs."""
        if run_name:
            return self.logs_dir / run_name
        return self.logs_dir


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    model_name: str,
    storage_manager: StorageManager,
    dtype: str = "bfloat16",
    use_flash_attention: bool = True,
    gradient_checkpointing: bool = True,
    init_from_scratch: bool = False,
    trust_remote_code: bool = False,
) -> AutoModelForCausalLM:
    """
    Load or initialize a causal language model.
    
    Args:
        model_name: HuggingFace model name or path
        storage_manager: StorageManager instance for caching
        dtype: Data type for model weights
        use_flash_attention: Whether to use Flash Attention 2
        gradient_checkpointing: Whether to enable gradient checkpointing
        init_from_scratch: If True, initialize weights randomly
        trust_remote_code: Whether to trust remote code
    
    Returns:
        Loaded or initialized model
    """
    # Resolve dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    # Model kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    
    # Add Flash Attention if available and requested
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Cache directory
    cache_dir = storage_manager.get_model_cache_dir()
    if cache_dir:
        model_kwargs["cache_dir"] = str(cache_dir)
    
    if init_from_scratch:
        # Load config only, then initialize random weights
        print(f"Initializing model from scratch using config from: {model_name}")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        # Apply Flash Attention manually if needed
        if use_flash_attention:
            model.config._attn_implementation = "flash_attention_2"
    else:
        # Load pretrained weights
        print(f"Loading pretrained model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
    
    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} total, {trainable_params:,} trainable")
    
    return model


def load_tokenizer(
    model_name: str,
    storage_manager: StorageManager,
    trust_remote_code: bool = False,
) -> AutoTokenizer:
    """
    Load tokenizer for a model.
    
    Args:
        model_name: HuggingFace model name or path
        storage_manager: StorageManager instance for caching
        trust_remote_code: Whether to trust remote code
    
    Returns:
        Loaded tokenizer
    """
    cache_dir = storage_manager.get_model_cache_dir()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


# =============================================================================
# Data Loading
# =============================================================================

def load_and_prepare_dataset(
    config: Dict[str, Any],
    tokenizer: AutoTokenizer,
    storage_manager: StorageManager,
) -> DatasetDict:
    """
    Load and tokenize dataset.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer for encoding text
        storage_manager: StorageManager instance for caching
    
    Returns:
        Tokenized dataset dict with 'train' and 'validation' splits
    """
    data_cfg = config['data']
    
    dataset_name = data_cfg['dataset_name']
    dataset_config = data_cfg.get('dataset_config')
    text_column = data_cfg.get('text_column', 'text')
    max_seq_length = data_cfg.get('max_seq_length', 2048)
    num_workers = data_cfg.get('preprocessing_num_workers', 4)
    eval_ratio = data_cfg.get('eval_ratio', 0.005)
    
    cache_dir = storage_manager.get_dataset_cache_dir()
    
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    # Handle train/validation split
    if 'validation' not in dataset:
        print(f"Creating validation split ({eval_ratio*100:.1f}% of training data)")
        split_dataset = dataset['train'].train_test_split(
            test_size=eval_ratio,
            seed=config['training'].get('seed', 42),
        )
        dataset = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test'],
        })
    
    # Tokenization function
    def tokenize_function(examples):
        # Tokenize
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=False,
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Group texts for efficient training
    def group_texts(examples):
        # Concatenate all texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        
        # Drop remainder that doesn't fit into a full block
        total_length = (total_length // max_seq_length) * max_seq_length
        
        # Split into chunks
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated.items()
        }
        
        return result
    
    # Apply tokenization
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing",
    )
    
    # Group texts
    print("Grouping texts into chunks...")
    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        desc="Grouping texts",
    )
    
    print(f"Training samples: {len(tokenized_dataset['train']):,}")
    print(f"Validation samples: {len(tokenized_dataset['validation']):,}")
    
    return tokenized_dataset


# =============================================================================
# Training Utilities
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Create cosine schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_surrogate_weight(
    current_step: int,
    total_steps: int,
    initial_weight: float = 1.0,
    final_weight: float = 0.0,
) -> float:
    """Calculate surrogate loss weight using cosine decay."""
    progress = current_step / max(1, total_steps)
    # Cosine decay from initial to final
    weight = final_weight + 0.5 * (initial_weight - final_weight) * (1 + math.cos(math.pi * progress))
    return weight


def collate_fn(batch, pad_token_id: int = 0):
    """Collate function for DataLoader."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(item['input_ids']) for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(item['labels']) for item in batch],
        batch_first=True,
        padding_value=-100,  # Ignore index for loss
    )
    attention_mask = (input_ids != pad_token_id).long()
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: Dict[str, Any]):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
    """
    # Initialize storage manager
    storage_manager = StorageManager(config)
    
    # Extract config sections
    model_cfg = config['model']
    surrogate_cfg = config.get('surrogate', {})
    distillation_cfg = config.get('distillation', {})
    training_cfg = config['training']
    
    # Set seed
    seed = training_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Device setup
    device_str = training_cfg.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Load tokenizer (use student model's tokenizer)
    tokenizer = load_tokenizer(
        model_cfg['name_or_path'],
        storage_manager,
        trust_remote_code=model_cfg.get('trust_remote_code', False),
    )
    
    # Load student model
    print("\n=== Loading Student Model ===")
    student_model = load_model(
        model_cfg['name_or_path'],
        storage_manager,
        dtype=model_cfg.get('dtype', 'bfloat16'),
        use_flash_attention=model_cfg.get('use_flash_attention', True),
        gradient_checkpointing=model_cfg.get('gradient_checkpointing', True),
        init_from_scratch=model_cfg.get('init_from_scratch', False),
        trust_remote_code=model_cfg.get('trust_remote_code', False),
    )
    student_model.to(device)
    
    # Load surrogate model if enabled
    surrogate_model = None
    surrogate_tokenizer = None
    if surrogate_cfg.get('enabled', False) and distillation_cfg.get('mode') != 'none':
        print("\n=== Loading Surrogate Model ===")
        surrogate_model = load_model(
            surrogate_cfg['name_or_path'],
            storage_manager,
            dtype=surrogate_cfg.get('dtype', 'bfloat16'),
            use_flash_attention=False,  # Surrogate doesn't need FA
            gradient_checkpointing=False,  # Surrogate is frozen
            init_from_scratch=False,  # Always use pretrained surrogate
            trust_remote_code=surrogate_cfg.get('trust_remote_code', False),
        )
        surrogate_model.to(device)
        surrogate_model.eval()
        
        # Freeze surrogate
        for param in surrogate_model.parameters():
            param.requires_grad = False
        
        # Load surrogate tokenizer if different from student
        if surrogate_cfg['name_or_path'] != model_cfg['name_or_path']:
            surrogate_tokenizer = load_tokenizer(
                surrogate_cfg['name_or_path'],
                storage_manager,
                trust_remote_code=surrogate_cfg.get('trust_remote_code', False),
            )
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    dataset = load_and_prepare_dataset(config, tokenizer, storage_manager)
    
    # Create data loaders
    train_batch_size = training_cfg.get('per_device_train_batch_size', 4)
    eval_batch_size = training_cfg.get('per_device_eval_batch_size', 4)
    
    from functools import partial
    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    
    train_loader = DataLoader(
        dataset['train'],
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )
    
    eval_loader = DataLoader(
        dataset['validation'],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )
    
    # Calculate training steps
    num_epochs = training_cfg.get('num_epochs', 1)
    gradient_accumulation_steps = training_cfg.get('gradient_accumulation_steps', 1)
    num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_training_steps = num_epochs * num_update_steps_per_epoch
    warmup_ratio = training_cfg.get('warmup_ratio', 0.01)
    num_warmup_steps = int(total_training_steps * warmup_ratio)
    
    print(f"\nTraining for {num_epochs} epochs")
    print(f"Steps per epoch: {num_update_steps_per_epoch:,}")
    print(f"Total training steps: {total_training_steps:,}")
    print(f"Warmup steps: {num_warmup_steps:,}")
    
    # Optimizer
    optimizer = AdamW(
        student_model.parameters(),
        lr=training_cfg.get('learning_rate', 3e-4),
        weight_decay=training_cfg.get('weight_decay', 0.1),
        betas=(0.9, 0.95),
    )
    
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )
    
    # Mixed precision
    mixed_precision = training_cfg.get('mixed_precision', 'bf16')
    if mixed_precision == 'bf16':
        scaler = None  # bf16 doesn't need scaler
        autocast_dtype = torch.bfloat16
    elif mixed_precision == 'fp16':
        scaler = torch.cuda.amp.GradScaler()
        autocast_dtype = torch.float16
    else:
        scaler = None
        autocast_dtype = torch.float32
    
    # Initialize wandb
    if WANDB_AVAILABLE and training_cfg.get('wandb_project'):
        run_name = training_cfg.get('wandb_run_name', 'training-run')
        wandb.init(
            project=training_cfg['wandb_project'],
            entity=training_cfg.get('wandb_entity'),
            name=run_name,
            config=config,
        )
    
    # Checkpoint directory
    run_name = training_cfg.get('wandb_run_name', 'training-run')
    checkpoint_dir = storage_manager.get_checkpoint_dir(run_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n=== Starting Training ===")
    global_step = 0
    best_eval_loss = float('inf')
    
    # Distillation settings
    distillation_mode = distillation_cfg.get('mode', 'none')
    prob_threshold = surrogate_cfg.get('prob_threshold', 0.03)
    max_tokens = surrogate_cfg.get('max_tokens', 100)
    kd_temperature = distillation_cfg.get('kd_temperature', 4.0)
    use_z_loss = training_cfg.get('use_z_loss', False)
    z_loss_weight = training_cfg.get('z_loss_multiplier', 1e-4)
    
    for epoch in range(num_epochs):
        student_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get surrogate logits if needed
            teacher_logits = None
            if surrogate_model is not None and distillation_mode != 'none':
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        # Handle different tokenizers if needed
                        if surrogate_tokenizer is not None:
                            # Re-tokenize for surrogate (simplified - in practice may need more care)
                            surrogate_outputs = surrogate_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )
                        else:
                            surrogate_outputs = surrogate_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )
                        teacher_logits = surrogate_outputs.logits
            
            # Forward pass with mixed precision
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                student_outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                student_logits = student_outputs.logits
                
                # Calculate surrogate weight (cosine decay)
                surrogate_weight = get_surrogate_weight(
                    global_step,
                    total_training_steps,
                    surrogate_cfg.get('loss_weight_initial', 1.0),
                    surrogate_cfg.get('loss_weight_final', 0.0),
                )
                
                # Compute combined loss
                loss_dict = compute_combined_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    distillation_mode=distillation_mode,
                    distillation_weight=surrogate_weight,
                    kd_temperature=kd_temperature,
                    prob_threshold=prob_threshold,
                    max_tokens=max_tokens,
                    use_z_loss=use_z_loss,
                    z_loss_weight=z_loss_weight,
                )
                
                loss = loss_dict['total_loss'] / gradient_accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                max_grad_norm = training_cfg.get('max_grad_norm', 1.0)
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
                
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % training_cfg.get('logging_steps', 50) == 0:
                    lr = scheduler.get_last_lr()[0]
                    log_dict = {
                        'train/loss': loss_dict['total_loss'].item(),
                        'train/ce_loss': loss_dict['ce_loss'].item(),
                        'train/learning_rate': lr,
                        'train/surrogate_weight': surrogate_weight,
                        'train/global_step': global_step,
                    }
                    
                    if 'surrogate_loss' in loss_dict:
                        log_dict['train/surrogate_loss'] = loss_dict['surrogate_loss'].item()
                    if 'kd_loss' in loss_dict:
                        log_dict['train/kd_loss'] = loss_dict['kd_loss'].item()
                    if 'z_loss' in loss_dict:
                        log_dict['train/z_loss'] = loss_dict['z_loss'].item()
                    
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log(log_dict, step=global_step)
                    
                    progress_bar.set_postfix({
                        'loss': f"{loss_dict['total_loss'].item():.4f}",
                        'lr': f"{lr:.2e}",
                    })
                
                # Evaluation
                if global_step % training_cfg.get('eval_steps', 2000) == 0:
                    eval_loss = evaluate(student_model, eval_loader, device, autocast_dtype)
                    print(f"\nStep {global_step}: Eval Loss = {eval_loss:.4f}")
                    
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({'eval/loss': eval_loss}, step=global_step)
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        # Save best model
                        save_checkpoint(
                            student_model, tokenizer, optimizer, scheduler,
                            global_step, checkpoint_dir / 'best',
                        )
                    
                    student_model.train()
                
                # Save checkpoint
                if global_step % training_cfg.get('save_steps', 5000) == 0:
                    save_checkpoint(
                        student_model, tokenizer, optimizer, scheduler,
                        global_step, checkpoint_dir / f'checkpoint-{global_step}',
                    )
            
            epoch_loss += loss_dict['total_loss'].item()
            num_batches += 1
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Final save
    save_checkpoint(
        student_model, tokenizer, optimizer, scheduler,
        global_step, checkpoint_dir / 'final',
    )
    
    print("\n=== Training Complete ===")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


def evaluate(model, eval_loader, device, autocast_dtype):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
            
            num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(model, tokenizer, optimizer, scheduler, step, path):
    """Save training checkpoint."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    
    # Save training state
    torch.save({
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path / 'training_state.pt')
    
    print(f"Checkpoint saved to: {path}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train language model with surrogate guidance")
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--storage-dir', type=str, default=None,
        help='Override base storage directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override output directory for checkpoints'
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override storage directory if provided
    if args.storage_dir:
        if 'storage' not in config:
            config['storage'] = {}
        config['storage']['base_dir'] = args.storage_dir
    
    # Override output directory if provided
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    
    # Run training
    train(config)


if __name__ == '__main__':
    main()
