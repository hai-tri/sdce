#!/usr/bin/env python3
"""
Surrogate-Assisted Language Model Training Script
Supports both continual pretraining (from pretrained weights) and 
pretraining from scratch (random weight initialization).
"""

import os
import math
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    default_data_collator,
    get_scheduler,
)
from datasets import load_dataset, Dataset, DatasetDict

import wandb

from losses import (
    compute_cross_entropy_loss,
    compute_surrogate_distillation_loss,
    compute_kd_loss,
    compute_z_loss,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def get_device(config: OmegaConf) -> torch.device:
    """Determine the device to use for training."""
    device_str = config.training.get("device", "auto")
    
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_str == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            logger.warning("TPU requested but torch_xla not available. Falling back to CPU.")
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.float32)


def load_model_and_tokenizer(config: OmegaConf, device: torch.device):
    """
    Load the base model and tokenizer.
    
    Supports two modes:
    - Continual pretraining: Load pretrained weights (init_from_scratch: false)
    - Pretraining from scratch: Random weight initialization (init_from_scratch: true)
    """
    model_config = config.model
    model_name = model_config.name_or_path
    dtype = get_dtype(model_config.dtype)
    init_from_scratch = model_config.get("init_from_scratch", False)
    
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if init_from_scratch:
        # Pretraining from scratch: load config only, initialize weights randomly
        logger.info(f"Initializing model from scratch with architecture from {model_name}")
        model_architecture_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=model_config.get("trust_remote_code", False),
        )
        
        # Apply any custom config modifications
        if model_config.get("custom_config"):
            for key, value in model_config.custom_config.items():
                if hasattr(model_architecture_config, key):
                    setattr(model_architecture_config, key, value)
                    logger.info(f"Set model config {key} = {value}")
        
        model = AutoModelForCausalLM.from_config(
            model_architecture_config,
            torch_dtype=dtype,
            trust_remote_code=model_config.get("trust_remote_code", False),
        )
        
        # Initialize weights properly
        logger.info("Initializing model weights...")
        model.apply(model._init_weights)
        
        # Log parameter count
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {num_params:,} parameters (random initialization)")
    else:
        # Continual pretraining: load pretrained weights
        logger.info(f"Loading pretrained model from {model_name}")
        
        # Configure Flash Attention if requested
        attn_implementation = None
        if model_config.get("use_flash_attention", False):
            attn_implementation = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=model_config.get("trust_remote_code", False),
            attn_implementation=attn_implementation,
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {num_params:,} parameters (pretrained weights)")
    
    # Enable gradient checkpointing if requested
    if model_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    model = model.to(device)
    return model, tokenizer


def load_surrogate_model(config: OmegaConf, device: torch.device):
    """Load the surrogate model for guidance signals."""
    surrogate_config = config.surrogate
    
    if not surrogate_config.get("enabled", True):
        logger.info("Surrogate model disabled")
        return None, None
    
    model_name = surrogate_config.name_or_path
    dtype = get_dtype(surrogate_config.dtype)
    
    logger.info(f"Loading surrogate model from {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=surrogate_config.get("trust_remote_code", False),
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=surrogate_config.get("trust_remote_code", False),
    )
    
    # Surrogate model is always frozen
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Surrogate model loaded with {num_params:,} parameters (frozen)")
    
    return model, tokenizer


def prepare_dataset(config: OmegaConf, tokenizer) -> DatasetDict:
    """Load and preprocess the dataset."""
    data_config = config.data
    
    # Load dataset
    if data_config.get("train_file"):
        # Custom dataset from files
        data_files = {"train": data_config.train_file}
        if data_config.get("eval_file"):
            data_files["validation"] = data_config.eval_file
        dataset = load_dataset("json", data_files=data_files)
    else:
        # HuggingFace dataset
        dataset_name = data_config.dataset_name
        dataset_config_name = data_config.get("dataset_config")
        
        logger.info(f"Loading dataset: {dataset_name}" + 
                   (f" ({dataset_config_name})" if dataset_config_name else ""))
        
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            trust_remote_code=True,
        )
    
    # Check if we need to create a validation split
    train_split = data_config.get("dataset_split", "train")
    eval_split = data_config.get("eval_split", "validation")
    
    if eval_split not in dataset and "test" not in dataset:
        # Create validation split from training data
        eval_ratio = data_config.get("eval_ratio", 0.01)  # Default 1% for eval
        logger.info(f"Creating validation split with {eval_ratio*100:.1f}% of training data")
        
        train_data = dataset[train_split]
        split_dataset = train_data.train_test_split(
            test_size=eval_ratio,
            seed=config.training.get("seed", 42),
        )
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        })
        eval_split = "validation"
    elif eval_split not in dataset and "test" in dataset:
        eval_split = "test"
    
    # Get text column
    text_column = data_config.get("text_column", "text")
    max_seq_length = data_config.max_seq_length
    
    def tokenize_function(examples):
        """Tokenize and chunk text into fixed-length sequences."""
        # Concatenate all texts
        texts = examples[text_column]
        
        # Filter out empty strings
        texts = [t for t in texts if t and len(t.strip()) > 0]
        
        if not texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        
        # Concatenate all token ids
        concatenated_ids = []
        for ids in tokenized["input_ids"]:
            concatenated_ids.extend(ids)
        
        # Chunk into fixed-length sequences
        total_length = len(concatenated_ids)
        total_length = (total_length // max_seq_length) * max_seq_length
        
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for i in range(0, total_length, max_seq_length):
            chunk = concatenated_ids[i:i + max_seq_length]
            result["input_ids"].append(chunk)
            result["attention_mask"].append([1] * len(chunk))
            result["labels"].append(chunk.copy())
        
        return result
    
    # Process dataset
    num_workers = data_config.get("preprocessing_num_workers", 4)
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset[train_split].column_names,
        desc="Tokenizing",
    )
    
    # Filter out empty examples
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) > 0
    )
    
    logger.info(f"Train examples: {len(tokenized_dataset[train_split])}")
    if eval_split in tokenized_dataset:
        logger.info(f"Eval examples: {len(tokenized_dataset[eval_split])}")
    
    return tokenized_dataset, train_split, eval_split


def get_surrogate_loss_weight(
    step: int,
    total_steps: int,
    initial_weight: float,
    final_weight: float,
) -> float:
    """Calculate surrogate loss weight using cosine decay schedule."""
    if total_steps == 0:
        return initial_weight
    
    progress = min(step / total_steps, 1.0)
    # Cosine decay from initial to final
    weight = final_weight + (initial_weight - final_weight) * 0.5 * (1 + math.cos(math.pi * progress))
    return weight


def get_kd_alpha(
    step: int,
    total_steps: int,
    config: OmegaConf,
) -> float:
    """Get KD alpha value, optionally with scheduling."""
    distill_config = config.distillation
    
    if not distill_config.get("kd_alpha_schedule", False):
        return distill_config.get("kd_alpha", 0.5)
    
    initial_alpha = distill_config.get("kd_alpha_initial", 0.9)
    final_alpha = distill_config.get("kd_alpha_final", 0.1)
    
    if total_steps == 0:
        return initial_alpha
    
    progress = min(step / total_steps, 1.0)
    # Cosine decay from initial to final
    alpha = final_alpha + (initial_alpha - final_alpha) * 0.5 * (1 + math.cos(math.pi * progress))
    return alpha


def evaluate(
    model,
    eval_dataloader,
    device,
    config: OmegaConf,
) -> Dict[str, float]:
    """Run evaluation and return metrics."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            loss = compute_cross_entropy_loss(outputs.logits, labels)
            
            # Count tokens (excluding padding)
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    model.train()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    
    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
    }


def run_benchmarks(
    model,
    tokenizer,
    config: OmegaConf,
    step: int,
) -> Dict[str, float]:
    """Run lm-evaluation-harness benchmarks."""
    eval_config = config.evaluation
    
    if not eval_config.get("enabled", False):
        return {}
    
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.warning("lm-eval not installed. Skipping benchmarks.")
        return {}
    
    logger.info(f"Running benchmarks at step {step}...")
    
    # Wrap model for lm-eval
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=eval_config.get("batch_size", 8),
    )
    
    tasks = eval_config.get("tasks", ["hellaswag"])
    num_fewshot = eval_config.get("num_fewshot", 0)
    limit = eval_config.get("limit")
    
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    
    metrics = {}
    
    if eval_config.get("log_individual_tasks", True):
        for task_name, task_results in results["results"].items():
            # Get the main accuracy metric
            for metric_name, value in task_results.items():
                if "acc" in metric_name.lower() and "stderr" not in metric_name.lower():
                    metrics[f"benchmark/{task_name}_{metric_name}"] = value
    
    if eval_config.get("log_aggregate_score", True):
        # Calculate mean accuracy across tasks
        accuracies = []
        for task_results in results["results"].values():
            for metric_name, value in task_results.items():
                if metric_name == "acc" or metric_name == "acc_norm":
                    accuracies.append(value)
                    break
        
        if accuracies:
            metrics["benchmark/mean_accuracy"] = np.mean(accuracies)
    
    return metrics


def train(config_path: str):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Set seed
    seed = config.training.get("seed", 42)
    set_seed(seed)
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    if config.training.get("wandb_project"):
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.get("wandb_run_name"),
            entity=config.training.get("wandb_entity"),
            config=OmegaConf.to_container(config, resolve=True),
        )
    
    # Load models
    model, tokenizer = load_model_and_tokenizer(config, device)
    surrogate_model, surrogate_tokenizer = load_surrogate_model(config, device)
    
    # Prepare dataset
    dataset, train_split, eval_split = prepare_dataset(config, tokenizer)
    
    # Create dataloaders
    train_batch_size = config.training.per_device_train_batch_size
    eval_batch_size = config.training.get("per_device_eval_batch_size", train_batch_size)
    
    train_dataloader = DataLoader(
        dataset[train_split],
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=2,
        pin_memory=True,
    )
    
    eval_dataloader = None
    if eval_split in dataset:
        eval_dataloader = DataLoader(
            dataset[eval_split],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
            num_workers=2,
            pin_memory=True,
        )
    
    # Calculate training steps
    num_epochs = config.training.num_epochs
    gradient_accumulation_steps = config.training.get("gradient_accumulation_steps", 1)
    num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    total_steps = num_update_steps_per_epoch * num_epochs
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Steps per epoch: {num_update_steps_per_epoch}")
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.get("weight_decay", 0.01),
    )
    
    # Setup scheduler
    warmup_steps = config.training.get("warmup_steps")
    if warmup_steps is None:
        warmup_ratio = config.training.get("warmup_ratio", 0.1)
        warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_scheduler(
        name=config.training.get("lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Mixed precision setup
    mixed_precision = config.training.get("mixed_precision", "no")
    scaler = None
    if mixed_precision in ["fp16", "float16"] and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        autocast_dtype = torch.float16
    elif mixed_precision in ["bf16", "bfloat16"]:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None
    
    # Training settings
    distillation_mode = config.distillation.get("mode", "surrogate")
    use_z_loss = config.training.get("use_z_loss", False)
    z_loss_multiplier = config.training.get("z_loss_multiplier", 1e-4)
    max_grad_norm = config.training.get("max_grad_norm", 1.0)
    logging_steps = config.training.get("logging_steps", 10)
    eval_steps = config.training.get("eval_steps", 500)
    save_steps = config.training.get("save_steps", 1000)
    eval_interval = config.evaluation.get("eval_interval", 1000) if config.get("evaluation") else None
    
    # Surrogate settings
    surrogate_weight_initial = config.surrogate.get("loss_weight_initial", 1.0)
    surrogate_weight_final = config.surrogate.get("loss_weight_final", 0.0)
    prob_threshold = config.surrogate.get("prob_threshold", 0.03)
    max_tokens = config.surrogate.get("max_tokens", 100)
    
    # KD settings
    kd_temperature = config.distillation.get("kd_temperature", 4.0)
    
    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    global_step = 0
    model.train()
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            if autocast_dtype:
                with torch.amp.autocast(device.type, dtype=autocast_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits
                    
                    # Base cross-entropy loss
                    ce_loss = compute_cross_entropy_loss(logits, labels)
                    total_loss = ce_loss
                    
                    # Distillation loss
                    distill_loss = torch.tensor(0.0, device=device)
                    surrogate_weight = 0.0
                    kd_alpha = 0.0
                    
                    if distillation_mode == "surrogate" and surrogate_model is not None:
                        surrogate_weight = get_surrogate_loss_weight(
                            global_step, total_steps,
                            surrogate_weight_initial, surrogate_weight_final,
                        )
                        
                        if surrogate_weight > 0:
                            with torch.no_grad():
                                surrogate_outputs = surrogate_model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                )
                            
                            distill_loss = compute_surrogate_distillation_loss(
                                student_logits=logits,
                                teacher_logits=surrogate_outputs.logits,
                                labels=labels,
                                prob_threshold=prob_threshold,
                                max_tokens=max_tokens,
                            )
                            total_loss = ce_loss + surrogate_weight * distill_loss
                    
                    elif distillation_mode == "kd" and surrogate_model is not None:
                        kd_alpha = get_kd_alpha(global_step, total_steps, config)
                        
                        if kd_alpha > 0:
                            with torch.no_grad():
                                teacher_outputs = surrogate_model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                )
                            
                            distill_loss = compute_kd_loss(
                                student_logits=logits,
                                teacher_logits=teacher_outputs.logits,
                                labels=labels,
                                temperature=kd_temperature,
                            )
                            total_loss = (1 - kd_alpha) * ce_loss + kd_alpha * distill_loss
                    
                    # Z-loss
                    z_loss = torch.tensor(0.0, device=device)
                    if use_z_loss:
                        z_loss = compute_z_loss(logits)
                        total_loss = total_loss + z_loss_multiplier * z_loss
                    
                    # Scale for gradient accumulation
                    total_loss = total_loss / gradient_accumulation_steps
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                
                ce_loss = compute_cross_entropy_loss(logits, labels)
                total_loss = ce_loss
                
                distill_loss = torch.tensor(0.0, device=device)
                surrogate_weight = 0.0
                kd_alpha = 0.0
                
                if distillation_mode == "surrogate" and surrogate_model is not None:
                    surrogate_weight = get_surrogate_loss_weight(
                        global_step, total_steps,
                        surrogate_weight_initial, surrogate_weight_final,
                    )
                    
                    if surrogate_weight > 0:
                        with torch.no_grad():
                            surrogate_outputs = surrogate_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )
                        
                        distill_loss = compute_surrogate_distillation_loss(
                            student_logits=logits,
                            teacher_logits=surrogate_outputs.logits,
                            labels=labels,
                            prob_threshold=prob_threshold,
                            max_tokens=max_tokens,
                        )
                        total_loss = ce_loss + surrogate_weight * distill_loss
                
                elif distillation_mode == "kd" and surrogate_model is not None:
                    kd_alpha = get_kd_alpha(global_step, total_steps, config)
                    
                    if kd_alpha > 0:
                        with torch.no_grad():
                            teacher_outputs = surrogate_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )
                        
                        distill_loss = compute_kd_loss(
                            student_logits=logits,
                            teacher_logits=teacher_outputs.logits,
                            labels=labels,
                            temperature=kd_temperature,
                        )
                        total_loss = (1 - kd_alpha) * ce_loss + kd_alpha * distill_loss
                
                z_loss = torch.tensor(0.0, device=device)
                if use_z_loss:
                    z_loss = compute_z_loss(logits)
                    total_loss = total_loss + z_loss_multiplier * z_loss
                
                total_loss = total_loss / gradient_accumulation_steps
            
            # Backward pass
            if scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            epoch_loss += total_loss.item() * gradient_accumulation_steps
            epoch_steps += 1
            
            # Gradient accumulation step
            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    current_lr = scheduler.get_last_lr()[0]
                    
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/ce_loss": ce_loss.item(),
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch + step / len(train_dataloader),
                        "train/global_step": global_step,
                    }
                    
                    if distillation_mode == "surrogate":
                        log_dict["train/surrogate_loss"] = distill_loss.item()
                        log_dict["train/surrogate_weight"] = surrogate_weight
                    elif distillation_mode == "kd":
                        log_dict["train/kd_loss"] = distill_loss.item()
                        log_dict["train/kd_alpha"] = kd_alpha
                    
                    if use_z_loss:
                        log_dict["train/z_loss"] = z_loss.item()
                    
                    if wandb.run:
                        wandb.log(log_dict, step=global_step)
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                    })
                
                # Evaluation
                if eval_dataloader and global_step % eval_steps == 0:
                    eval_metrics = evaluate(model, eval_dataloader, device, config)
                    logger.info(f"Step {global_step}: {eval_metrics}")
                    
                    if wandb.run:
                        wandb.log(eval_metrics, step=global_step)
                
                # Benchmark evaluation
                if eval_interval and global_step % eval_interval == 0:
                    benchmark_metrics = run_benchmarks(model, tokenizer, config, global_step)
                    if benchmark_metrics:
                        logger.info(f"Benchmarks at step {global_step}: {benchmark_metrics}")
                        if wandb.run:
                            wandb.log(benchmark_metrics, step=global_step)
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Save optimizer and scheduler state
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }, checkpoint_dir / "training_state.pt")
                    
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
                    
                    # Clean up old checkpoints
                    save_total_limit = config.training.get("save_total_limit", 3)
                    checkpoints = sorted(output_dir.glob("checkpoint-*"), 
                                        key=lambda x: int(x.name.split("-")[1]))
                    if len(checkpoints) > save_total_limit:
                        for ckpt in checkpoints[:-save_total_limit]:
                            import shutil
                            shutil.rmtree(ckpt)
                            logger.info(f"Deleted old checkpoint: {ckpt}")
    
    # Final evaluation
    if eval_dataloader:
        final_metrics = evaluate(model, eval_dataloader, device, config)
        logger.info(f"Final evaluation: {final_metrics}")
        if wandb.run:
            wandb.log({"final/" + k: v for k, v in final_metrics.items()}, step=global_step)
    
    # Final benchmark evaluation
    final_benchmarks = run_benchmarks(model, tokenizer, config, global_step)
    if final_benchmarks:
        logger.info(f"Final benchmarks: {final_benchmarks}")
        if wandb.run:
            wandb.log({"final/" + k: v for k, v in final_benchmarks.items()}, step=global_step)
    
    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved final model to {final_dir}")
    
    if wandb.run:
        wandb.finish()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a language model with surrogate guidance")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    
    train(args.config)
