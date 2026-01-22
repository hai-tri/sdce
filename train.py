"""
Surrogate-Assisted Language Model Training Script

Trains a language model from scratch with optional surrogate model guidance.
Supports configurable storage directories for models, datasets, and checkpoints.

Device Support:
  - CUDA (single GPU and multi-GPU via torchrun/DDP)
  - MPS (Apple Silicon)
  - TPU (via torch_xla)
  - CPU

Distributed Training:
  - torchrun / torch.distributed for multi-GPU DDP
  - torch_xla for TPU multi-core

Usage:
  # Single GPU / CPU / MPS
  python train.py --config config.yaml

  # Multi-GPU with torchrun (recommended)
  torchrun --nproc_per_node=4 train.py --config config.yaml

  # Multi-GPU with torch.distributed.launch (legacy)
  python -m torch.distributed.launch --nproc_per_node=4 train.py --config config.yaml

  # TPU
  USE_TPU=1 python train.py --config config.yaml --tpu-cores 8
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
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

# TPU support via torch_xla
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.utils.utils as xu
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

from losses import compute_combined_loss


# =============================================================================
# Distributed Training Utilities
# =============================================================================

class DistributedManager:
    """
    Manages distributed training state and utilities.
    Supports torchrun, torch.distributed.launch, and single-process training.
    """
    
    def __init__(self):
        """Initialize distributed manager, detecting if we're in a distributed context."""
        self._initialized = False
        self._backend = None
        
        # Check if launched with torchrun or torch.distributed.launch
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        
        # Determine if we should initialize distributed
        self.is_distributed = self.local_rank != -1
        
        if self.is_distributed:
            self._init_distributed()
    
    def _init_distributed(self):
        """Initialize the distributed process group."""
        if self._initialized:
            return
        
        # Determine backend
        if torch.cuda.is_available():
            self._backend = "nccl"
            # Set device for this process
            torch.cuda.set_device(self.local_rank)
        else:
            self._backend = "gloo"
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self._backend,
                init_method="env://",
            )
        
        self._initialized = True
        
        if self.is_main_process:
            print(f"Distributed training initialized:")
            print(f"  Backend: {self._backend}")
            print(f"  World size: {self.world_size}")
            print(f"  Rank: {self.rank}")
            print(f"  Local rank: {self.local_rank}")
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
    
    @property
    def device(self) -> torch.device:
        """Get the device for this process in distributed mode."""
        if self.is_distributed and torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return None  # Let DeviceManager handle non-distributed case
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce a tensor across all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> list:
        """Gather tensors from all processes."""
        if not self.is_distributed or not dist.is_initialized():
            return [tensor]
        
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return gathered
    
    def cleanup(self):
        """Clean up distributed process group."""
        if self._initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
    
    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """
        Wrap model with DDP if in distributed mode.
        
        Args:
            model: The model to wrap
            device: The device the model is on
            
        Returns:
            Original model or DDP-wrapped model
        """
        if not self.is_distributed:
            return model
        
        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,  # Set to True if you have unused params
        )
        
        if self.is_main_process:
            print(f"Model wrapped with DistributedDataParallel")
        
        return model
    
    def get_sampler(self, dataset, shuffle: bool = True):
        """
        Get appropriate sampler for distributed training.
        
        Args:
            dataset: The dataset
            shuffle: Whether to shuffle
            
        Returns:
            DistributedSampler or None
        """
        if not self.is_distributed:
            return None
        
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )
    
    def set_epoch(self, sampler, epoch: int):
        """Set the epoch for the distributed sampler (for proper shuffling)."""
        if sampler is not None and isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)


# Global distributed manager instance
_distributed_manager: Optional[DistributedManager] = None


def get_distributed_manager() -> DistributedManager:
    """Get or create the global distributed manager."""
    global _distributed_manager
    if _distributed_manager is None:
        _distributed_manager = DistributedManager()
    return _distributed_manager


def is_main_process() -> bool:
    """Check if this is the main process."""
    return get_distributed_manager().is_main_process


@contextmanager
def main_process_first():
    """Context manager to run code on main process first, then others."""
    dist_manager = get_distributed_manager()
    
    if not dist_manager.is_distributed:
        yield
        return
    
    if dist_manager.is_main_process:
        yield
    
    dist_manager.barrier()
    
    if not dist_manager.is_main_process:
        yield
    
    dist_manager.barrier()


def print_rank0(*args, **kwargs):
    """Print only on the main process."""
    if is_main_process():
        print(*args, **kwargs)


# =============================================================================
# Device Management
# =============================================================================

class DeviceManager:
    """
    Manages device selection and provides device-aware utilities.
    Supports CUDA (single and multi-GPU), MPS, TPU, and CPU.
    """
    
    DEVICE_CUDA = "cuda"
    DEVICE_MPS = "mps"
    DEVICE_TPU = "tpu"
    DEVICE_CPU = "cpu"
    
    def __init__(self, device_config: str = "auto", distributed_manager: Optional[DistributedManager] = None):
        """
        Initialize device manager.
        
        Args:
            device_config: Device specification - "auto", "cuda", "mps", "tpu", "cpu",
                          or specific device like "cuda:0"
            distributed_manager: Optional DistributedManager for multi-GPU training
        """
        self.device_config = device_config
        self.dist_manager = distributed_manager or get_distributed_manager()
        self.device = self._resolve_device(device_config)
        self.device_type = self._get_device_type()
        self.is_tpu = self.device_type == self.DEVICE_TPU
        self.is_cuda = self.device_type == self.DEVICE_CUDA
        self.is_mps = self.device_type == self.DEVICE_MPS
        self.is_cpu = self.device_type == self.DEVICE_CPU
        
    def _resolve_device(self, device_config: str) -> torch.device:
        """Resolve device string to torch.device."""
        # If in distributed mode, use the device assigned by distributed manager
        if self.dist_manager.is_distributed and self.dist_manager.device is not None:
            return self.dist_manager.device
        
        if device_config == "auto":
            return self._auto_detect_device()
        elif device_config == "tpu":
            if not TPU_AVAILABLE:
                raise RuntimeError("TPU requested but torch_xla is not installed")
            return xm.xla_device()
        else:
            return torch.device(device_config)
    
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect the best available device."""
        # Check for TPU first (if explicitly enabled via environment)
        if TPU_AVAILABLE and os.environ.get("USE_TPU", "").lower() in ("1", "true"):
            print_rank0("TPU detected and enabled via USE_TPU environment variable")
            return xm.xla_device()
        
        # Check for CUDA
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Additional check for MPS functionality
            try:
                # Test if MPS actually works
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                return torch.device("mps")
            except Exception:
                pass
        
        # Fallback to CPU
        return torch.device("cpu")
    
    def _get_device_type(self) -> str:
        """Get the device type string for autocast and other operations."""
        if self.is_tpu:
            return self.DEVICE_TPU
        device_str = str(self.device)
        if "cuda" in device_str:
            return self.DEVICE_CUDA
        elif "mps" in device_str:
            return self.DEVICE_MPS
        else:
            return self.DEVICE_CPU
    
    def get_autocast_context(self, dtype: torch.dtype, enabled: bool = True):
        """
        Get the appropriate autocast context for the device.
        
        Args:
            dtype: The dtype for autocast (e.g., torch.bfloat16, torch.float16)
            enabled: Whether autocast should be enabled
            
        Returns:
            Context manager for autocast
        """
        if not enabled or self.is_cpu:
            # CPU autocast has limited support; often better to disable
            return torch.autocast(device_type="cpu", enabled=False)
        
        if self.is_tpu:
            # TPU uses bfloat16 natively, autocast not needed in the same way
            # Return a no-op context
            return torch.autocast(device_type="cpu", enabled=False)
        
        if self.is_mps:
            # MPS supports float16 autocast
            # Note: bfloat16 support on MPS is limited
            if dtype == torch.bfloat16:
                # MPS doesn't fully support bfloat16, fall back to float16
                dtype = torch.float16
            return torch.autocast(device_type="mps", dtype=dtype, enabled=enabled)
        
        # CUDA - full support
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)
    
    def get_grad_scaler(self, mixed_precision: str):
        """
        Get gradient scaler if applicable for the device and precision.
        
        Args:
            mixed_precision: "fp16", "bf16", or "fp32"
            
        Returns:
            GradScaler instance or None
        """
        # GradScaler is only needed for fp16 on CUDA
        if mixed_precision == "fp16" and self.is_cuda:
            return torch.cuda.amp.GradScaler()
        
        # MPS has its own scaler in newer PyTorch versions
        if mixed_precision == "fp16" and self.is_mps:
            try:
                # PyTorch 2.0+ has GradScaler that works with MPS
                return torch.amp.GradScaler("mps")
            except (TypeError, AttributeError):
                # Older PyTorch - no scaler for MPS
                return None
        
        # TPU and CPU don't use GradScaler
        return None
    
    def get_autocast_dtype(self, mixed_precision: str) -> torch.dtype:
        """
        Get the appropriate dtype for the mixed precision setting.
        
        Args:
            mixed_precision: "fp16", "bf16", or "fp32"
            
        Returns:
            torch.dtype
        """
        if mixed_precision == "bf16":
            if self.is_mps:
                # MPS has limited bfloat16 support, use float16
                print_rank0("Warning: MPS has limited bfloat16 support, using float16 instead")
                return torch.float16
            return torch.bfloat16
        elif mixed_precision == "fp16":
            return torch.float16
        else:
            return torch.float32
    
    def supports_flash_attention(self) -> bool:
        """Check if the device supports Flash Attention."""
        if not self.is_cuda:
            return False
        
        # Flash Attention requires Ampere (SM 8.0) or newer
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            return capability[0] >= 8
        return False
    
    def synchronize(self):
        """Synchronize the device (wait for all operations to complete)."""
        if self.is_cuda:
            torch.cuda.synchronize()
        elif self.is_tpu:
            xm.mark_step()
        elif self.is_mps:
            torch.mps.synchronize()
        # CPU doesn't need synchronization
    
    def get_world_size(self) -> int:
        """Get the number of devices (for distributed training)."""
        # First check distributed manager
        if self.dist_manager.is_distributed:
            return self.dist_manager.world_size
        
        if self.is_tpu:
            return xm.xrt_world_size()
        elif self.is_cuda:
            return max(torch.cuda.device_count(), 1)
        else:
            return 1
    
    def optimizer_step(self, optimizer, scaler=None):
        """
        Perform optimizer step with device-specific handling.
        
        Args:
            optimizer: The optimizer
            scaler: Optional GradScaler for mixed precision
        """
        if self.is_tpu:
            # TPU requires xm.optimizer_step
            xm.optimizer_step(optimizer)
        elif scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    
    def __str__(self) -> str:
        return f"DeviceManager(device={self.device}, type={self.device_type}, distributed={self.dist_manager.is_distributed})"


# =============================================================================
# Training Metrics
# =============================================================================

class TrainingMetrics:
    """Tracks training throughput metrics: tokens/sec, FLOPs, grad_norm, etc."""
    
    def __init__(
        self,
        num_params: int,
        world_size: int = 1,
    ):
        """
        Initialize metrics tracker.
        
        Args:
            num_params: Number of model parameters
            world_size: Number of devices (for distributed training)
        """
        self.num_params = num_params
        self.world_size = world_size
        
        # Running totals
        self.total_tokens = 0
        self.total_flops = 0
        
        # Per-step tracking
        self.step_start_time = None
        self.step_tokens = 0
        
        # Smoothed metrics (exponential moving average)
        self.ema_tokens_per_sec = None
        self.ema_alpha = 0.1  # Smoothing factor
    
    def start_step(self):
        """Call at the beginning of each optimizer step (before accumulation)."""
        self.step_start_time = time.perf_counter()
        self.step_tokens = 0
    
    def add_tokens(self, num_tokens: int):
        """Add tokens processed in this micro-batch."""
        self.step_tokens += num_tokens
    
    def end_step(self) -> dict:
        """
        Call at the end of each optimizer step (after optimizer.step).
        
        Returns:
            Dictionary with throughput metrics
        """
        if self.step_start_time is None:
            return {}
        
        elapsed = time.perf_counter() - self.step_start_time
        
        # Update totals (multiply by world_size for total across all devices)
        self.total_tokens += self.step_tokens * self.world_size
        
        # Estimate FLOPs for this step
        # Standard approximation: 6 * num_params * num_tokens
        # (2 for forward, 4 for backward with activations)
        step_flops = 6 * self.num_params * self.step_tokens * self.world_size
        self.total_flops += step_flops
        
        # Compute rates (total across all devices)
        tokens_per_sec = (self.step_tokens * self.world_size) / elapsed if elapsed > 0 else 0
        tokens_per_sec_per_device = self.step_tokens / elapsed if elapsed > 0 else 0
        flops_per_sec = step_flops / elapsed if elapsed > 0 else 0
        flops_per_sec_per_device = flops_per_sec / self.world_size
        
        # Update EMA for smoothed tokens/sec
        if self.ema_tokens_per_sec is None:
            self.ema_tokens_per_sec = tokens_per_sec
        else:
            self.ema_tokens_per_sec = (
                self.ema_alpha * tokens_per_sec + 
                (1 - self.ema_alpha) * self.ema_tokens_per_sec
            )
        
        return {
            'tokens_per_sec': tokens_per_sec,
            'tokens_per_sec_per_device': tokens_per_sec_per_device,
            'tokens_per_sec_smoothed': self.ema_tokens_per_sec,
            'total_tokens': self.total_tokens,
            'total_flops': self.total_flops,
            'flops_per_sec': flops_per_sec,
            'flops_per_sec_per_device': flops_per_sec_per_device,
            'tflops_per_sec': flops_per_sec / 1e12,
            'tflops_per_sec_per_device': flops_per_sec_per_device / 1e12,
            'step_time_sec': elapsed,
        }
    
    def get_total_metrics(self) -> dict:
        """Get cumulative metrics."""
        return {
            'total_tokens': self.total_tokens,
            'total_flops': self.total_flops,
        }


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
        
        # Only create directories on main process
        if is_main_process():
            self._create_directories()
        
        # Barrier to ensure directories exist before other processes continue
        get_distributed_manager().barrier()
        
        # Set environment variables if requested (all processes)
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
        # HF_HOME is the main cache directory
        os.environ['HF_HOME'] = str(self.models_dir)
        
        # These are more specific cache locations
        os.environ['HF_DATASETS_CACHE'] = str(self.datasets_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(self.models_dir / 'transformers')
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(self.models_dir / 'hub')
        
        if is_main_process():
            print(f"Set HF_HOME={os.environ['HF_HOME']}")
            print(f"Set HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")
            print(f"Set TRANSFORMERS_CACHE={os.environ['TRANSFORMERS_CACHE']}")
    
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
    device_manager: DeviceManager,
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
        device_manager: DeviceManager instance for device-aware settings
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
    
    # Adjust dtype for device compatibility
    if device_manager.is_mps and torch_dtype == torch.bfloat16:
        print_rank0("Warning: MPS has limited bfloat16 support, using float16 for model weights")
        torch_dtype = torch.float16
    
    # Model kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    
    # Add Flash Attention if available and requested
    # Only enable for CUDA with appropriate GPU capability
    if use_flash_attention and device_manager.supports_flash_attention():
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print_rank0("Flash Attention 2 enabled")
    elif use_flash_attention:
        print_rank0(f"Flash Attention requested but not available on {device_manager.device_type}")
    
    # Cache directory
    cache_dir = storage_manager.get_model_cache_dir()
    if cache_dir:
        model_kwargs["cache_dir"] = str(cache_dir)
    
    # Load model on main process first to handle downloads
    with main_process_first():
        if init_from_scratch:
            # Load config only, then initialize random weights
            print_rank0(f"Initializing model from scratch using config from: {model_name}")
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
            # Apply Flash Attention manually if needed and supported
            if use_flash_attention and device_manager.supports_flash_attention():
                model.config._attn_implementation = "flash_attention_2"
        else:
            # Load pretrained weights
            print_rank0(f"Loading pretrained model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
    
    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print_rank0("Gradient checkpointing enabled")
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"Model parameters: {num_params:,} total, {trainable_params:,} trainable")
    
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
    
    # Load on main process first
    with main_process_first():
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
    
    print_rank0(f"Loading dataset: {dataset_name}")
    
    # Load dataset on main process first
    with main_process_first():
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        
        # Handle train/validation split
        if 'validation' not in dataset:
            print_rank0(f"Creating validation split ({eval_ratio*100:.1f}% of training data)")
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
    
    # Apply tokenization (can be done in parallel by all processes)
    print_rank0("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing",
    )
    
    # Group texts
    print_rank0("Grouping texts into chunks...")
    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        desc="Grouping texts",
    )
    
    print_rank0(f"Training samples: {len(tokenized_dataset['train']):,}")
    print_rank0(f"Validation samples: {len(tokenized_dataset['validation']):,}")
    
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


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn,
    device_manager: DeviceManager,
    dist_manager: DistributedManager,
    num_workers: int = 4,
):
    """
    Create a DataLoader with device-specific and distributed settings.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size per device
        shuffle: Whether to shuffle
        collate_fn: Collate function
        device_manager: DeviceManager instance
        dist_manager: DistributedManager instance
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (DataLoader, sampler) - sampler may be None
    """
    # pin_memory is only beneficial for CUDA
    pin_memory = device_manager.is_cuda
    
    # For TPU, num_workers should typically be 0 or handled differently
    if device_manager.is_tpu:
        num_workers = 0
    
    # Get distributed sampler if needed
    sampler = dist_manager.get_sampler(dataset, shuffle=shuffle)
    
    # If using distributed sampler, don't shuffle in DataLoader
    loader_shuffle = shuffle if sampler is None else False
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=loader_shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for consistency in distributed
    )
    
    # Wrap with ParallelLoader for TPU
    if device_manager.is_tpu:
        loader = pl.ParallelLoader(loader, [device_manager.device]).per_device_loader(device_manager.device)
        sampler = None  # TPU handles its own parallelism
    
    return loader, sampler


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: Dict[str, Any]):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
    """
    # Get distributed manager
    dist_manager = get_distributed_manager()
    
    # Initialize storage manager
    storage_manager = StorageManager(config)
    
    # Extract config sections
    model_cfg = config['model']
    surrogate_cfg = config.get('surrogate', {})
    sdce_cfg = config.get('sdce', {})
    training_cfg = config['training']
    
    # Set seed (adjust for distributed to ensure different data per process)
    seed = training_cfg.get('seed', 42)
    torch.manual_seed(seed + dist_manager.rank)
    np.random.seed(seed + dist_manager.rank)
    
    # Initialize device manager
    device_str = training_cfg.get('device', 'auto')
    device_manager = DeviceManager(device_str, dist_manager)
    device = device_manager.device
    
    print_rank0(f"\n=== Device Configuration ===")
    print_rank0(f"Device: {device}")
    print_rank0(f"Device type: {device_manager.device_type}")
    print_rank0(f"Distributed: {dist_manager.is_distributed}")
    if dist_manager.is_distributed:
        print_rank0(f"World size: {dist_manager.world_size}")
    print_rank0(f"Flash Attention support: {device_manager.supports_flash_attention()}")
    
    # Load tokenizer (use student model's tokenizer)
    tokenizer = load_tokenizer(
        model_cfg['name_or_path'],
        storage_manager,
        trust_remote_code=model_cfg.get('trust_remote_code', False),
    )
    
    # Load student model
    print_rank0("\n=== Loading Student Model ===")
    student_model = load_model(
        model_cfg['name_or_path'],
        storage_manager,
        device_manager,
        dtype=model_cfg.get('dtype', 'bfloat16'),
        use_flash_attention=model_cfg.get('use_flash_attention', True),
        gradient_checkpointing=model_cfg.get('gradient_checkpointing', True),
        init_from_scratch=model_cfg.get('init_from_scratch', False),
        trust_remote_code=model_cfg.get('trust_remote_code', False),
    )
    student_model.to(device)
    
    # Wrap with DDP if distributed
    student_model = dist_manager.wrap_model(student_model, device)
    
    # Load surrogate model if SDCE mode requires it
    surrogate_model = None
    surrogate_tokenizer = None
    if sdce_cfg.get('mode') not in ('none', None):
        print_rank0("\n=== Loading Surrogate Model ===")
        surrogate_model = load_model(
            surrogate_cfg['name_or_path'],
            storage_manager,
            device_manager,
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
        
        # Note: Don't wrap surrogate with DDP since it's frozen and only used for inference
        
        # Load surrogate tokenizer if different from student
        if surrogate_cfg['name_or_path'] != model_cfg['name_or_path']:
            surrogate_tokenizer = load_tokenizer(
                surrogate_cfg['name_or_path'],
                storage_manager,
                trust_remote_code=surrogate_cfg.get('trust_remote_code', False),
            )
    
    # Load dataset
    print_rank0("\n=== Loading Dataset ===")
    dataset = load_and_prepare_dataset(config, tokenizer, storage_manager)
    
    # Create data loaders
    train_batch_size = training_cfg.get('per_device_train_batch_size', 4)
    eval_batch_size = training_cfg.get('per_device_eval_batch_size', 4)
    
    from functools import partial
    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    
    train_loader, train_sampler = create_dataloader(
        dataset['train'],
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate,
        device_manager=device_manager,
        dist_manager=dist_manager,
        num_workers=4,
    )
    
    eval_loader, _ = create_dataloader(
        dataset['validation'],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        device_manager=device_manager,
        dist_manager=dist_manager,
        num_workers=4,
    )
    
    # Calculate training steps
    num_epochs = training_cfg.get('num_epochs', 1)
    gradient_accumulation_steps = training_cfg.get('gradient_accumulation_steps', 1)
    num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_training_steps = num_epochs * num_update_steps_per_epoch
    warmup_ratio = training_cfg.get('warmup_ratio', 0.01)
    num_warmup_steps = int(total_training_steps * warmup_ratio)
    
    print_rank0(f"\nTraining for {num_epochs} epochs")
    print_rank0(f"Steps per epoch: {num_update_steps_per_epoch:,}")
    print_rank0(f"Total training steps: {total_training_steps:,}")
    print_rank0(f"Warmup steps: {num_warmup_steps:,}")
    print_rank0(f"Effective batch size: {train_batch_size * gradient_accumulation_steps * dist_manager.world_size}")
    
    # Optimizer - use the underlying model's parameters if using DDP
    model_for_optimizer = student_model.module if hasattr(student_model, 'module') else student_model
    optimizer = AdamW(
        model_for_optimizer.parameters(),
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
    
    # Mixed precision setup
    mixed_precision = training_cfg.get('mixed_precision', 'bf16')
    autocast_dtype = device_manager.get_autocast_dtype(mixed_precision)
    scaler = device_manager.get_grad_scaler(mixed_precision)
    autocast_enabled = mixed_precision in ('fp16', 'bf16') and not device_manager.is_cpu
    
    print_rank0(f"\nMixed precision: {mixed_precision}")
    print_rank0(f"Autocast dtype: {autocast_dtype}")
    print_rank0(f"Autocast enabled: {autocast_enabled}")
    print_rank0(f"Gradient scaler: {'enabled' if scaler is not None else 'disabled'}")
    
    # Initialize wandb (only on main process)
    if WANDB_AVAILABLE and training_cfg.get('wandb_project') and is_main_process():
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
    if is_main_process():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist_manager.barrier()
    
    # Training loop
    print_rank0("\n=== Starting Training ===")
    global_step = 0
    best_eval_loss = float('inf')
    
    # Initialize training metrics tracker
    model_for_params = student_model.module if hasattr(student_model, 'module') else student_model
    num_params = sum(p.numel() for p in model_for_params.parameters())
    world_size = device_manager.get_world_size()
    metrics_tracker = TrainingMetrics(num_params=num_params, world_size=world_size)
    print_rank0(f"Metrics tracking: {num_params:,} params, {world_size} device(s)")
    
    # SDCE settings
    sdce_mode = sdce_cfg.get('mode', 'none')
    prob_threshold = surrogate_cfg.get('prob_threshold', 0.03)
    max_tokens = surrogate_cfg.get('max_tokens', 100)
    kd_temperature = sdce_cfg.get('kd_temperature', 4.0)
    use_z_loss = training_cfg.get('use_z_loss', False)
    z_loss_weight = training_cfg.get('z_loss_multiplier', 1e-4)
    
    for epoch in range(num_epochs):
        student_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Set epoch for distributed sampler
        dist_manager.set_epoch(train_sampler, epoch)
        
        # Only show progress bar on main process
        if is_main_process():
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        else:
            progress_bar = train_loader
        
        for step, batch in enumerate(progress_bar):
            # Start timing at beginning of accumulation window
            if step % gradient_accumulation_steps == 0:
                metrics_tracker.start_step()
            
            # Move batch to device (TPU ParallelLoader handles this automatically)
            if not device_manager.is_tpu:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
            else:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
            
            # Count tokens in this batch (non-padding tokens)
            batch_tokens = attention_mask.sum().item()
            metrics_tracker.add_tokens(batch_tokens)
            
            # Get surrogate logits if needed
            teacher_logits = None
            if surrogate_model is not None and sdce_mode != 'none':
                with torch.no_grad():
                    with device_manager.get_autocast_context(autocast_dtype, enabled=autocast_enabled):
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
            with device_manager.get_autocast_context(autocast_dtype, enabled=autocast_enabled):
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
                    sdce_mode=sdce_mode,
                    sdce_weight=surrogate_weight,
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
                
                # Compute gradient norm (returned by clip_grad_norm_)
                model_for_grad = student_model.module if hasattr(student_model, 'module') else student_model
                grad_norm = torch.nn.utils.clip_grad_norm_(model_for_grad.parameters(), max_grad_norm)
                
                # Optimizer step (device-aware)
                device_manager.optimizer_step(optimizer, scaler)
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # TPU: mark step for XLA compilation
                if device_manager.is_tpu:
                    xm.mark_step()
                
                # End step timing and get throughput metrics
                throughput_metrics = metrics_tracker.end_step()
                
                # Logging (only on main process)
                if global_step % training_cfg.get('logging_steps', 50) == 0 and is_main_process():
                    lr = scheduler.get_last_lr()[0]
                    
                    # Convert grad_norm to float
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    
                    log_dict = {
                        'train/loss': loss_dict['total_loss'].item(),
                        'train/ce_loss': loss_dict['ce_loss'].item(),
                        'train/learning_rate': lr,
                        'train/surrogate_weight': surrogate_weight,
                        'train/global_step': global_step,
                        # Gradient norm
                        'train/grad_norm': grad_norm_val,
                        # Throughput metrics
                        'throughput/tokens_per_sec': throughput_metrics.get('tokens_per_sec', 0),
                        'throughput/tokens_per_sec_per_device': throughput_metrics.get('tokens_per_sec_per_device', 0),
                        'throughput/tokens_per_sec_smoothed': throughput_metrics.get('tokens_per_sec_smoothed', 0),
                        'throughput/total_tokens': throughput_metrics.get('total_tokens', 0),
                        'throughput/total_flops': throughput_metrics.get('total_flops', 0),
                        'throughput/flops_per_sec': throughput_metrics.get('flops_per_sec', 0),
                        'throughput/flops_per_sec_per_device': throughput_metrics.get('flops_per_sec_per_device', 0),
                        'throughput/tflops_per_sec': throughput_metrics.get('tflops_per_sec', 0),
                        'throughput/tflops_per_sec_per_device': throughput_metrics.get('tflops_per_sec_per_device', 0),
                        'throughput/step_time_sec': throughput_metrics.get('step_time_sec', 0),
                    }
                    
                    if 'surrogate_loss' in loss_dict:
                        log_dict['train/surrogate_loss'] = loss_dict['surrogate_loss'].item()
                    if 'kd_loss' in loss_dict:
                        log_dict['train/kd_loss'] = loss_dict['kd_loss'].item()
                    if 'z_loss' in loss_dict:
                        log_dict['train/z_loss'] = loss_dict['z_loss'].item()
                    
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log(log_dict, step=global_step)
                    
                    # Format tokens/sec for display
                    tps = throughput_metrics.get('tokens_per_sec_smoothed', 0)
                    tps_str = f"{tps/1000:.1f}K" if tps >= 1000 else f"{tps:.0f}"
                    
                    progress_bar.set_postfix({
                        'loss': f"{loss_dict['total_loss'].item():.4f}",
                        'lr': f"{lr:.2e}",
                        'grad': f"{grad_norm_val:.2f}",
                        'tok/s': tps_str,
                    })
                
                # Evaluation
                if global_step % training_cfg.get('eval_steps', 2000) == 0:
                    eval_loss = evaluate(
                        student_model, eval_loader, device_manager, 
                        autocast_dtype, autocast_enabled, dist_manager
                    )
                    print_rank0(f"\nStep {global_step}: Eval Loss = {eval_loss:.4f}")
                    
                    if WANDB_AVAILABLE and wandb.run is not None and is_main_process():
                        wandb.log({'eval/loss': eval_loss}, step=global_step)
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        # Save best model (only on main process)
                        if is_main_process():
                            save_checkpoint(
                                student_model, tokenizer, optimizer, scheduler,
                                global_step, checkpoint_dir / 'best',
                            )
                    
                    dist_manager.barrier()
                    student_model.train()
                
                # Save checkpoint (only on main process)
                if global_step % training_cfg.get('save_steps', 5000) == 0:
                    if is_main_process():
                        save_checkpoint(
                            student_model, tokenizer, optimizer, scheduler,
                            global_step, checkpoint_dir / f'checkpoint-{global_step}',
                        )
                    dist_manager.barrier()
            
            epoch_loss += loss_dict['total_loss'].item()
            num_batches += 1
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        print_rank0(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Final save (only on main process)
    if is_main_process():
        save_checkpoint(
            student_model, tokenizer, optimizer, scheduler,
            global_step, checkpoint_dir / 'final',
        )
    dist_manager.barrier()
    
    # Get final metrics
    final_metrics = metrics_tracker.get_total_metrics()
    
    print_rank0("\n=== Training Complete ===")
    print_rank0(f"Best eval loss: {best_eval_loss:.4f}")
    print_rank0(f"Total tokens trained: {final_metrics['total_tokens']:,}")
    print_rank0(f"Total FLOPs: {final_metrics['total_flops']:.2e}")
    print_rank0(f"Checkpoints saved to: {checkpoint_dir}")
    
    if WANDB_AVAILABLE and wandb.run is not None and is_main_process():
        # Log final summary metrics
        wandb.log({
            'summary/total_tokens': final_metrics['total_tokens'],
            'summary/total_flops': final_metrics['total_flops'],
            'summary/best_eval_loss': best_eval_loss,
        })
        wandb.finish()
    
    # Cleanup distributed
    dist_manager.cleanup()


def evaluate(model, eval_loader, device_manager: DeviceManager, autocast_dtype, 
             autocast_enabled, dist_manager: DistributedManager):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    device = device_manager.device
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", disable=not is_main_process()):
            # Move batch to device (TPU ParallelLoader handles this automatically)
            if not device_manager.is_tpu:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
            else:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
            
            with device_manager.get_autocast_context(autocast_dtype, enabled=autocast_enabled):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # All-reduce the loss across processes in distributed mode
    if dist_manager.is_distributed:
        loss_tensor = torch.tensor([avg_loss, num_batches], device=device)
        dist_manager.all_reduce(loss_tensor)
        avg_loss = loss_tensor[0].item() / loss_tensor[1].item()
    
    return avg_loss


def save_checkpoint(model, tokenizer, optimizer, scheduler, step, path):
    """Save training checkpoint."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Get the underlying model if wrapped with DDP
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # Save model and tokenizer
    model_to_save.save_pretrained(path)
    tokenizer.save_pretrained(path)
    
    # Save training state
    torch.save({
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path / 'training_state.pt')
    
    print(f"Checkpoint saved to: {path}")


# =============================================================================
# TPU-specific Training Entry Point
# =============================================================================

def _mp_fn(index, config):
    """
    Entry point for TPU multiprocessing.
    Called by xmp.spawn for each TPU core.
    """
    # Set the device for this process
    device = xm.xla_device()
    print(f"TPU process {index} using device: {device}")
    
    # Run training
    train(config)


def train_tpu(config: Dict[str, Any], num_cores: int = 8):
    """
    Launch training on TPU with multiprocessing.
    
    Args:
        config: Configuration dictionary
        num_cores: Number of TPU cores to use (default: 8 for TPU v3)
    """
    if not TPU_AVAILABLE:
        raise RuntimeError("TPU training requested but torch_xla is not installed")
    
    print(f"Launching TPU training on {num_cores} cores...")
    xmp.spawn(_mp_fn, args=(config,), nprocs=num_cores)


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
    parser.add_argument(
        '--device', type=str, default=None,
        help='Override device (auto, cuda, mps, tpu, cpu)'
    )
    parser.add_argument(
        '--tpu-cores', type=int, default=8,
        help='Number of TPU cores to use (only for TPU training)'
    )
    # Note: --local_rank is automatically added by torch.distributed.launch
    # torchrun uses environment variables instead
    parser.add_argument(
        '--local_rank', type=int, default=-1,
        help='Local rank for distributed training (set automatically by torch.distributed.launch)'
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
    
    # Override device if provided
    if args.device:
        config['training']['device'] = args.device
    
    # Check if TPU training with multiprocessing
    device_str = config['training'].get('device', 'auto')
    if device_str == 'tpu' or (device_str == 'auto' and TPU_AVAILABLE and os.environ.get("USE_TPU", "").lower() in ("1", "true")):
        train_tpu(config, num_cores=args.tpu_cores)
    else:
        # Standard training (handles both single-process and torchrun)
        train(config)


if __name__ == '__main__':
    main()
