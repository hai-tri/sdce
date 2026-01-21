"""
Loss functions for surrogate-assisted language model training.

Includes:
- Cross-entropy loss (standard LM loss)
- Surrogate-guided distillation loss (SDCE)
- Knowledge distillation loss (Hinton et al., 2015)
- Z-loss (PaLM auxiliary loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute standard cross-entropy loss for language modeling.
    
    Args:
        logits: Model logits of shape (batch_size, seq_len, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_len)
        ignore_index: Label index to ignore (padding tokens)
    
    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM: predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten for loss computation
    vocab_size = shift_logits.size(-1)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index,
        reduction="mean",
    )
    
    return loss


def compute_surrogate_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    prob_threshold: float = 0.03,
    max_tokens: int = 100,
    ignore_index: int = -100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute surrogate-guided distillation cross-entropy (SDCE) loss.
    
    This loss focuses the student model on tokens that the surrogate/teacher
    model considers probable, weighted by the surrogate's confidence.
    
    Args:
        student_logits: Student model logits (batch, seq_len, vocab_size)
        teacher_logits: Teacher/surrogate model logits (batch, seq_len, vocab_size)
        labels: Ground truth labels (batch, seq_len)
        prob_threshold: Minimum probability threshold for token selection
        max_tokens: Maximum number of tokens to consider per position
        ignore_index: Label index to ignore
        temperature: Temperature for softening distributions
    
    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM
    student_logits = student_logits[..., :-1, :].contiguous()
    teacher_logits = teacher_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # Get teacher probabilities
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Create mask for valid positions (not padding)
    valid_mask = (labels != ignore_index).float()
    
    # Get student log probabilities
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    # For each position, compute loss weighted by teacher probabilities
    # Only consider tokens above threshold
    
    # Create threshold mask
    threshold_mask = (teacher_probs > prob_threshold).float()
    
    # Limit to top-k tokens if needed (for memory efficiency)
    if max_tokens < vocab_size:
        # Get top-k teacher probs
        top_probs, top_indices = torch.topk(teacher_probs, k=max_tokens, dim=-1)
        
        # Create sparse mask
        topk_mask = torch.zeros_like(teacher_probs)
        topk_mask.scatter_(-1, top_indices, 1.0)
        
        # Combine with threshold mask
        threshold_mask = threshold_mask * topk_mask
    
    # Compute weighted cross-entropy
    # Loss = -sum(teacher_prob * student_log_prob) for selected tokens
    weighted_loss = -(teacher_probs * student_log_probs * threshold_mask).sum(dim=-1)
    
    # Normalize by number of selected tokens per position (avoid division by zero)
    num_selected = threshold_mask.sum(dim=-1).clamp(min=1e-8)
    weighted_loss = weighted_loss / num_selected
    
    # Apply valid mask and compute mean
    weighted_loss = weighted_loss * valid_mask
    loss = weighted_loss.sum() / valid_mask.sum().clamp(min=1e-8)
    
    return loss


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute standard knowledge distillation loss (Hinton et al., 2015).
    
    Uses KL divergence between softened teacher and student distributions.
    
    Args:
        student_logits: Student model logits (batch, seq_len, vocab_size)
        teacher_logits: Teacher model logits (batch, seq_len, vocab_size)
        labels: Ground truth labels (for masking padding)
        temperature: Temperature for softening distributions
        ignore_index: Label index to ignore
    
    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM
    student_logits = student_logits[..., :-1, :].contiguous()
    teacher_logits = teacher_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    
    # Create mask for valid positions
    valid_mask = (labels != ignore_index).float()
    
    # Compute softened distributions
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL divergence: sum(p * log(p/q)) = sum(p * log(p)) - sum(p * log(q))
    # Since we want gradient to flow to student only:
    # KL = -sum(teacher_probs * student_log_probs) + const
    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)
    
    # Apply mask and compute mean
    kl_div = kl_div * valid_mask
    
    # Scale by T^2 as per original KD paper
    loss = (temperature ** 2) * kl_div.sum() / valid_mask.sum().clamp(min=1e-8)
    
    return loss


def compute_z_loss(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute z-loss (auxiliary loss from PaLM).
    
    The z-loss encourages the logits to stay small, which helps
    with training stability, especially for large models.
    
    z_loss = mean(log(sum(exp(logits))))^2
    
    Args:
        logits: Model logits (batch, seq_len, vocab_size)
        labels: Optional labels for masking (if provided)
        ignore_index: Label index to ignore
    
    Returns:
        Scalar loss tensor
    """
    # Shift logits for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    
    # Compute log-sum-exp (numerically stable)
    log_z = torch.logsumexp(shift_logits, dim=-1)
    
    # Square it
    z_loss = log_z ** 2
    
    # Apply mask if labels provided
    if labels is not None:
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = (shift_labels != ignore_index).float()
        z_loss = z_loss * valid_mask
        z_loss = z_loss.sum() / valid_mask.sum().clamp(min=1e-8)
    else:
        z_loss = z_loss.mean()
    
    return z_loss


def compute_combined_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor],
    labels: torch.Tensor,
    distillation_mode: str = "surrogate",
    distillation_weight: float = 0.5,
    kd_temperature: float = 4.0,
    prob_threshold: float = 0.03,
    max_tokens: int = 100,
    use_z_loss: bool = False,
    z_loss_weight: float = 1e-4,
    ignore_index: int = -100,
) -> dict:
    """
    Compute combined loss with optional distillation and z-loss.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits (optional)
        labels: Ground truth labels
        distillation_mode: "surrogate", "kd", or "none"
        distillation_weight: Weight for distillation loss
        kd_temperature: Temperature for KD
        prob_threshold: Probability threshold for surrogate
        max_tokens: Max tokens for surrogate
        use_z_loss: Whether to add z-loss
        z_loss_weight: Weight for z-loss
        ignore_index: Label index to ignore
    
    Returns:
        Dictionary with total_loss and individual loss components
    """
    # Base cross-entropy loss
    ce_loss = compute_cross_entropy_loss(student_logits, labels, ignore_index)
    total_loss = ce_loss
    
    result = {
        "total_loss": total_loss,
        "ce_loss": ce_loss,
    }
    
    # Distillation loss
    if teacher_logits is not None and distillation_mode != "none":
        if distillation_mode == "surrogate":
            distill_loss = compute_surrogate_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                prob_threshold=prob_threshold,
                max_tokens=max_tokens,
                ignore_index=ignore_index,
            )
            total_loss = ce_loss + distillation_weight * distill_loss
            result["surrogate_loss"] = distill_loss
            
        elif distillation_mode == "kd":
            distill_loss = compute_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=kd_temperature,
                ignore_index=ignore_index,
            )
            total_loss = (1 - distillation_weight) * ce_loss + distillation_weight * distill_loss
            result["kd_loss"] = distill_loss
        
        result["total_loss"] = total_loss
    
    # Z-loss
    if use_z_loss:
        z_loss = compute_z_loss(student_logits, labels, ignore_index)
        total_loss = total_loss + z_loss_weight * z_loss
        result["z_loss"] = z_loss
        result["total_loss"] = total_loss
    
    return result
