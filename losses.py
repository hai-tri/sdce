"""
Loss functions for surrogate-assisted language model training.

Includes:
- Cross-entropy loss (standard LM loss)
- Surrogate-guided distillation loss (SDCE)
- Unified CE + surrogate loss with combined normalization
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


def get_surrogate_weight(
    current_step: int,
    total_steps: int,
    initial_weight: float = 1.0,
    final_weight: float = 0.0,
    schedule: str = "cosine",
) -> float:
    """
    Compute the surrogate loss weight at a given training step.
    
    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        initial_weight: Starting weight for surrogate loss
        final_weight: Final weight for surrogate loss
        schedule: Schedule type - "cosine", "linear", or "constant"
    
    Returns:
        Weight value for current step
    """
    if schedule == "constant":
        return initial_weight
    
    # Clamp progress to [0, 1]
    progress = min(max(current_step / max(total_steps, 1), 0.0), 1.0)
    
    if schedule == "linear":
        return initial_weight + (final_weight - initial_weight) * progress
    
    elif schedule == "cosine":
        # Cosine annealing: starts slow, accelerates, then slows again
        import math
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return final_weight + (initial_weight - final_weight) * cosine_decay
    
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Use 'cosine', 'linear', or 'constant'.")


def compute_unified_surrogate_ce_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    prob_threshold: float = 0.03,
    max_tokens: int = 100,
    ignore_index: int = -100,
    temperature: float = 1.0,
    surrogate_weight: float = 1.0,
) -> dict:
    """
    Compute unified CE + surrogate loss with combined normalization.
    
    Instead of separately normalizing CE and surrogate losses, this function
    combines both into a single loss normalized by the total number of targets:
    
        L = (CE_sum + λ * surrogate_sum) / (N_base + N_aux)
    
    where:
        - N_base = number of valid positions (ground-truth targets)
        - N_aux = total number of selected surrogate tokens across all positions
        - λ = surrogate_weight (from cosine scheduler, applied to numerator only)
    
    Args:
        student_logits: Student model logits (batch, seq_len, vocab_size)
        teacher_logits: Teacher/surrogate model logits (batch, seq_len, vocab_size)
        labels: Ground truth labels (batch, seq_len)
        prob_threshold: Minimum probability threshold for token selection
        max_tokens: Maximum number of tokens to consider per position
        ignore_index: Label index to ignore
        temperature: Temperature for softening distributions
        surrogate_weight: Weight for surrogate loss (numerator only, e.g. from cosine scheduler)
    
    Returns:
        Dictionary with:
            - total_loss: Combined normalized loss
            - ce_loss: Individually normalized CE loss (for logging)
            - surrogate_loss: Individually normalized surrogate loss (for logging)
            - ce_loss_unnorm: Unnormalized sum of CE losses
            - surrogate_loss_unnorm: Unnormalized sum of surrogate losses
            - num_base_targets: Number of ground-truth targets
            - num_aux_targets: Number of auxiliary (surrogate-selected) targets
    """
    # Shift for causal LM
    student_logits = student_logits[..., :-1, :].contiguous()
    teacher_logits = teacher_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # Create mask for valid positions (not padding)
    valid_mask = (labels != ignore_index)
    valid_mask_float = valid_mask.float()
    
    # ==================== CE Loss (unnormalized) ====================
    # Get student log probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    
    # Gather log probs for ground-truth labels
    # Handle ignore_index by clamping labels to valid range
    safe_labels = labels.clamp(min=0)
    ce_per_token = -student_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    
    # Zero out padded positions
    ce_per_token = ce_per_token * valid_mask_float
    
    # Sum (not mean) for unnormalized CE loss
    ce_loss_unnorm = ce_per_token.sum()
    num_base_targets = valid_mask_float.sum()
    
    # ==================== Surrogate Loss (unnormalized) ====================
    # Get teacher probabilities
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Get student log probabilities (with temperature if needed)
    if temperature != 1.0:
        student_log_probs_t = F.log_softmax(student_logits / temperature, dim=-1)
    else:
        student_log_probs_t = student_log_probs
    
    # Create threshold mask
    threshold_mask = (teacher_probs > prob_threshold).float()
    
    # Limit to top-k tokens if needed (for memory efficiency)
    if max_tokens < vocab_size:
        top_probs, top_indices = torch.topk(teacher_probs, k=max_tokens, dim=-1)
        topk_mask = torch.zeros_like(teacher_probs)
        topk_mask.scatter_(-1, top_indices, 1.0)
        threshold_mask = threshold_mask * topk_mask
    
    # Apply valid mask to threshold mask (zero out padded positions entirely)
    threshold_mask = threshold_mask * valid_mask_float.unsqueeze(-1)
    
    # Compute surrogate loss: -sum(teacher_prob * student_log_prob) for selected tokens
    # This is summed over all selected tokens (not normalized per position)
    surrogate_loss_unnorm = -(teacher_probs * student_log_probs_t * threshold_mask).sum()
    
    # Count total auxiliary targets (sum of selected tokens across all valid positions)
    num_aux_targets = threshold_mask.sum()
    
    # ==================== Combined Normalization ====================
    # L = (CE_sum + λ * surrogate_sum) / (N_base + N_aux)
    # Weight only in numerator - controls surrogate contribution
    # Denominator normalizes by total target count uniformly
    total_numerator = ce_loss_unnorm + surrogate_weight * surrogate_loss_unnorm
    total_denominator = (num_base_targets + num_aux_targets).clamp(min=1e-8)
    
    total_loss = total_numerator / total_denominator
    
    # Also compute individually normalized losses for logging
    ce_loss_normalized = ce_loss_unnorm / num_base_targets.clamp(min=1e-8)
    surrogate_loss_normalized = surrogate_loss_unnorm / num_aux_targets.clamp(min=1e-8)
    
    return {
        "total_loss": total_loss,
        "ce_loss": ce_loss_normalized,
        "surrogate_loss": surrogate_loss_normalized,
        "ce_loss_unnorm": ce_loss_unnorm,
        "surrogate_loss_unnorm": surrogate_loss_unnorm,
        "num_base_targets": num_base_targets,
        "num_aux_targets": num_aux_targets,
    }


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
    sdce_mode: str = "surrogate",
    sdce_weight: float = 0.5,
    kd_temperature: float = 4.0,
    prob_threshold: float = 0.03,
    max_tokens: int = 100,
    use_z_loss: bool = False,
    z_loss_weight: float = 1e-4,
    ignore_index: int = -100,
    unified_normalization: bool = False,
) -> dict:
    """
    Compute combined loss with optional SDCE/KD and z-loss.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits (optional)
        labels: Ground truth labels
        sdce_mode: "surrogate", "kd", or "none"
        sdce_weight: Weight for SDCE loss (or auxiliary_weight if unified)
        kd_temperature: Temperature for KD
        prob_threshold: Probability threshold for surrogate
        max_tokens: Max tokens for surrogate
        use_z_loss: Whether to add z-loss
        z_loss_weight: Weight for z-loss
        ignore_index: Label index to ignore
        unified_normalization: If True, use unified normalization for surrogate mode
    
    Returns:
        Dictionary with total_loss and individual loss components
    """
    # Use unified normalization for surrogate mode if requested
    if (unified_normalization and 
        teacher_logits is not None and 
        sdce_mode == "surrogate"):
        
        result = compute_unified_surrogate_ce_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            prob_threshold=prob_threshold,
            max_tokens=max_tokens,
            ignore_index=ignore_index,
            surrogate_weight=sdce_weight,
        )
        
        # Add z-loss if requested
        if use_z_loss:
            z_loss = compute_z_loss(student_logits, labels, ignore_index)
            result["total_loss"] = result["total_loss"] + z_loss_weight * z_loss
            result["z_loss"] = z_loss
        
        return result
    
    # Original behavior: separate normalization
    # Base cross-entropy loss
    ce_loss = compute_cross_entropy_loss(student_logits, labels, ignore_index)
    total_loss = ce_loss
    
    result = {
        "total_loss": total_loss,
        "ce_loss": ce_loss,
    }
    
    # SDCE/KD loss
    if teacher_logits is not None and sdce_mode != "none":
        if sdce_mode == "surrogate":
            sdce_loss = compute_surrogate_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                prob_threshold=prob_threshold,
                max_tokens=max_tokens,
                ignore_index=ignore_index,
            )
            total_loss = ce_loss + sdce_weight * sdce_loss
            result["surrogate_loss"] = sdce_loss
            
        elif sdce_mode == "kd":
            kd_loss = compute_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=kd_temperature,
                ignore_index=ignore_index,
            )
            total_loss = (1 - sdce_weight) * ce_loss + sdce_weight * kd_loss
            result["kd_loss"] = kd_loss
        
        result["total_loss"] = total_loss
    
    # Z-loss
    if use_z_loss:
        z_loss = compute_z_loss(student_logits, labels, ignore_index)
        total_loss = total_loss + z_loss_weight * z_loss
        result["z_loss"] = z_loss
        result["total_loss"] = total_loss
    
    return result
