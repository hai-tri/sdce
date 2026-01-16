"""
Custom Cross-Entropy Loss with Surrogate Guidance

This module provides loss functions for language model training with optional
surrogate model guidance.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    perp_values: Optional[torch.Tensor] = None,
    perp_indices: Optional[torch.Tensor] = None,
    lookup_surrogate_to_self_tokens: Optional[torch.Tensor] = None,
    surrogate_weight: float = 1.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Cross-entropy loss with optional surrogate-guided auxiliary term.
    
    This function computes the standard cross-entropy loss and optionally adds
    a surrogate-guided term that encourages the model to assign probability mass
    to tokens that the surrogate model finds important.
    
    Args:
        logits: Model output logits. Shape can be:
            - (batch_size * seq_len, vocab_size) for flattened inputs, or
            - (batch_size, seq_len, vocab_size) when using surrogate guidance
        labels: Target labels. Shape matches logits batch dimensions.
        perp_values: Perplexity values from surrogate model.
            Shape: (batch_size, seq_len, max_tokens) where max_tokens is variable
            based on probability threshold selection. Invalid entries are inf.
        perp_indices: Token indices from surrogate model (in surrogate vocab).
            Shape: (batch_size, seq_len, max_tokens). Invalid entries are -1.
        lookup_surrogate_to_self_tokens: Lookup table mapping surrogate vocabulary
            indices to base model vocabulary indices. Shape: (surrogate_vocab_size,)
        surrogate_weight: Weight for the surrogate loss term (decays over training via cosine schedule)
        ignore_index: Index to ignore in loss computation (typically -100 for padding)
        reduction: How to reduce the loss - "mean", "sum", or "none"
        compute_z_loss: Whether to compute auxiliary z-loss (from PaLM paper)
        z_loss_multiplier: Coefficient for z-loss term
        
    Returns:
        Tuple of (loss, z_loss) where z_loss is None if compute_z_loss=False
        
    Example:
        >>> # Standard cross-entropy (no surrogate)
        >>> loss, _ = cross_entropy_loss(logits, labels)
        
        >>> # With surrogate guidance (threshold-based selection)
        >>> loss, z_loss = cross_entropy_loss(
        ...     logits, labels,
        ...     perp_values=surr_perp_values,
        ...     perp_indices=surr_perp_indices,
        ...     lookup_surrogate_to_self_tokens=lookup_table,
        ...     surrogate_weight=0.5,  # Decayed weight
        ...     compute_z_loss=True
        ... )
    """
    device = logits.device
    
    if perp_indices is not None and lookup_surrogate_to_self_tokens is not None and surrogate_weight > 0:
        # =====================================================================
        # Surrogate-guided loss computation
        # =====================================================================
        
        # Get dimensions
        if logits.dim() == 2:
            # Need to infer batch/seq dimensions from perp_indices
            batch_size, seq_len, max_tokens = perp_indices.shape
            vocab_size = logits.shape[-1]
            logits = logits.view(batch_size, seq_len, vocab_size)
            labels = labels.view(batch_size, seq_len)
        else:
            batch_size, seq_len, vocab_size = logits.shape
            max_tokens = perp_indices.shape[-1]
        
        # Translate surrogate indices to base model vocabulary
        # Handle invalid indices (-1) by clamping first
        valid_indices_mask = perp_indices >= 0
        safe_perp_indices = perp_indices.clamp(min=0)
        translated_perp_indices = lookup_surrogate_to_self_tokens[safe_perp_indices]
        translated_perp_indices[~valid_indices_mask] = -100  # Mark invalid
        
        # Create mask for valid translations (not -100 and within vocab)
        valid_translation_mask = (translated_perp_indices >= 0) & (translated_perp_indices < vocab_size)
        
        # Clamp to valid range for gather operation (will be masked out anyway)
        safe_translated_indices = translated_perp_indices.clamp(0, vocab_size - 1)
        
        # Gather logits for the translated tokens
        # gathered_logits: (batch_size, seq_len, max_tokens)
        gathered_logits = torch.gather(logits, dim=2, index=safe_translated_indices)
        
        # Compute log probabilities for gathered tokens
        gathered_log_probs = F.log_softmax(gathered_logits, dim=-1)
        gathered_nll = -gathered_log_probs  # Negative log likelihood
        
        # Identify rows where all perplexity values are inf (invalid positions)
        # This happens when the label is -100 or no valid surrogate tokens exist
        all_inf_mask = torch.isinf(perp_values).all(dim=-1)  # (batch_size, seq_len)
        
        # Create combined validity mask
        # A position is valid if:
        # 1. Not all perp values are inf
        # 2. The translation is valid
        # 3. The perp value itself is not inf (threshold-based selection)
        row_valid_mask = ~all_inf_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        token_valid_mask = ~torch.isinf(perp_values)  # (batch_size, seq_len, max_tokens)
        combined_mask = row_valid_mask & valid_translation_mask & token_valid_mask
        
        # Count valid surrogate entries for normalization
        num_valid_surrogate = combined_mask.sum().item()
        
        # Compute softmax weights from perplexity
        # Lower perplexity = higher weight (use negative perplexity in softmax)
        # Mask invalid entries before computing softmax
        masked_perp = perp_values.clone()
        masked_perp[~combined_mask] = float('inf')
        
        # Softmax over the token dimension (the selected tokens)
        softmax_weights = F.softmax(-masked_perp, dim=-1)
        
        # Zero out invalid positions
        softmax_weights = softmax_weights * combined_mask.float()
        
        # Compute weighted surrogate loss (scaled by surrogate_weight)
        weighted_nll = gathered_nll * softmax_weights
        surrogate_loss_sum = surrogate_weight * weighted_nll.sum()
        
        # Compute standard cross-entropy loss
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        ce_loss_sum = F.cross_entropy(
            logits_flat, labels_flat,
            ignore_index=ignore_index,
            reduction='sum'
        )
        
        # Count valid labels for normalization
        num_valid_labels = (labels_flat != ignore_index).sum().item()
        
        # Combine losses
        total_loss = ce_loss_sum + surrogate_loss_sum
        
        if reduction == "mean":
            # Normalize by total number of valid entries
            total_count = num_valid_labels + num_valid_surrogate
            loss = total_loss / (total_count + 1e-8)
        elif reduction == "sum":
            loss = total_loss
        else:  # reduction == "none"
            # Return per-position losses
            ce_per_position = F.cross_entropy(
                logits_flat, labels_flat,
                ignore_index=ignore_index,
                reduction='none'
            ).view(batch_size, seq_len)
            surrogate_per_position = weighted_nll.sum(dim=-1)
            loss = ce_per_position + surrogate_per_position
            
    else:
        # =====================================================================
        # Standard cross-entropy loss (no surrogate guidance)
        # =====================================================================
        loss = F.cross_entropy(
            logits, labels,
            ignore_index=ignore_index,
            reduction=reduction
        )
    
    # =========================================================================
    # Optional Z-loss computation (from PaLM paper)
    # =========================================================================
    z_loss = None
    if compute_z_loss:
        # Z-loss penalizes large logits to stabilize training
        # z = logsumexp(logits) => z_loss = z^2
        if logits.dim() == 2:
            z_squared = logits.logsumexp(dim=-1).pow(2)
            label_mask = (labels != ignore_index).float()
        else:
            z_squared = logits.view(-1, logits.size(-1)).logsumexp(dim=-1).pow(2)
            label_mask = (labels.view(-1) != ignore_index).float()
        
        if reduction == "mean":
            z_loss = z_loss_multiplier * (z_squared * label_mask).sum() / (label_mask.sum() + 1e-8)
        elif reduction == "sum":
            z_loss = z_loss_multiplier * (z_squared * label_mask).sum()
        else:  # reduction == "none"
            z_loss = z_loss_multiplier * z_squared
    
    return loss, z_loss


def compute_perplexity_guidance(
    surrogate_logits: torch.Tensor,
    labels: torch.Tensor,
    lookup_base_to_surrogate: torch.Tensor,
    permitted_surrogate_ids: torch.Tensor,
    prob_threshold: float,
    max_tokens: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute probability threshold-based guidance from surrogate model.
    
    Instead of selecting a fixed top-k tokens, this function selects all tokens
    whose probability exceeds the specified threshold.
    
    Args:
        surrogate_logits: Logits from surrogate model. Shape: (batch, seq_len, surrogate_vocab)
        labels: Target labels in base model vocabulary. Shape: (batch, seq_len)
        lookup_base_to_surrogate: Lookup table from base vocab to surrogate vocab
        permitted_surrogate_ids: Surrogate token IDs that are in vocabulary intersection
        prob_threshold: Probability threshold for token selection (e.g., 0.03 selects
            tokens with probability > 0.03)
        max_tokens: Maximum number of tokens to select per position (for memory efficiency).
            Positions with more tokens above threshold will keep only the top max_tokens.
        
    Returns:
        Tuple of:
        - perp_values: Perplexity values for selected tokens. Shape: (batch, seq_len, max_tokens)
            Invalid/padded entries are set to inf.
        - perp_indices: Surrogate vocab indices for selected tokens. Shape: (batch, seq_len, max_tokens)
            Invalid/padded entries are set to -1.
    """
    device = surrogate_logits.device
    batch_size, seq_len, vocab_size = surrogate_logits.shape
    
    # Compute probabilities from logits
    surrogate_probs = F.softmax(surrogate_logits, dim=-1)
    
    # Compute perplexity (reciprocal of probability)
    surrogate_perp = torch.reciprocal(surrogate_probs + 1e-8)
    
    # Mask tokens not in vocabulary intersection
    vocab_indices = torch.arange(vocab_size, device=device)
    not_in_intersection = ~torch.isin(vocab_indices, permitted_surrogate_ids)
    surrogate_probs[:, :, not_in_intersection] = 0.0
    surrogate_perp[:, :, not_in_intersection] = float('inf')
    
    # Mask positions where labels are invalid
    invalid_label_mask = (labels == -100).unsqueeze(-1)
    surrogate_probs = surrogate_probs.masked_fill(invalid_label_mask, 0.0)
    surrogate_perp = surrogate_perp.masked_fill(invalid_label_mask, float('inf'))
    
    # Translate labels to surrogate vocab and mask them out
    # (we don't want to include the actual label in selection)
    safe_labels = labels.clone()
    safe_labels[labels == -100] = 0
    translated_labels = lookup_base_to_surrogate[safe_labels]
    
    valid_translation_mask = translated_labels != -100
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
    seq_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create safe indices for scatter
    safe_translated = translated_labels.clone()
    safe_translated[~valid_translation_mask] = 0
    
    # Mask out the actual labels (set their probability to 0)
    surrogate_probs[batch_idx, seq_idx, safe_translated] = 0.0
    surrogate_perp[batch_idx, seq_idx, safe_translated] = float('inf')
    
    # Create mask for tokens above probability threshold
    above_threshold_mask = surrogate_probs > prob_threshold
    
    # Set perplexity to inf for tokens below threshold
    surrogate_perp[~above_threshold_mask] = float('inf')
    
    # Get the top max_tokens lowest perplexity tokens (these are the ones above threshold)
    # Using topk with largest=False gives us the smallest perplexities (highest probs)
    topk_result = torch.topk(surrogate_perp, k=max_tokens, largest=False, sorted=True, dim=-1)
    
    perp_values = topk_result.values  # (batch, seq_len, max_tokens)
    perp_indices = topk_result.indices  # (batch, seq_len, max_tokens)
    
    # Mark indices as invalid (-1) where perplexity is inf (below threshold or invalid)
    invalid_mask = torch.isinf(perp_values)
    perp_indices = perp_indices.masked_fill(invalid_mask, -1)
    
    return perp_values, perp_indices


class SurrogateCrossEntropyLoss(torch.nn.Module):
    """
    PyTorch module wrapper for surrogate-guided cross-entropy loss.
    
    Example:
        >>> criterion = SurrogateCrossEntropyLoss(
        ...     lookup_table=lookup_surrogate_to_base,
        ...     compute_z_loss=True
        ... )
        >>> loss, z_loss = criterion(logits, labels, perp_values, perp_indices)
    """
    
    def __init__(
        self,
        lookup_table: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 1e-4,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.compute_z_loss = compute_z_loss
        self.z_loss_multiplier = z_loss_multiplier
        
        if lookup_table is not None:
            self.register_buffer('lookup_table', lookup_table)
        else:
            self.lookup_table = None
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        perp_values: Optional[torch.Tensor] = None,
        perp_indices: Optional[torch.Tensor] = None,
        surrogate_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return cross_entropy_loss(
            logits=logits,
            labels=labels,
            perp_values=perp_values,
            perp_indices=perp_indices,
            lookup_surrogate_to_self_tokens=self.lookup_table,
            surrogate_weight=surrogate_weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            compute_z_loss=self.compute_z_loss,
            z_loss_multiplier=self.z_loss_multiplier,
        )
