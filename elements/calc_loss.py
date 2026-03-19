import torch
import torch.utils
import torch.utils.data
import torch.nn as nn

NoneType = type(None)
import torch.nn.functional as F
from typing import Callable
import numpy as np


def calculate_class_weights(dataset, method='inverse', power=1.0, min_weight=0.01):
    """
    Calculate class weights for imbalanced datasets.

    Args:
        dataset: Dataset with labels attribute (N, num_classes) or Subset of such dataset
        method: 'inverse' or 'effective'
            - inverse: weight = (1 / frequency) ^ power
            - effective: weight = (1 - beta) / (1 - beta^n) where beta=0.9999
        power: Power for inverse frequency weighting (0.5-2.0)
        min_weight: Minimum weight to prevent extreme values

    Returns:
        torch.Tensor: Class weights of shape (num_classes,)
    """
    # Handle Subset wrapper
    if hasattr(dataset, 'dataset'):
        # This is a Subset, get the underlying dataset
        base_dataset = dataset.dataset
        indices = dataset.indices
        # Get labels for the subset
        if hasattr(base_dataset, 'labels'):
            labels = base_dataset.labels[indices]
        else:
            raise ValueError("Base dataset must have 'labels' attribute")
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        raise ValueError("Dataset must have 'labels' attribute")

    # Calculate class frequencies (mean across all samples)
    if isinstance(labels, torch.Tensor):
        # Handle both 2D (N, num_classes) and 4D (N, num_classes, H, W) labels
        if labels.ndim == 4:
            # For pixel-level labels, average over spatial dimensions first
            class_freq = labels.mean(dim=(0, 2, 3)).cpu().numpy()
        elif labels.ndim == 2:
            # For tile-level labels
            class_freq = labels.mean(dim=0).cpu().numpy()
        else:
            raise ValueError(f"Unexpected label shape: {labels.shape}. Expected 2D or 4D.")
    else:
        # NumPy array
        if labels.ndim == 4:
            class_freq = np.mean(labels, axis=(0, 2, 3))
        elif labels.ndim == 2:
            class_freq = np.mean(labels, axis=0)
        else:
            raise ValueError(f"Unexpected label shape: {labels.shape}. Expected 2D or 4D.")

    num_classes = len(class_freq)

    # Print for debugging
    print(f"Class frequencies: {class_freq}")

    # Identify valid classes (freq > threshold to exclude near-empty classes)
    # Use a small threshold to catch classes with almost no samples
    freq_threshold = 0.001  # 0.1% - classes below this are considered invalid
    valid_mask = class_freq > freq_threshold

    # Initialize weights
    weights = np.zeros(num_classes, dtype=np.float32)

    if method == 'inverse':
        # Inverse frequency: weight = (1 / frequency) ^ power
        # Only calculate for valid classes
        weights[valid_mask] = (1.0 / class_freq[valid_mask]) ** power

    elif method == 'effective':
        # Effective number of samples: weight = (1 - beta) / (1 - beta^n)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_freq[valid_mask] * len(labels))
        weights[valid_mask] = (1.0 - beta) / (effective_num + 1e-8)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'inverse' or 'effective'")

    print(f"Raw weights (before normalization): {weights}")

    # Normalize weights to have mean = 1.0 (only for valid classes)
    if valid_mask.sum() > 0:
        mean_weight = weights[valid_mask].mean()
        weights[valid_mask] = weights[valid_mask] / mean_weight
        # Set weight=0 for invalid classes (they will be ignored anyway)
        weights[~valid_mask] = 0.0

    print(f"Normalized weights: {weights}")

    # Apply minimum weight threshold (only to valid classes)
    weights[valid_mask] = np.maximum(weights[valid_mask], min_weight)

    print(f"Final weights (after min_weight={min_weight}): {weights}")

    return torch.tensor(weights, dtype=torch.float32)


class MaskedKLDivergenceLoss(nn.Module):
    def __init__(self, ignore_indices=None, class_weights=None):
        """
        Custom KL divergence loss function that ignores the specified classes.

        For pixel-level segmentation, completely excludes pixels where ground truth 
        is in ignore_indices (traditional semantic segmentation behavior).
        
        For tile-level classification, excludes the specified class columns from loss.

        :param ignore_indices: List of indices of classes to ignore in loss computation.
                              For pixel-level: pixels with GT in these classes are excluded.
                              For tile-level: these class columns are excluded.
        :param class_weights: Optional tensor of shape (num_classes,) for class weighting.
        """
        super(MaskedKLDivergenceLoss, self).__init__()
        self.ignore_indices = ignore_indices if ignore_indices is not None else []
        self.class_weights = class_weights

    def forward(self, outputs, targets):
        """
        Forward pass for masked KL divergence loss.

        Supports both:
        - Tile-level: outputs (B, C), targets (B, C)
        - Pixel-level: outputs (B, C, H, W), targets (B, C, H, W)
        
        For pixel-level, completely excludes pixels where ground truth is in ignore_indices.
        This follows traditional semantic segmentation ignore_index behavior.
        """
        # Handle both 2D (tile-level) and 4D (pixel-level) inputs
        if outputs.ndim == 4 and targets.ndim == 4:
            # Pixel-level predictions: (B, C, H, W)
            batch_size, num_classes, height, width = outputs.shape

            # Reshape to (B*H*W, C) for easier processing
            outputs_flat = outputs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
            targets_flat = targets.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)

            # Create pixel-level mask: exclude pixels where GT is in ignore_indices
            # This is the traditional semantic segmentation behavior
            gt_classes = targets_flat.argmax(dim=1)  # (B*H*W,)
            valid_pixels = torch.ones(gt_classes.shape[0], dtype=torch.bool, device=gt_classes.device)
            for index in self.ignore_indices:
                valid_pixels &= (gt_classes != index)
            
            # Apply pixel mask
            if valid_pixels.sum() == 0:
                # All pixels are ignored, return zero loss
                return torch.tensor(0.0, device=outputs.device, requires_grad=True)
            
            masked_outputs = outputs_flat[valid_pixels]  # (N_valid, C)
            masked_targets = targets_flat[valid_pixels]  # (N_valid, C)

            # Compute KLD loss
            loss = F.kl_div(F.log_softmax(masked_outputs, dim=-1), masked_targets, reduction='none')

            # Apply class weights if provided
            if self.class_weights is not None:
                # Use full class weights (don't remove ignored classes)
                loss = loss * self.class_weights.to(loss.device).unsqueeze(0)

            return loss.mean()

        elif outputs.ndim == 2 and targets.ndim == 2:
            # Tile-level predictions: (B, C) - original implementation
            # create a mask to exclude ignored indices
            mask = torch.ones_like(targets, dtype=torch.bool)
            for index in self.ignore_indices:
                mask[:, index] = False

            # apply the mask
            masked_outputs = outputs[mask].view(outputs.size(0), -1)
            masked_targets = targets[mask].view(targets.size(0), -1)

            # compute the KLD loss (per-sample, per-class)
            loss = F.kl_div(F.log_softmax(masked_outputs, dim=-1), masked_targets, reduction='none')

            # apply class weights if provided
            if self.class_weights is not None:
                # Get weights for non-ignored classes
                active_indices = [i for i in range(targets.size(1)) if i not in self.ignore_indices]
                active_weights = self.class_weights[active_indices].to(loss.device)
                # Broadcast weights: (1, num_active_classes) * (batch, num_active_classes)
                loss = loss * active_weights.unsqueeze(0)

            return loss.mean()
        else:
            raise ValueError(f"Unsupported shapes: outputs {outputs.shape}, targets {targets.shape}. "
                             f"Expected both 2D (tile-level) or both 4D (pixel-level).")


def calc_divergence_loss(ignore_indices=None, class_weights=None) -> Callable:
    """
    Initialize the MaskedKLDivergenceLoss with an optional ignore_indices parameter.

    :param ignore_indices: Indices to ignore when calculating the loss.
    :param class_weights: Optional tensor of shape (num_classes,) for class weighting.
    :return: An instance of MaskedKLDivergenceLoss configured with the given parameters.
    """
    return MaskedKLDivergenceLoss(ignore_indices=ignore_indices, class_weights=class_weights)