"""
Risk Extrapolation (REx) algorithm for domain generalization.
"""
import torch
from torch import nn
from typing import Optional


class RExAlgorithm:
    """
    Implements Risk Extrapolation (REx) for domain generalization.
    
    REx adds a penalty term that encourages the model to perform 
    consistently across different environments/domains.
    """
    
    def __init__(
        self,
        rex_weight: float = 1.0,
        rex_penalty_anneal_iters: int = 100
    ):
        """
        Args:
            rex_weight: Weight for the REx penalty term
            rex_penalty_anneal_iters: Number of iterations before applying full penalty
        """
        self.rex_weight = rex_weight
        self.rex_penalty_anneal_iters = rex_penalty_anneal_iters
    
    def _compute_per_env_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module
    ) -> torch.Tensor:
        """
        Compute per-environment losses
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            environments: Environment/domain IDs for each sample
            base_loss_fn: Base loss function
            
        Returns:
            Tensor of per-environment losses
        """
        # Compute per-sample losses
        if len(predictions.shape) == 1:
            # Regression case - add dimension for consistent processing
            sample_losses = base_loss_fn(predictions.unsqueeze(-1), targets.unsqueeze(-1)).squeeze(-1)
        else:
            # For other cases, compute loss reduction='none' to get per-sample losses
            if hasattr(base_loss_fn, 'reduction'):
                original_reduction = base_loss_fn.reduction
                base_loss_fn.reduction = 'none'
                sample_losses = base_loss_fn(predictions, targets)
                base_loss_fn.reduction = original_reduction
            else:
                # Fallback for custom loss functions
                sample_losses = torch.sum((predictions - targets) ** 2, dim=-1)
        
        # Get unique environments and their indices
        unique_envs, inverse_indices = torch.unique(environments, return_inverse=True)
        
        # If only one environment, return mean loss
        if len(unique_envs) <= 1:
            if sample_losses.dim() > 1:
                return sample_losses.view(sample_losses.size(0), -1).mean().unsqueeze(0)
            else:
                return sample_losses.mean().unsqueeze(0)
        
        # Compute per-environment losses efficiently
        env_losses = torch.zeros(len(unique_envs), device=predictions.device, dtype=predictions.dtype)
        env_counts = torch.bincount(inverse_indices)
        
        # Flatten sample losses to 1D if needed (to handle multi-dimensional loss outputs)
        if sample_losses.dim() > 1:
            # Take mean over all dimensions except the batch dimension
            sample_losses_flat = sample_losses.view(sample_losses.size(0), -1).mean(dim=1)
        else:
            sample_losses_flat = sample_losses
        
        # Expand sample losses for vectorized computation
        sample_losses_expanded = sample_losses_flat.unsqueeze(0).expand(len(unique_envs), -1)
        env_mask = (inverse_indices.unsqueeze(0) == torch.arange(len(unique_envs), device=predictions.device).unsqueeze(1))
        
        # Compute mean loss per environment
        env_losses = (sample_losses_expanded * env_mask.float()).sum(dim=1) / env_counts.float()
        
        # Remove environments with no samples (if any)
        valid_envs = env_counts > 0
        env_losses = env_losses[valid_envs]
        
        return env_losses

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module,
        update_count: int = 0
    ) -> torch.Tensor:
        """
        Compute REx loss with environment penalty
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            environments: Environment/domain IDs for each sample
            base_loss_fn: Base loss function (e.g., MSE, CrossEntropy)
            update_count: Current training step count for annealing
            
        Returns:
            REx loss (base loss + penalty term)
        """
        # Determine penalty weight based on annealing schedule
        penalty_weight = (
            self.rex_weight 
            if update_count >= self.rex_penalty_anneal_iters 
            else 1.0
        )
        
        # Compute per-environment losses
        env_losses = self._compute_per_env_losses(predictions, targets, environments, base_loss_fn)
        
        mean_loss = env_losses.mean()
        
        # Compute REx penalty
        if len(env_losses) > 1:
            penalty = ((env_losses - mean_loss) ** 2).mean()
            loss = mean_loss + penalty_weight * penalty
        else:
            loss = mean_loss
        
        return loss
    
    def compute_penalty(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module
    ) -> torch.Tensor:
        """
        Compute just the REx penalty term (for logging purposes)
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            environments: Environment/domain IDs for each sample
            base_loss_fn: Base loss function
            
        Returns:
            REx penalty term
        """
        # Compute per-environment losses
        env_losses = self._compute_per_env_losses(predictions, targets, environments, base_loss_fn)
        
        if len(env_losses) > 1:
            mean_loss = env_losses.mean()
            penalty = ((env_losses - mean_loss) ** 2).mean()
            return penalty
        else:
            return torch.tensor(0.0, device=predictions.device)
    
    def is_enabled(self) -> bool:
        """Check if REx is enabled (rex_weight > 0)"""
        return self.rex_weight > 0.0