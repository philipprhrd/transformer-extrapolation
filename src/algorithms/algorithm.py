from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import Optional, Union, Tuple


class BaseAlgorithm(ABC):
    """Base class for domain generalization algorithms."""
    
    def __init__(self, **kwargs):
        """Initialize the algorithm with hyperparameters."""
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module,
        update_count: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """Compute the algorithm-specific loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            environments: Environment/domain IDs for each sample
            base_loss_fn: Base loss function (e.g., MSE, CrossEntropy)
            update_count: Current training step count for annealing
            **kwargs: Additional algorithm-specific arguments
            
        Returns:
            Algorithm-specific loss
        """
        pass
    
    @abstractmethod
    def compute_penalty(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Compute the penalty term(s) for logging purposes.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            environments: Environment/domain IDs for each sample
            base_loss_fn: Base loss function
            **kwargs: Additional algorithm-specific arguments
            
        Returns:
            Penalty term(s)
        """
        pass
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if the algorithm is enabled."""
        pass
    
    def should_reset_optimizer(self, update_count: int) -> bool:
        """Check if optimizer should be reset due to penalty annealing.
        
        Args:
            update_count: Current training step count
            
        Returns:
            True if optimizer should be reset
        """
        return False
    
    def _compute_per_env_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module
    ) -> torch.Tensor:
        """Helper method to compute per-environment losses."""
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

        if sample_losses.dim() > 1:
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


class ErmAlgorithm(BaseAlgorithm):
    """Empirical Risk Minimization (ERM) - vanilla training without domain regularization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module,
        update_count: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """Compute standard ERM loss."""
        return base_loss_fn(predictions, targets)
    
    def compute_penalty(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module,
        **kwargs
    ) -> torch.Tensor:
        """ERM has no penalty term."""
        return torch.tensor(0.0, device=predictions.device)
    
    def is_enabled(self) -> bool:
        """ERM is always enabled."""
        return True