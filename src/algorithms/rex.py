import torch
from torch import nn
from typing import Optional
from .algorithm import BaseAlgorithm

class RExAlgorithm(BaseAlgorithm):    
    def __init__(
        self,
        rex_weight: float = 1.0,
        rex_penalty_anneal_iters: int = 100
    ):
        self.rex_weight = rex_weight
        self.rex_penalty_anneal_iters = rex_penalty_anneal_iters

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