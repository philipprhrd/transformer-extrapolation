"""
Information Bottleneck based IRM (IB_IRM) algorithm for domain generalization.
"""
import torch
from torch import nn, autograd
import torch.nn.functional as F
from typing import Optional


class IBIRMAlgorithm:
    """
    Implements Information Bottleneck based IRM for domain generalization.
    
    IB_IRM combines Invariant Risk Minimization (IRM) with Information Bottleneck
    regularization on features to encourage domain-invariant representations.
    """
    
    def __init__(
        self,
        irm_weight: float = 1.0,
        ib_weight: float = 1.0,
        irm_penalty_anneal_iters: int = 100,
        ib_penalty_anneal_iters: int = 100
    ):
        """
        Args:
            irm_weight: Weight for the IRM penalty term
            ib_weight: Weight for the Information Bottleneck penalty term
            irm_penalty_anneal_iters: Number of iterations before applying full IRM penalty
            ib_penalty_anneal_iters: Number of iterations before applying full IB penalty
        """
        self.irm_weight = irm_weight
        self.ib_weight = ib_weight
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
        self.ib_penalty_anneal_iters = ib_penalty_anneal_iters
    
    @staticmethod
    def _irm_penalty(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute IRM penalty term.
        
        Args:
            logits: Model predictions
            targets: Ground truth targets
            
        Returns:
            IRM penalty term
        """
        device = "cuda" if logits.is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, targets[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, targets[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result
    
    def _compute_per_env_losses_and_features(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-environment losses, IRM penalties, and IB penalties
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth targets
            features: Feature representations from the model
            environments: Environment/domain IDs for each sample
            base_loss_fn: Base loss function
            
        Returns:
            Tuple of (per-env losses, per-env IRM penalties, per-env IB penalties)
        """
        # Get unique environments and their indices
        unique_envs, inverse_indices = torch.unique(environments, return_inverse=True)
        
        env_losses = torch.zeros(len(unique_envs), device=predictions.device, dtype=predictions.dtype)
        env_irm_penalties = torch.zeros(len(unique_envs), device=predictions.device, dtype=predictions.dtype)
        env_ib_penalties = torch.zeros(len(unique_envs), device=predictions.device, dtype=predictions.dtype)
        
        for i, env_id in enumerate(unique_envs):
            # Get samples for this environment
            env_mask = (inverse_indices == i)
            env_predictions = predictions[env_mask]
            env_targets = targets[env_mask]
            env_features = features[env_mask]
            
            # Compute NLL loss for this environment
            env_losses[i] = base_loss_fn(env_predictions, env_targets)
            
            # Compute IRM penalty for this environment
            if len(env_predictions) >= 2:  # Need at least 2 samples for IRM penalty
                env_irm_penalties[i] = self._irm_penalty(env_predictions, env_targets)
            else:
                env_irm_penalties[i] = torch.tensor(0.0, device=predictions.device)
            
            # Compute IB penalty (variance across features) for this environment
            if len(env_features) > 1:
                env_ib_penalties[i] = env_features.var(dim=0).mean()
            else:
                env_ib_penalties[i] = torch.tensor(0.0, device=predictions.device)
        
        return env_losses, env_irm_penalties, env_ib_penalties

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module,
        update_count: int = 0
    ) -> torch.Tensor:
        """
        Compute IB_IRM loss with IRM and Information Bottleneck penalties
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth targets
            features: Feature representations from the model
            environments: Environment/domain IDs for each sample
            base_loss_fn: Base loss function (e.g., CrossEntropy)
            update_count: Current training step count for annealing
            
        Returns:
            IB_IRM loss (base loss + IRM penalty + IB penalty)
        """
        # Determine penalty weights based on annealing schedule
        irm_penalty_weight = (
            self.irm_weight 
            if update_count >= self.irm_penalty_anneal_iters 
            else 1.0
        )
        
        ib_penalty_weight = (
            self.ib_weight 
            if update_count >= self.ib_penalty_anneal_iters 
            else 0.0
        )
        
        # Compute per-environment losses and penalties
        env_losses, env_irm_penalties, env_ib_penalties = self._compute_per_env_losses_and_features(
            predictions, targets, features, environments, base_loss_fn
        )
        
        # Compute mean values
        mean_loss = env_losses.mean()
        mean_irm_penalty = env_irm_penalties.mean()
        mean_ib_penalty = env_ib_penalties.mean()
        
        # Compile total loss
        loss = mean_loss
        loss += irm_penalty_weight * mean_irm_penalty
        loss += ib_penalty_weight * mean_ib_penalty
        
        return loss
    
    def compute_penalties(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        environments: torch.Tensor,
        base_loss_fn: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute just the IRM and IB penalty terms (for logging purposes)
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth targets
            features: Feature representations from the model
            environments: Environment/domain IDs for each sample
            base_loss_fn: Base loss function
            
        Returns:
            Tuple of (IRM penalty, IB penalty)
        """
        # Compute per-environment penalties
        _, env_irm_penalties, env_ib_penalties = self._compute_per_env_losses_and_features(
            predictions, targets, features, environments, base_loss_fn
        )
        
        irm_penalty = env_irm_penalties.mean()
        ib_penalty = env_ib_penalties.mean()
        
        return irm_penalty, ib_penalty
    
    def is_enabled(self) -> bool:
        """Check if IB_IRM is enabled (either weight > 0)"""
        return self.irm_weight > 0.0 or self.ib_weight > 0.0
    
    def should_reset_optimizer(self, update_count: int) -> bool:
        """
        Check if optimizer should be reset due to penalty annealing.
        
        Args:
            update_count: Current training step count
            
        Returns:
            True if optimizer should be reset
        """
        return (update_count == self.irm_penalty_anneal_iters or 
                update_count == self.ib_penalty_anneal_iters)
