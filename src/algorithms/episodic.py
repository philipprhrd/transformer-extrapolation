"""
Episodic learning algorithm for domain generalization.
"""
import random
import torch
from torch import nn
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .rex import RExAlgorithm


class EpisodicAlgorithm:
    """
    Implements episodic learning for domain generalization.
    
    This algorithm samples episodes consisting of support and query sets
    from different domains and performs meta-learning style training.
    """
    
    def __init__(
        self,
        n_support: int = 16,
        n_query: int = 16,
        n_domains_per_episode: int = 2,
        inner_lr: float = 1e-3,
        n_inner_steps: int = 5
    ):
        """
        Args:
            n_support: Number of support samples per domain per episode
            n_query: Number of query samples per domain per episode
            n_domains_per_episode: Number of domains to sample per episode
            inner_lr: Learning rate for inner optimization
            n_inner_steps: Number of inner optimization steps
        """
        self.n_support = n_support
        self.n_query = n_query
        self.n_domains_per_episode = n_domains_per_episode
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps
        
        # Store domains from training data for episodic sampling
        self.domain_data = {}
        self.available_domains = []
    
    def collect_domain_data(self, batch: Tuple[torch.Tensor, ...]) -> None:
        """Collect and organize data by domain for episodic training"""
        X_cont, X_cat, y, e = batch
        
        for i in range(len(e)):
            domain_id = e[i].item()
            if domain_id not in self.domain_data:
                self.domain_data[domain_id] = {
                    'X_cont': [],
                    'X_cat': [],
                    'y': []
                }
            
            self.domain_data[domain_id]['X_cont'].append(X_cont[i])
            self.domain_data[domain_id]['X_cat'].append(X_cat[i])
            self.domain_data[domain_id]['y'].append(y[i])
        
        self.available_domains = list(self.domain_data.keys())
    
    def sample_episode(self, device: torch.device) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """
        Sample support and query sets for episodic training
        
        Returns:
            Tuple of (support_set, query_set) where each set is a tuple of 
            (X_cont, X_cat, y, e) tensors, or (None, None) if sampling fails
        """
        if len(self.available_domains) < self.n_domains_per_episode:
            # If not enough domains, use all available
            selected_domains = self.available_domains
        else:
            selected_domains = random.sample(self.available_domains, self.n_domains_per_episode)
        
        support_X_cont, support_X_cat, support_y, support_e = [], [], [], []
        query_X_cont, query_X_cat, query_y, query_e = [], [], [], []
        
        for domain_id in selected_domains:
            domain_data = self.domain_data[domain_id]
            n_samples = len(domain_data['X_cont'])
            
            if n_samples < self.n_support + self.n_query:
                # If not enough samples, use all for support and query
                n_support = min(self.n_support, n_samples // 2)
                n_query = n_samples - n_support
            else:
                n_support = self.n_support
                n_query = self.n_query
            
            # Sample indices
            indices = random.sample(range(n_samples), min(n_support + n_query, n_samples))
            support_indices = indices[:n_support]
            query_indices = indices[n_support:n_support + n_query]
            
            # Collect support set
            for idx in support_indices:
                support_X_cont.append(domain_data['X_cont'][idx])
                support_X_cat.append(domain_data['X_cat'][idx])
                support_y.append(domain_data['y'][idx])
                support_e.append(torch.tensor(domain_id, dtype=torch.long))
            
            # Collect query set
            for idx in query_indices:
                query_X_cont.append(domain_data['X_cont'][idx])
                query_X_cat.append(domain_data['X_cat'][idx])
                query_y.append(domain_data['y'][idx])
                query_e.append(torch.tensor(domain_id, dtype=torch.long))
        
        # Convert to tensors and move to device
        if support_X_cont:
            support_X_cont = torch.stack(support_X_cont).to(device)
            support_X_cat = torch.stack(support_X_cat).to(device)
            support_y = torch.stack(support_y).to(device)
            support_e = torch.stack(support_e).to(device)
        else:
            return None, None
            
        if query_X_cont:
            query_X_cont = torch.stack(query_X_cont).to(device)
            query_X_cat = torch.stack(query_X_cat).to(device)
            query_y = torch.stack(query_y).to(device)
            query_e = torch.stack(query_e).to(device)
        else:
            return None, None
        
        return (support_X_cont, support_X_cat, support_y, support_e), \
               (query_X_cont, query_X_cat, query_y, query_e)
    
    def inner_loop(
        self, 
        support_set: Tuple[torch.Tensor, ...], 
        model: nn.Module,
        loss_fn: nn.Module,
        rex_algorithm: Optional['RExAlgorithm'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform inner loop optimization on support set
        
        Args:
            support_set: Tuple of (X_cont, X_cat, y, e) tensors
            model: The model to optimize
            loss_fn: Loss function to use
            rex_algorithm: Optional REx algorithm for regularization
            
        Returns:
            Dictionary of fast weights after inner optimization
        """
        support_X_cont, support_X_cat, support_y, support_e = support_set
        
        # Create a copy of model parameters for inner loop
        fast_weights = {}
        for name, param in model.named_parameters():
            fast_weights[name] = param.clone()
        
        for step in range(self.n_inner_steps):
            # Forward pass with current fast weights
            predictions = self._forward_with_weights(support_X_cont, support_X_cat, fast_weights, model)
            
            # Compute loss
            if rex_algorithm is not None:
                loss = rex_algorithm.compute_loss(predictions, support_y, support_e, loss_fn)
            else:
                loss = loss_fn(predictions, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), 
                                      create_graph=True, retain_graph=True)
            
            # Update fast weights
            for (name, param), grad in zip(fast_weights.items(), grads):
                fast_weights[name] = param - self.inner_lr * grad
        
        return fast_weights
    
    def _forward_with_weights(
        self, 
        X_cont: torch.Tensor, 
        X_cat: torch.Tensor, 
        weights: Dict[str, torch.Tensor],
        model: nn.Module
    ) -> torch.Tensor:
        """
        Forward pass using specific weights
        
        Note: This is a simplified version. In practice, you'd need to implement
        a way to use custom weights with the FTTransformer properly.
        For now, we'll use the regular forward pass.
        """
        return model(X_cont, X_cat)
    
    def clear_domain_data(self) -> None:
        """Clear domain data to prevent memory accumulation"""
        self.domain_data.clear()
        self.available_domains.clear()
    
    def has_sufficient_domains(self) -> bool:
        """Check if we have enough domains for episodic training"""
        return len(self.available_domains) >= 1
    
    def get_episodic_batch(
        self, 
        support_set: Tuple[torch.Tensor, ...], 
        query_set: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Combine support and query sets into a single batch for training
        
        Returns:
            Combined batch as (X_cont, X_cat, y, e) tensors
        """
        all_X_cont = torch.cat([support_set[0], query_set[0]], dim=0)
        all_X_cat = torch.cat([support_set[1], query_set[1]], dim=0)
        all_y = torch.cat([support_set[2], query_set[2]], dim=0)
        all_e = torch.cat([support_set[3], query_set[3]], dim=0)
        
        return all_X_cont, all_X_cat, all_y, all_e