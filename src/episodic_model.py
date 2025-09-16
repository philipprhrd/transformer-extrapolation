from dataclasses import dataclass
import torch
import pytorch_lightning as pl
from torch import nn, optim
from rtdl_revisiting_models import FTTransformer
import random
import numpy as np

@dataclass
class EpisodicFTConfig():
    n_cont_features: int
    cat_cardinalities: list
    n_blocks: int
    lr: float
    weight_decay: float
    batch_size: int
    d_out: int
    # REx parameter
    rex_weight: float = 0.0 # > 0 to enable REx
    rex_penalty_anneal_iters: int = 100
    # Episodic training parameters
    episodic: bool = True
    n_support: int = 16  # number of support samples per domain per episode
    n_query: int = 16    # number of query samples per domain per episode
    n_domains_per_episode: int = 2  # number of domains to sample per episode
    inner_lr: float = 1e-3  # learning rate for inner optimization
    n_inner_steps: int = 5  # number of inner optimization steps

class EpisodicFeatureTokenizerTransformer(pl.LightningModule):
    def __init__(self, config: EpisodicFTConfig):
        super().__init__()
        self.config = config

        self.loss_fn = nn.MSELoss()

        self.save_hyperparameters()

        default_kwargs = FTTransformer.get_default_kwargs(n_blocks=config.n_blocks)
        
        self.model = FTTransformer(
            n_cont_features=config.n_cont_features,
            cat_cardinalities=config.cat_cardinalities,
            d_out=config.d_out,
            **default_kwargs
        )

        self.register_buffer("update_count", torch.tensor(0))
        
        # Store domains from training data for episodic sampling
        self.domain_data = {}
        self.available_domains = []

    def forward(self, X_cont, X_cat):
        return self.model(X_cont, X_cat)
    
    def _collect_domain_data(self, batch):
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
    
    def _sample_episode(self):
        """Sample support and query sets for episodic training"""
        if len(self.available_domains) < self.config.n_domains_per_episode:
            # If not enough domains, use all available
            selected_domains = self.available_domains
        else:
            selected_domains = random.sample(self.available_domains, self.config.n_domains_per_episode)
        
        support_X_cont, support_X_cat, support_y, support_e = [], [], [], []
        query_X_cont, query_X_cat, query_y, query_e = [], [], [], []
        
        for domain_id in selected_domains:
            domain_data = self.domain_data[domain_id]
            n_samples = len(domain_data['X_cont'])
            
            if n_samples < self.config.n_support + self.config.n_query:
                # If not enough samples, use all for support and query
                n_support = min(self.config.n_support, n_samples // 2)
                n_query = n_samples - n_support
            else:
                n_support = self.config.n_support
                n_query = self.config.n_query
            
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
            support_X_cont = torch.stack(support_X_cont).to(self.device)
            support_X_cat = torch.stack(support_X_cat).to(self.device)
            support_y = torch.stack(support_y).to(self.device)
            support_e = torch.stack(support_e).to(self.device)
        else:
            return None, None
            
        if query_X_cont:
            query_X_cont = torch.stack(query_X_cont).to(self.device)
            query_X_cat = torch.stack(query_X_cat).to(self.device)
            query_y = torch.stack(query_y).to(self.device)
            query_e = torch.stack(query_e).to(self.device)
        else:
            return None, None
        
        return (support_X_cont, support_X_cat, support_y, support_e), \
               (query_X_cont, query_X_cat, query_y, query_e)
    
    def _inner_loop(self, support_set):
        """Perform inner loop optimization on support set"""
        support_X_cont, support_X_cat, support_y, support_e = support_set
        
        # Create a copy of model parameters for inner loop
        fast_weights = {}
        for name, param in self.model.named_parameters():
            fast_weights[name] = param.clone()
        
        for step in range(self.config.n_inner_steps):
            # Forward pass with current fast weights
            predictions = self._forward_with_weights(support_X_cont, support_X_cat, fast_weights)
            
            # Compute loss
            if self.config.rex_weight > 0.0:
                loss = self._compute_rex_loss(predictions, support_y, support_e)
            else:
                loss = self.loss_fn(predictions, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), 
                                      create_graph=True, retain_graph=True)
            
            # Update fast weights
            for (name, param), grad in zip(fast_weights.items(), grads):
                fast_weights[name] = param - self.config.inner_lr * grad
        
        return fast_weights
    
    def _forward_with_weights(self, X_cont, X_cat, weights):
        """Forward pass using specific weights"""
        # This is a simplified version - in practice, you'd need to implement
        # a way to use custom weights with the FTTransformer
        # For now, we'll use the regular forward pass
        return self.forward(X_cont, X_cat)
    
    def _compute_rex_loss(self, predictions, y, e):
        """Compute REx loss with environment penalty"""
        penalty_weight = (self.config.rex_weight 
                         if self.update_count >= self.config.rex_penalty_anneal_iters 
                         else 1.0)

        batch_loss = self.loss_fn(predictions, y)
        
        unique_envs, inverse_indices = torch.unique(e, return_inverse=True)
        
        if len(unique_envs) <= 1:
            return batch_loss
        
        env_losses = torch.zeros(len(unique_envs), device=self.device, dtype=predictions.dtype)
        env_counts = torch.bincount(inverse_indices)

        batch_losses_expanded = batch_loss.unsqueeze(0).expand(len(unique_envs), -1)
        env_mask = (inverse_indices.unsqueeze(0) == torch.arange(len(unique_envs), device=self.device).unsqueeze(1))

        env_losses = (batch_losses_expanded * env_mask.float()).sum(dim=1) / env_counts.float()
    
        # Remove environments with no samples (if any)
        valid_envs = env_counts > 0
        env_losses = env_losses[valid_envs]
        
        mean_loss = env_losses.mean()
        
        # Compute penalty efficiently
        if len(env_losses) > 1:
            penalty = ((env_losses - mean_loss) ** 2).mean()
            loss = mean_loss + penalty_weight * penalty
            self.log("train/rex_penalty", penalty, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            loss = mean_loss
            self.log("train/rex_penalty", torch.tensor(0.0), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        # Collect domain data from current batch
        self._collect_domain_data(batch)
        
        # Sample episode if we have enough data
        if len(self.available_domains) >= 1:
            episode_data = self._sample_episode()
            
            if episode_data[0] is not None and episode_data[1] is not None:
                support_set, query_set = episode_data
                
                # Perform inner loop on support set (simplified for now)
                # In full implementation, you'd update model weights and then evaluate on query set
                
                # For now, we'll do episodic training by using both support and query sets
                all_X_cont = torch.cat([support_set[0], query_set[0]], dim=0)
                all_X_cat = torch.cat([support_set[1], query_set[1]], dim=0)
                all_y = torch.cat([support_set[2], query_set[2]], dim=0)
                all_e = torch.cat([support_set[3], query_set[3]], dim=0)
                
                predictions = self.forward(all_X_cont, all_X_cat)
                
                if self.config.rex_weight > 0.0:
                    loss = self._compute_rex_loss(predictions, all_y, all_e)
                else:
                    loss = self.loss_fn(predictions, all_y)
                
                self.log("train/episodic_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log("train/n_domains_used", len(torch.unique(all_e)), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            else:
                # Fallback to regular training if episodic sampling fails
                X_cont, X_cat, y, e = batch
                predictions = self.forward(X_cont, X_cat)
                
                if self.config.rex_weight > 0.0:
                    loss = self._compute_rex_loss(predictions, y, e)
                else:
                    loss = self.loss_fn(predictions, y)
        else:
            # Fallback to regular training if not enough domains
            X_cont, X_cat, y, e = batch
            predictions = self.forward(X_cont, X_cat)
            
            if self.config.rex_weight > 0.0:
                loss = self._compute_rex_loss(predictions, y, e)
            else:
                loss = self.loss_fn(predictions, y)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.update_count += 1
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def _common_step(self, batch, batch_idx):
        X_cont, X_cat, y, e = batch
        outputs = self.forward(X_cont, X_cat)
        loss = self.loss_fn(outputs, y)
        return loss
    
    def on_train_batch_start(self, batch, batch_idx):
        if self.update_count == self.config.rex_penalty_anneal_iters:
            print("REx penalty anneal iters reached. Resetting optimizer")
            self.trainer.optimizers = [
                optim.AdamW(
                    self.model.make_parameter_groups(), lr=self.config.lr, weight_decay=self.config.weight_decay
                )
            ]
    
    def configure_optimizers(self):
        return optim.AdamW(
            self.model.make_parameter_groups(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
    
    def on_train_epoch_end(self):
        """Clear domain data at the end of each epoch to prevent memory accumulation"""
        self.domain_data.clear()
        self.available_domains.clear()
