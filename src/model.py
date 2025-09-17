from dataclasses import dataclass
from typing import Optional
import torch
import pytorch_lightning as pl
from torch import nn, optim
from rtdl_revisiting_models import FTTransformer
from .algorithms import EpisodicAlgorithm, RExAlgorithm

@dataclass
class FTConfig():
    n_cont_features: int
    cat_cardinalities: list
    n_blocks: int
    lr: float
    weight_decay: float
    batch_size: int
    d_out: int
    # REx parameters
    rex_weight: float = 0.0 # > 0 to enable REx
    rex_penalty_anneal_iters: int = 100
    # Episodic training parameters
    episodic: bool = False # whether to use episodic training
    n_support: int = 16  # number of support samples per domain per episode
    n_query: int = 16    # number of query samples per domain per episode
    n_domains_per_episode: int = 2  # number of domains to sample per episode
    inner_lr: float = 1e-3  # learning rate for inner optimization
    n_inner_steps: int = 5  # number of inner optimization steps

class FeatureTokenizerTransformer(pl.LightningModule):
    def __init__(self, config: FTConfig):
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
        
        # Initialize algorithms based on config
        self.rex_algorithm = None
        if config.rex_weight > 0.0:
            self.rex_algorithm = RExAlgorithm(
                rex_weight=config.rex_weight,
                rex_penalty_anneal_iters=config.rex_penalty_anneal_iters
            )
        
        self.episodic_algorithm = None
        if config.episodic:
            self.episodic_algorithm = EpisodicAlgorithm(
                n_support=config.n_support,
                n_query=config.n_query,
                n_domains_per_episode=config.n_domains_per_episode,
                inner_lr=config.inner_lr,
                n_inner_steps=config.n_inner_steps
            )

    def forward(self, X_cont, X_cat):
        return self.model(X_cont, X_cat)
    
    def training_step(self, batch, batch_idx):
        if self.episodic_algorithm is not None:
            return self._episodic_training_step(batch, batch_idx)
        else:
            return self._standard_training_step(batch, batch_idx)
    
    def _episodic_training_step(self, batch, batch_idx):
        """Training step using episodic learning"""
        # Collect domain data from current batch
        self.episodic_algorithm.collect_domain_data(batch)
        
        # Sample episode if we have enough data
        if self.episodic_algorithm.has_sufficient_domains():
            episode_data = self.episodic_algorithm.sample_episode(self.device)
            
            if episode_data[0] is not None and episode_data[1] is not None:
                support_set, query_set = episode_data
                
                # For now, we'll do episodic training by using both support and query sets
                # In a full implementation, you'd perform inner loop optimization
                all_X_cont, all_X_cat, all_y, all_e = self.episodic_algorithm.get_episodic_batch(
                    support_set, query_set
                )
                
                predictions = self.forward(all_X_cont, all_X_cat)
                
                if self.rex_algorithm is not None:
                    loss = self.rex_algorithm.compute_loss(
                        predictions, all_y, all_e, self.loss_fn, self.update_count.item()
                    )
                    penalty = self.rex_algorithm.compute_penalty(predictions, all_y, all_e, self.loss_fn)
                    self.log("train/rex_penalty", penalty, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                else:
                    loss = self.loss_fn(predictions, all_y)
                
                self.log("train/episodic_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log("train/n_domains_used", len(torch.unique(all_e)), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            else:
                # Fallback to standard training if episodic sampling fails
                loss = self._standard_training_step(batch, batch_idx)
        else:
            # Fallback to standard training if not enough domains
            loss = self._standard_training_step(batch, batch_idx)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.update_count += 1
        
        return loss
    
    def _standard_training_step(self, batch, batch_idx):
        """Standard training step"""
        X_cont, X_cat, y, e = batch
        predictions = self.forward(X_cont, X_cat)
        
        if self.rex_algorithm is not None:
            loss = self.rex_algorithm.compute_loss(
                predictions, y, e, self.loss_fn, self.update_count.item()
            )
            penalty = self.rex_algorithm.compute_penalty(predictions, y, e, self.loss_fn)
            self.log("train/rex_penalty", penalty, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            loss = self.loss_fn(predictions, y)
        
        if not hasattr(self, '_logged_standard_loss'):
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
        if (self.rex_algorithm is not None and 
            self.update_count == self.rex_algorithm.rex_penalty_anneal_iters):
            print("REx penalty anneal iters reached. Resetting optimizer")
            self.trainer.optimizers = [
                optim.AdamW(
                    self.model.make_parameter_groups(), 
                    lr=self.config.lr, 
                    weight_decay=self.config.weight_decay
                )
            ]
    
    def configure_optimizers(self):
        return optim.AdamW(
            self.model.make_parameter_groups(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
    
    def on_train_epoch_end(self):
        """Clear domain data at the end of each epoch to prevent memory accumulation"""
        if self.episodic_algorithm is not None:
            self.episodic_algorithm.clear_domain_data()
    