from dataclasses import dataclass
from typing import Optional
import torch
import pytorch_lightning as pl
from torch import nn, optim
from rtdl_revisiting_models import FTTransformer
from .algorithms import RExAlgorithm, IBIRMAlgorithm

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
    # IB_IRM parameters
    irm_weight: float = 0.0 # > 0 to enable IRM penalty
    ib_weight: float = 0.0 # > 0 to enable Information Bottleneck penalty
    irm_penalty_anneal_iters: int = 100
    ib_penalty_anneal_iters: int = 100

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
        
        self.ib_irm_algorithm = None
        if config.irm_weight > 0.0 or config.ib_weight > 0.0:
            self.ib_irm_algorithm = IBIRMAlgorithm(
                irm_weight=config.irm_weight,
                ib_weight=config.ib_weight,
                irm_penalty_anneal_iters=config.irm_penalty_anneal_iters,
                ib_penalty_anneal_iters=config.ib_penalty_anneal_iters
            )

    def forward(self, X_cont, X_cat, return_features=False):
        if return_features:
            # For IB_IRM, we need both predictions and features
            # This assumes FTTransformer has internal feature extraction
            # If not available, we'll need to modify this based on the actual model structure
            output = self.model(X_cont, X_cat)
            # For now, use the output as features (this may need adjustment based on model internals)
            features = output.detach()
            return output, features
        else:
            return self.model(X_cont, X_cat)
    
    def training_step(self, batch, batch_idx):
        """Standard training step with support for REx and IB_IRM algorithms"""
        X_cont, X_cat, y, e = batch
        
        # Get predictions and features if IB_IRM is enabled
        if self.ib_irm_algorithm is not None:
            predictions, features = self.forward(X_cont, X_cat, return_features=True)
            
            # Use IB_IRM algorithm for loss computation
            loss = self.ib_irm_algorithm.compute_loss(
                predictions, y, features, e, self.loss_fn, self.update_count.item()
            )
            
            # Log individual penalties for monitoring
            irm_penalty, ib_penalty = self.ib_irm_algorithm.compute_penalties(
                predictions, y, features, e, self.loss_fn
            )
            self.log("train/irm_penalty", irm_penalty, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train/ib_penalty", ib_penalty, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
        elif self.rex_algorithm is not None:
            predictions = self.forward(X_cont, X_cat)
            
            # Use REx algorithm for loss computation
            loss = self.rex_algorithm.compute_loss(
                predictions, y, e, self.loss_fn, self.update_count.item()
            )
            penalty = self.rex_algorithm.compute_penalty(predictions, y, e, self.loss_fn)
            self.log("train/rex_penalty", penalty, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            # Standard training without domain adaptation
            predictions = self.forward(X_cont, X_cat)
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
        print(X_cat)
        outputs = self.forward(X_cont, X_cat)
        loss = self.loss_fn(outputs, y)
        return loss
    
    def on_train_batch_start(self, batch, batch_idx):
        # Reset optimizer when REx penalty annealing kicks in
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
        
        # Reset optimizer when IB_IRM penalty annealing kicks in
        if (self.ib_irm_algorithm is not None and 
            self.ib_irm_algorithm.should_reset_optimizer(self.update_count)):
            print("IB_IRM penalty anneal iters reached. Resetting optimizer")
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
    