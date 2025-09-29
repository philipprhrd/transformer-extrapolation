from dataclasses import dataclass
import torch
import pytorch_lightning as pl
from torch import nn, optim
from rtdl_revisiting_models import FTTransformer
from .algorithms import BaseAlgorithm, ErmAlgorithm, RExAlgorithm, IBIRMAlgorithm

@dataclass
class FTConfig():
    n_cont_features: int
    cat_cardinalities: list
    n_blocks: int
    lr: float
    weight_decay: float
    batch_size: int
    d_out: int
    
    # REX hyperparameters
    rex_weight: float = 0.0
    rex_penalty_anneal_iters: int = 100
    
    # IB_IRM hyperparameters
    irm_weight: float = 0.0
    ib_weight: float = 0.0
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
        
        # Initialize algorithm based on config
        self.algorithm = self._create_algorithm()
    
    def _create_algorithm(self) -> BaseAlgorithm:
        """Create the appropriate algorithm based on configuration."""
        if self.config.rex_weight > 0:
            return RExAlgorithm(
                rex_weight=self.config.rex_weight,
                rex_penalty_anneal_iters=self.config.rex_penalty_anneal_iters
            )
        elif self.config.ib_weight > 0:
            return IBIRMAlgorithm(
                irm_weight=self.config.irm_weight,
                ib_weight=self.config.ib_weight,
                irm_penalty_anneal_iters=self.config.irm_penalty_anneal_iters,
                ib_penalty_anneal_iters=self.config.ib_penalty_anneal_iters
            )
        else:  # Default to ERM
            return ErmAlgorithm()

    def forward(self, X_cont, X_cat):
        return self.model(X_cont, X_cat)
    
    def forward_with_features(self, X_cont, X_cat):
        """Forward pass that also returns intermediate features for IB_IRM"""
        # This is a simplified feature extraction
        # In practice, you might want to access intermediate layers of the transformer
        predictions = self.model(X_cont, X_cat)
        # For now, use the predictions as features (this should be improved)
        # In a real implementation, you'd extract features from before the final layer
        features = predictions.detach()
        return predictions, features
    
    def training_step(self, batch, batch_idx):
        return self._standard_training_step(batch, batch_idx)
    
    def _standard_training_step(self, batch, batch_idx):
        """Standard training step using the configured algorithm"""
        X_cont, X_cat, y, e = batch
        
        # Get features for algorithms that need them (like IB_IRM)
        if self.config.ib_weight > 0:
            predictions, features = self.forward_with_features(X_cont, X_cat)
        else:
            predictions = self.forward(X_cont, X_cat)
            features = None
        
        # Compute loss using the algorithm
        loss = self.algorithm.compute_loss(
            predictions, y, e, self.loss_fn, 
            self.update_count.item(), features=features
        )
        
        # Compute and log penalty terms
        penalty = self.algorithm.compute_penalty(
            predictions, y, e, self.loss_fn, features=features
        )
        
        # Log penalty terms based on algorithm type
        if self.config.rex_weight > 0:
            self.log("train/rex_penalty", penalty, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        elif self.config.ib_weight > 0:
            irm_penalty, ib_penalty = penalty
            self.log("train/irm_penalty", irm_penalty, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train/ib_penalty", ib_penalty, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
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
        # Check if optimizer should be reset based on algorithm
        if self.algorithm.should_reset_optimizer(self.update_count.item()):
            print(f"Algorithm penalty anneal iters reached. Resetting optimizer")
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
    