from dataclasses import dataclass
import torch
import pytorch_lightning as pl
from torch import nn, optim
from rtdl_revisiting_models import FTTransformer

@dataclass
class FTConfig():
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
    episodic: bool = False # whether to use episodic training (only for LOCO)

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

    def forward(self, X_cont, X_cat):
        return self.model(X_cont, X_cat)
    
    def training_step(self, batch, batch_idx):
        X_cont, X_cat, y, e = batch

        if self.config.rex_weight > 0.0:
            penalty_weight = (self.config.rex_weight 
                         if self.update_count >= self.config.rex_penalty_anneal_iters 
                         else 1.0)

            predictions = self.forward(X_cont, X_cat)
            batch_loss = self.loss_fn(predictions, y)

            #unique_envs, env_counts = torch.unique(e, return_counts=True)
            #losses = torch.zeros(len(unique_envs), device=self.device)

            unique_envs, inverse_indices = torch.unique(e, return_inverse=True)

            #for i, env_id in enumerate(unique_envs):
            #    env_mask = (e == env_id)

                #X_cont_env = X_cont[env_mask]
                #X_cat_env = X_cat[env_mask]
                #y_env = y[env_mask]

                #if X_cont_env.shape[0] == 0:
                #    continue

                #predictions = self.forward(X_cont_env, X_cat_env)
                #env_loss = self.loss_fn(predictions, y_env)
                #losses[i] = env_loss

            #mean_loss = losses.mean()
            #penalty = torch.tensor(0.0, device=self.device)
            #if len(losses) > 1:
            #    penalty = ((losses - mean_loss) ** 2).mean()

            #loss = mean_loss + penalty_weight * penalty

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

                self.log("train/rex_penalty", penalty, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            outputs = self.forward(X_cont, X_cat)
            loss = self.loss_fn(outputs, y)
        
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
    