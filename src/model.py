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
    d_out: int
    # REx parameter
    rex_weight: float = 0.0 # > 0 to enable REx

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

    def forward(self, X_cont, X_cat):
        return self.model(X_cont, X_cat)
    
    def training_step(self, batch, batch_idx):
        X_cont, X_cat, y, e = batch
        outputs = self.forward(X_cont, X_cat)
        loss = self.loss_fn(outputs, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X_cont, X_cat, y, e = batch
        outputs = self.forward(X_cont, X_cat)
        loss = self.loss_fn(outputs, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        X_cont, X_cat, y, e = batch
        outputs = self.forward(X_cont, X_cat)
        loss = self.loss_fn(outputs, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(
            self.model.make_parameter_groups(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
    