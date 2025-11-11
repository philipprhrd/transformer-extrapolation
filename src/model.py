from dataclasses import dataclass
from typing import Optional, cast, List
import torch
import pytorch_lightning as pl
from torch import Tensor, nn, optim
from rtdl_revisiting_models import FTTransformer, LinearEmbeddings, CategoricalEmbeddings, MultiheadAttention, _named_sequential, _ReGLU, _INTERNAL_ERROR, _CLSEmbedding
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

class OwnFeatureTokenizerBackbone(nn.Module):
    def __init__(
        self,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        coeffs,
        ffn_activation = 'reglu',
        ffn_d_hidden: Optional[int] = None,
        n_tokens: Optional[int] = None,
        ffn_d_hidden_multiplier: Optional[int] = None
    ):
        super().__init__()

        if ffn_d_hidden is None:
            if ffn_d_hidden_multiplier is None:
                raise ValueError(
                    'If ffn_d_hidden is None,'
                    ' then ffn_d_hidden_multiplier must not be None'
                )
            ffn_d_hidden = int(d_block * cast(float, ffn_d_hidden_multiplier))
        else:
            if ffn_d_hidden_multiplier is not None:
                raise ValueError(
                    'If ffn_d_hidden is not None,'
                    ' then ffn_d_hidden_multiplier must be None'
                )
        
        self.d_block = d_block

        ffn_use_reglu = ffn_activation == 'reglu'
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        # >>> attention
                        'attention': MultiheadAttention(
                            d_embedding=d_block,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            n_tokens=n_tokens
                        ),
                        'attention_residual_dropout': nn.Dropout(residual_dropout),
                        # >>> feed-forward
                        'ffn_normalization': nn.LayerNorm(d_block),
                        'ffn': _named_sequential(
                            (
                                'linear1',
                                # ReGLU divides dimension by 2,
                                # so multiplying by 2 to compensate for this.
                                nn.Linear(
                                    d_block, ffn_d_hidden * (2 if ffn_use_reglu else 1)
                                ),
                            ),
                            ('activation', _ReGLU() if ffn_use_reglu else nn.ReLU()),
                            ('dropout', nn.Dropout(ffn_dropout)),
                            ('linear2', nn.Linear(ffn_d_hidden, d_block)),
                        ),
                        'ffn_residual_dropout': nn.Dropout(residual_dropout),
                        # >>> output (for hook-based introspection)
                        'output': nn.Identity(),
                        # >>> the very first normalization
                        **(
                            {}
                            if layer_idx == 0
                            else {'attention_normalization': nn.LayerNorm(d_block)}
                        ),
                    }
                )
                for layer_idx in range(n_blocks)
            ]
        )
        self.output = (
            None
            if d_out is None
            else _named_sequential(
                ('normalization', nn.LayerNorm(d_block)),
                ('activation', nn.ReLU()),
                ('linear', nn.Linear(d_block, d_out)),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim != 3:
            raise ValueError(
                f'The input must have exactly three dimension, however: {x.ndim=}'
            )

        n_blocks = len(self.blocks)
        for i_block, block in enumerate(self.blocks):
            block = cast(nn.ModuleDict, block)

            x_identity = x
            if 'attention_normalization' in block:
                x = block['attention_normalization'](x)
            x = block['attention'](x[:, :1] if i_block + 1 == n_blocks else x, x)
            x = block['attention_residual_dropout'](x)
            x = x_identity + x

            x_identity = x
            x = block['ffn_normalization'](x)
            x = block['ffn'](x)
            x = block['ffn_residual_dropout'](x)
            x = x_identity + x

            x = block['output'](x)

        x = x[:, 0]  # The representation of [CLS]-token.

        if self.output is not None:
            x = self.output(x)
        return x

class FeatureTokenizerTransformer(pl.LightningModule):
    def __init__(self, config: FTConfig):
        super().__init__()
        self.config = config

        self.loss_fn = nn.MSELoss()

        self.save_hyperparameters()

        self.backbone = OwnFeatureTokenizerBackbone(
            n_tokens=(None),
            n_blocks=config.n_blocks,
            d_out=config.d_out,
            coeffs=config.coeffs,
            d_block = [96, 128, 192, 256, 320, 384][config.n_blocks - 1],
            attention_n_heads=8,
            attention_dropout=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35][config.n_blocks - 1],
            ffn_dropout=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25][config.n_blocks - 1],
            residual_dropout=0.0,
            ffn_d_hidden_multiplier = 4 / 3,
            ffn_activation="snake"
        )

        self.cls_embedding = _CLSEmbedding(self.backbone.d_block)

        self.cont_embeddings = (
            LinearEmbeddings(config.n_cont_features, self.backbone.d_block) if config.n_cont_features > 0 else None
        )
        self.cat_embeddings = (
            CategoricalEmbeddings(config.cat_cardinalities, self.backbone.d_block, True)
            if config.cat_cardinalities
            else None
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

    def forward(self, x_cont, x_cat):
        """Do the forward pass."""
        x_any = x_cat if x_cont is None else x_cont
        if x_any is None:
            raise ValueError('At least one of x_cont and x_cat must be provided.')

        x_embeddings: List[Tensor] = []
        if self.cls_embedding is not None:
            x_embeddings.append(self.cls_embedding(x_any.shape[:-1]))

        for argname, argvalue, module in [
            ('x_cont', x_cont, self.cont_embeddings),
            ('x_cat', x_cat, self.cat_embeddings),
        ]:
            if module is None:
                if argvalue is not None:
                    raise ValueError(
                        FTTransformer._FORWARD_BAD_ARGS_MESSAGE.format(
                            f'{argname} must be None'
                        )
                    )
            else:
                if argvalue is None:
                    raise ValueError(
                        FTTransformer._FORWARD_BAD_ARGS_MESSAGE.format(
                            f'{argname} must not be None'
                        )
                    )
                x_embeddings.append(module(argvalue))
        assert x_embeddings, _INTERNAL_ERROR
        x = torch.cat(x_embeddings, dim=1)
        x = self.backbone(x)
        return x
    
    def forward_with_features(self, X_cont, X_cat):
        """Forward pass that also returns intermediate features for IB_IRM"""
        predictions = self.forward(X_cont, X_cat)
        features = predictions.detach()
        return predictions, features
    
    def training_step(self, batch, batch_idx):
        return self._standard_training_step(batch, batch_idx)
    
    def _standard_training_step(self, batch, batch_idx):
        """Standard training step using the configured algorithm"""
        X_cont, X_cat, y, e = batch
        

        if self.config.ib_weight > 0:
            predictions, features = self.forward_with_features(X_cont, X_cat)
        else:
            predictions = self.forward(X_cont, X_cat)
            features = None
        
        if isinstance(self.algorithm, RExAlgorithm):
            loss = self.algorithm.compute_loss(
                predictions, y, e, self.loss_fn, 
                self.update_count.item()
            )
            
            penalty = self.algorithm.compute_penalty(
                predictions, y, e, self.loss_fn
            )
        else:
            loss = self.algorithm.compute_loss(
                predictions, y, e, self.loss_fn, 
                self.update_count.item(), features=features
            )
            
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
                    self.parameters(), 
                    lr=self.config.lr, 
                    weight_decay=self.config.weight_decay
                )
            ]
    
    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
    