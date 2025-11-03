from dataclasses import dataclass
from typing import Optional, cast
import torch
import pytorch_lightning as pl
from torch import Tensor, nn, optim
from rtdl_revisiting_models import FTTransformer, LinearEmbeddings, CategoricalEmbeddings, MultiheadAttention, _named_sequential, _ReGLU
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

class PolynomActivation(nn.Module):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = nn.Parameter(torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for power, coeff in enumerate(self.coeffs):
            result += coeff * (x ** power)
        return result

class OwnFeatureTokenizerBackbone(nn.Module):
    def __init__(
        self,
        *,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: Optional[int] = None,
        ffn_d_hidden_multiplier: Optional[float],
        ffn_dropout: float,
        n_tokens: Optional[int] = None,
        ffn_activation = 'ReGLU',
        residual_dropout: float,
        coeffs
    ):
        super().__init__()
        ffn_use_reglu = ffn_activation == 'ReGLU'
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
                            ('activation', _ReGLU() if ffn_use_reglu else PolynomActivation(coeffs)),
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

        self.cont_embeddings = (
            LinearEmbeddings(config.n_cont_features, config.n_blocks) if config.n_cont_features > 0 else None
        )
        self.cat_embeddings = (
            CategoricalEmbeddings(config.cat_cardinalities, config.n_blocks, True)
            if config.cat_cardinalities
            else None
        )

        self.backbone = OwnFeatureTokenizerBackbone(
            n_tokens=(
                None
            ),
        )

        #default_kwargs = FTTransformer.get_default_kwargs(n_blocks=config.n_blocks)

        #self.model = FTTransformer(
        #    n_cont_features=config.n_cont_features,
        #    cat_cardinalities=config.cat_cardinalities,
        #    d_out=config.d_out,
        #    **default_kwargs
        #)

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
        if isinstance(self.algorithm, RExAlgorithm):
            loss = self.algorithm.compute_loss(
                predictions, y, e, self.loss_fn, 
                self.update_count.item()
            )
            
            # Compute and log penalty terms
            penalty = self.algorithm.compute_penalty(
                predictions, y, e, self.loss_fn
            )
        else:
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
    