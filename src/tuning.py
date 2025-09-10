import optuna
import argparse
from src.model import FeatureTokenizerTransformer, FTConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from src.utils import read_data_config
from src.datamodules import BaseDataModule, LOCODataModule

def get_args_parser():
    parser = argparse.ArgumentParser(description="Train FT Transformer model")
    
    parser.add_argument("--data", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--patience", type=int, default=32, help="Early stopping patience")
    parser.add_argument("--dataset", type=str, choices=["erm", "loco"], default="erm", help="Type of dataset to use")
    parser.add_argument("--split", type=str, choices=["random", "kmeans", "custom"], default="random", help="Type of split to use")
    parser.add_argument("--domain_cols", type=list, default=[], help="List of domain columns")

    return parser


def load_datamodule(args, cont_features, cat_features, labels, batch_size):
    if args.split in ["kmeans", "custom"]:
        assert args.dataset == "loco", "LOCO dataset must be used with kmeans or custom split"
        #return LOCODataModule(f"data/{args.data}.csv", dataset_type=args.dataset, num_workers=11, batch_size=batch_size)
    else:
        assert args.split == "random", "Only random split is supported for ERM dataset"
        return BaseDataModule(
            data=f"data/{args.data}.csv", 
            dataset_type=args.dataset, 
            num_workers=11, 
            batch_size=batch_size,
            cont_features=cont_features,
            cat_features=cat_features,
            labels=labels,
            domain_cols=args.domain_cols,
        )


def objective(trial: optuna.trial.Trial) -> float:
    # Get global args (this will be set in main)
    global args, cont_features, cat_features, labels
    
    # Hyperparameters to tune
    n_blocks = trial.suggest_int("n_blocks", 2, 4)  # Reduced range for stability
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)  # Better learning rate range
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)  # Better weight decay range
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])  # Common batch sizes

    # Load data module with the suggested batch_size
    dm = load_datamodule(args, cont_features, cat_features, labels, batch_size)

    # Model configuration
    config = FTConfig(
        n_cont_features=len(cont_features),
        cat_cardinalities=dm.train_ds.get_cardinalities(),
        d_out=len(labels),
        n_blocks=n_blocks,
        lr=lr,
        weight_decay=weight_decay
    )

    # Callbacks and logger
    early_stopping = EarlyStopping(monitor="val/loss", patience=args.patience, mode="min")

    experiment_name = f"{args.data}-hyperparam"
    logger = TensorBoardLogger("runs", name=experiment_name, default_hp_metric=False)

    ckpt_cb = ModelCheckpoint(
        dirpath="artifacts",
        filename=f"{experiment_name}-{logger.name}-v{logger.version}"+"-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1
    )

    # Trainer
    trainer = Trainer(
        max_epochs=50,
        callbacks=[early_stopping, ckpt_cb],
        logger=logger
    )

    # Model
    model = FeatureTokenizerTransformer(config)

    # Training
    trainer.fit(model, dm)

    # Return the best validation loss for optimization
    return trainer.callback_metrics["val/loss"].item()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    
    # Load data configuration - make these global so objective function can access them
    global cont_features, cat_features, labels
    cont_features, cat_features, labels = read_data_config(f"configs/{args.data}.json")
    
    print(f"Dataset: {args.data}")
    print(f"Continuous features: {len(cont_features)}")
    print(f"Categorical features: {len(cat_features)}")
    print(f"Labels: {len(labels)}")
    print("Starting hyperparameter optimization...")

    sampler = optuna.samplers.GPSampler(seed=1)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=5
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="ft_transformer_tuning"
    )

    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")