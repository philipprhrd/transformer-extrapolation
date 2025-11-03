import optuna
import argparse
from src.model import FeatureTokenizerTransformer, FTConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from src.utils import read_data_config
from src.train import load_datamodule

def get_args_parser():
    """Sample usage:

    "Interpolation"-tuning with 1 cluster as extrapolation test:
    
    ``python -m src.tuning <...> --mode normal --cluster_eval <1..n_cluster>``

    "LOCO"-tuning with 1 cluster as extrapolation test:

    ``python -m src.tuning <...> --mode loco --cluster_eval <1..n_cluster>``

    Completely custom logic (you have to define all splits in a python file and pass the name here):

    ``python -m src.tuning <...> --custom <python_file_name (in src/custom)>``
    """

    parser = argparse.ArgumentParser(description="Train FT Transformer model")
    
    parser.add_argument("--data", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--n_blocks", type=int, default=None, help="Number of transformer blocks (if None, will be tuned)")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--mode", type=str, choices=["normal", "loco"], default="normal", help="Training mode")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for LOCO mode")
    parser.add_argument("--cluster_eval", type=int, default=None)

    # REx
    parser.add_argument("--rex", action="store_true", help="Use REx for training")

    # IRM
    parser.add_argument("--irm", action="store_true", help="Use IRM for training")

    # IB IRM
    parser.add_argument("--ib", action="store_true", help="Use IB for training")

    parser.add_argument("--custom", type=str, default=None, help="Use custom split from file")

    parser.add_argument("--sampler", type=str, choices=["TPESampler", "GPSampler"], default="TPESampler", help="Optuna sampler")

    return parser

def objective(trial: optuna.trial.Trial) -> float:
    global args, cont_features, cat_features, labels
    
    # Hyperparameters to tune
    n_blocks = trial.suggest_int("n_blocks", 2, 6) if args.n_blocks is None else args.n_blocks  # Reduced range for stability
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)  # Better learning rate range
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)  # Better weight decay range
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])  # Common batch sizes
    rex_weight = trial.suggest_float("rex_weight", 0.1, 100, log=True) if args.rex else 0.0 # REx weight
    irm_weight = trial.suggest_float("irm_weight", 1, 10000, log=True) if args.irm else 0.0 # IRM weight
    ib_irm_weight = trial.suggest_float("ib_irm_weight", 0.01, 10, log=True) if args.ib else 0.0 # IB-IRM weight

    # Load data module with the suggested batch_size
    dm = load_datamodule(args, batch_size, cont_features, cat_features, labels)
    dm.init()

    # Model configuration
    config = FTConfig(
        n_cont_features=len(cont_features),
        cat_cardinalities=dm.train_ds.get_cardinalities(),
        d_out=len(labels),
        n_blocks=n_blocks,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        rex_weight=rex_weight,
        rex_penalty_anneal_iters=280,
        irm_weight=irm_weight,
        ib_weight=ib_irm_weight,
        irm_penalty_anneal_iters=280,
        ib_penalty_anneal_iters=280
    )

    early_stopping = EarlyStopping(monitor="val/loss", patience=args.patience, mode="min")

    logger = TensorBoardLogger("runs", name=trial.study.study_name, default_hp_metric=False)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"artifacts/{trial.study.study_name}",
        filename=f"{trial.study.study_name}-best",
        monitor="val/loss",
        mode="min",
        save_top_k=1
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
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
    
    global cont_features, cat_features, labels
    cont_features, cat_features, labels = read_data_config(f"configs/{args.data}.json")
    
    print("Starting hyperparameter optimization...")

    if args.sampler == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=1)
    else:
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
        study_name=f"{args.data}-{args.mode}{'-rex' if args.rex else ''}{'-irm' if args.irm else ''}{'-ib' if args.ib else ''}-hyperparam"
    )

    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")