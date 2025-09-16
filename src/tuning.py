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
    #parser.add_argument("--n_blocks", type=int, default=4, help="Number of FT Transformer blocks")
    #parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    #parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    #parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=32, help="Early stopping patience")
    #parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

    parser.add_argument("--mode", type=str, choices=["normal", "loco", "episodic"], default="normal", help="Training mode")
    #parser.add_argument("--rex_weight", type=float, default=0.0, help="Use REx for training")
    parser.add_argument("--rex_anneal_iters", type=int, default=280, help="Number of iterations to anneal REx weight")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for LOCO mode")
    parser.add_argument("--cluster_eval", type=int, default=None)

    parser.add_argument("--custom", type=str, default=None, help="Use custom split from file")

    return parser

def objective(trial: optuna.trial.Trial) -> float:
    # Get global args (this will be set in main)
    global args, cont_features, cat_features, labels
    
    # Hyperparameters to tune
    n_blocks = trial.suggest_int("n_blocks", 2, 4)  # Reduced range for stability
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)  # Better learning rate range
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)  # Better weight decay range
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])  # Common batch sizes
    rex_weight = trial.suggest_float("rex_weight", 1, 10000, log=True)  # REx weight

    # Load data module with the suggested batch_size
    dm = load_datamodule(args, cont_features, cat_features, labels)

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
        rex_penalty_anneal_iters=args.rex_anneal_iters,
        episodic=(args.mode == "episodic")
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
    
    # Load data configuration - make these global so objective function can access them
    global cont_features, cat_features, labels
    cont_features, cat_features, labels = read_data_config(f"configs/{args.data}.json")
    
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
        study_name=f"{args.data}-{args.mode}{'-rex' if args.rex_weight > 0 else ''}-hyperparam"
    )

    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")