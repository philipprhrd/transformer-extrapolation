import argparse
from src.model import FeatureTokenizerTransformer, FTConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from src.utils import read_data_config
from src.datamodules import BaseDataModule, LOCODataModule
import importlib

def get_args_parser():
    """Sample usage:

    "Interpolation"-training with 1 cluster as extrapolation test:
    
    ``python -m src.train <...> --mode normal --cluster_eval <1..n_cluster>``

    "LOCO"-training with 1 cluster as extrapolation test:

    ``python -m src.train <...> --mode loco --cluster_eval <1..n_cluster>``

    "LOCO"-training with REx:

    ``python -m src.train <...> --mode loco --cluster_eval <1..n_cluster> --rex_weight <float>``

    "LOCO"-training with episodic training (and REx):

    ``python -m src.train <...> --mode episodic --cluster_eval <1..n_cluster> [--rex_weight <float>]``

    Completely custom logic (you have to define all splits in a python file and pass the name here):

    ``python -m src.train <...> --custom <python_file_name (in src/custom)>``
    """

    parser = argparse.ArgumentParser(description="Train FT Transformer model")
    
    parser.add_argument("--data", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--n_blocks", type=int, default=4, help="Number of FT Transformer blocks")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=32, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

    parser.add_argument("--mode", type=str, choices=["normal", "loco", "episodic"], default="normal", help="Training mode")
    parser.add_argument("--rex_weight", type=float, default=0.0, help="Use REx for training")
    parser.add_argument("--rex_anneal_iters", type=int, default=280, help="Number of iterations to anneal REx weight")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for LOCO mode")
    parser.add_argument("--cluster_eval", type=int, default=None)

    parser.add_argument("--custom", type=str, default=None, help="Use custom split from file")

    return parser


def load_datamodule(args, cont_features, cat_features, labels):
    if args.custom:
        module = importlib.import_module(f"src.custom.{args.custom}")
        DataModuleClass = getattr(module, "CustomDataModule")

        return DataModuleClass(
            data=f"data/{args.data}.csv",
            num_workers=11, 
            batch_size=args.batch_size,
            cont_features=cont_features,
            cat_features=cat_features,
            labels=labels
        )

    if args.mode in ["loco", "episodic"]:
        return LOCODataModule(
            data=f"data/{args.data}.csv",
            num_workers=11, 
            batch_size=args.batch_size,
            cont_features=cont_features,
            cat_features=cat_features,
            labels=labels,
            n_clusters=args.n_clusters,
            cluster_eval=args.cluster_eval,
            dataset_type=args.mode
        )
    
    return BaseDataModule(
        data=f"data/{args.data}.csv",
        num_workers=11,
        batch_size=args.batch_size,
        cont_features=cont_features,
        cat_features=cat_features,
        labels=labels,
        n_clusters=args.n_clusters,
        cluster_eval=args.cluster_eval
    )

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    # Data loading

    cont_features, cat_features, labels = read_data_config(f"configs/{args.data}.json")

    dm = load_datamodule(args, cont_features, cat_features, labels)

    # Model configuration

    config = FTConfig(
        n_cont_features=len(cont_features),
        cat_cardinalities=dm.train_ds.get_cardinalities(),
        d_out=len(labels),
        n_blocks=args.n_blocks,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        rex_weight=args.rex_weight,
        rex_penalty_anneal_iters=args.rex_anneal_iters,
        episodic=(args.mode == "episodic")
    )

    print("-- Config --")
    print(config)
    print("------------")

    early_stopping = EarlyStopping(monitor="val/loss", patience=args.patience, mode="min")

    experiment_name = f"{args.data}-{args.mode}{'-rex' if args.rex_weight > 0 else ''}-finetuning"
    logger = TensorBoardLogger("runs", name=experiment_name, default_hp_metric=False)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"artifacts/{experiment_name}",
        filename=f"{experiment_name}-best",
        monitor="val/loss",
        mode="min",
        save_top_k=1
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[early_stopping, ckpt_cb],
        logger=logger
    )

    model = FeatureTokenizerTransformer(config)

    # Training

    trainer.fit(model, dm)

    # Post training

    best_path = ckpt_cb.best_model_path
    print(f"Best model path: {best_path}")