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
    parser.add_argument("--n_blocks", type=int, default=4, help="Number of FT Transformer blocks")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=32, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

    parser.add_argument("--dataset", type=str, choices=["erm", "loco"], default="erm", help="Type of dataset to use")
    parser.add_argument("--split", type=str, choices=["random", "kmeans", "custom"], default="random", help="Type of split to use")
    parser.add_argument("--domain_cols", type=list, default=[], help="List of domain columns")

    return parser


def load_datamodule(args, cont_features, cat_features, labels):
    if args.split in ["kmeans", "custom"]:
        assert args.dataset == "loco", "LOCO dataset must be used with kmeans or custom split"
        #return LOCODataModule(f"data/{args.data}.csv", dataset_type=args.dataset, num_workers=11, batch_size=args.batch_size)
    else:
        assert args.split == "random", "Only random split is supported for ERM dataset"
        return BaseDataModule(
            data=f"data/{args.data}.csv", 
            dataset_type=args.dataset, 
            num_workers=11, 
            batch_size=args.batch_size,
            cont_features=cont_features,
            cat_features=cat_features,
            labels=labels,
            domain_cols=args.domain_cols,
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
        weight_decay=args.weight_decay
    )

    print("-- Config --")
    print(config)
    print("------------")

    early_stopping = EarlyStopping(monitor="val/loss", patience=args.patience, mode="min")

    logger = TensorBoardLogger("runs", name="finetuning", default_hp_metric=False)

    ckpt_cb = ModelCheckpoint(
        dirpath="artifacts",
        filename=f"-{args.data}-{logger.name}-v{logger.version}"+"-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[early_stopping, ckpt_cb],
        logger=logger,
        devices="auto"
    )

    model = FeatureTokenizerTransformer(config)

    # Training

    trainer.fit(model, dm)

    # Post training

    best_path = ckpt_cb.best_model_path
    print(f"Best model path: {best_path}")