import argparse
from src.model import FeatureTokenizerTransformer, FTConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from src.utils import read_data_config
from src.train import load_datamodule
import pytorch_lightning as pl
from scipy import stats

def get_args_parser():
    parser = argparse.ArgumentParser(description="Train FT Transformer model")
    
    parser.add_argument("--data", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--n_blocks", type=int, default=4, help="Number of FT Transformer blocks")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=32, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

    parser.add_argument("--mode", type=str, choices=["normal", "loco"], default="normal", help="Training mode")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for LOCO mode")
    parser.add_argument("--cluster_eval", type=int, default=None)

    # REx
    parser.add_argument("--rex_weight", type=float, default=0.0, help="Use REx for training")
    parser.add_argument("--rex_anneal_iters", type=int, default=280, help="Number of iterations to anneal REx weight")

    # IRM
    parser.add_argument("--irm_weight", type=int, default=0.0, help="Number of support samples per domain per episode")
    parser.add_argument("--irm_anneal_iters", type=int, default=280, help="Number of iterations to anneal IRM weight")

    # IB IRM
    parser.add_argument("--ib_weight", type=int, default=0.0, help="Number of support samples per domain per episode")
    parser.add_argument("--ib_anneal_iters", type=int, default=280, help="Number of iterations to anneal IB weight")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--custom", type=str, default=None, help="Use custom split from file")

    return parser

def run(seed, iteration) -> float:
    # Get global args (this will be set in main)
    global args, cont_features, cat_features, labels

    # Set the seed for this iteration
    pl.seed_everything(seed)

    # Load data module with the suggested batch_size
    dm = load_datamodule(args, args.batch_size, cont_features, cat_features, labels)
    dm.init()

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
        irm_weight=args.irm_weight,
        ib_weight=args.ib_weight,
        irm_penalty_anneal_iters=args.irm_anneal_iters,
        ib_penalty_anneal_iters=args.ib_anneal_iters
    )

    early_stopping = EarlyStopping(monitor="val/loss", patience=args.patience, mode="min")

    study_name = f"{args.data}-{args.mode}{'-rex' if args.rex_weight > 0 else ''}{'-irm' if args.irm_weight > 0 else ''}{'-ib' if args.ib_weight > 0 else ''}-significance"
    logger = TensorBoardLogger("runs", name=study_name, version=f"seed_{seed}", default_hp_metric=False)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"artifacts/{study_name}",
        filename=f"{study_name}-seed-{seed}-best",
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
    
    # ltt-normal
    comparison_values = [

    ]

    # ltt-normal-rex
#     comparison_values = [
#     ]
    
    print("Starting significance testing with 20 iterations...")

    # Store all loss values
    loss_values = []
    
    # Run 20 iterations with different seeds
    for i in range(20):
        seed = i +1
        print(f"Running iteration {i+1}/20 with seed {seed}")
        
        # Run training with seed i
        loss = run(seed=seed, iteration=i)
        loss_values.append(loss)
        
        print(f"Iteration {i+1} completed with loss: {loss:.6f}")

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Number of iterations completed: {len(loss_values)}")
    print("\nAll loss values:")
    for i, loss in enumerate(loss_values):
        print(f"  Seed {i}: {loss:.6f}")
    
    print(f"\nStatistics:")
    print(f"  Mean loss: {sum(loss_values)/len(loss_values):.6f}")
    print(f"  Min loss:  {min(loss_values):.6f}")
    print(f"  Max loss:  {max(loss_values):.6f}")
    print(f"  Std dev:   {(sum([(x - sum(loss_values)/len(loss_values))**2 for x in loss_values])/len(loss_values))**0.5:.6f}")
    
    # T-Test durchführen, falls Vergleichswerte vorhanden sind
    if comparison_values:
        print(f"\nT-Test Analysis:")
        print("-" * 40)
        t_statistic, p_value = stats.ttest_ind(loss_values, comparison_values)
        print(f"  T-statistic: {t_statistic:.6f}")
        print(f"  P-value:     {p_value:.6f}")
        print(f"  Comparison values count: {len(comparison_values)}")
        print(f"  Current values count:    {len(loss_values)}")
        
        # Interpretation
        alpha = 0.05
        if p_value < alpha:
            print(f"  Result: Statistically significant difference (p < {alpha})")
        else:
            print(f"  Result: No statistically significant difference (p >= {alpha})")
    else:
        print(f"\nT-Test Analysis:")
        print("-" * 40)
        print("  No comparison values provided - t-test skipped")
        print("  To perform t-test, add values to 'comparison_values' list")
    
    print("="*60)