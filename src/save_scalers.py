import pandas as pd
from src.utils import read_data_config, preprocess_data, save_scalers

def main():
    path = "lin"
    config_path = f"configs/{path}.json"
    data_path = f"data/{path}.csv"
    cont_features, cat_features, labels = read_data_config(config_path)
    print(f"Using config: {config_path}\n  cont_features={cont_features}\n  cat_features={cat_features}\n  labels={labels}")

    df = pd.read_csv(data_path)

    # Use only training rows if `split` exists
    if "split" in df.columns:
        df_train = df[df["split"] == "train"].reset_index(drop=True)
        print(f"Selected {len(df_train)} training rows from {data_path}")
    else:
        df_train = df
        print(f"No 'split' column found; using all {len(df_train)} rows as training data")

    # Fit transformers
    x_cont, x_cat, y, feature_transformer, label_transformer, cat_encoders = preprocess_data(
        df_train, cont_features, cat_features, labels, train=True, feature_transformer=None, label_transformer=None, cat_encoders=None
    )

    save_scalers(feature_transformer, label_transformer, cat_encoders, f"artifacts/{path}_encoders")


if __name__ == "__main__":
    main()
