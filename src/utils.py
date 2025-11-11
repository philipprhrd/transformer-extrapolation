import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import json
import joblib
from typing import List, Tuple
import os

def read_data_config(path: str):
    with open(path, "r") as f:
        params = json.load(f)

    cont_features = params.get("cont_features", [])
    cat_features = params.get("cat_features", [])
    labels = params.get("labels", [])

    return cont_features, cat_features, labels

def load_domains(df: pd.DataFrame, domain_cols: list):
    df["_domain"] = df[domain_cols].astype(str).agg("_".join, axis=1)

    domain_mapping = {domain: idx for idx, domain in enumerate(df["_domain"].unique())}

    df["domain_id"] = df["_domain"].map(domain_mapping)

    df = df.drop(columns=["_domain"])

    return df, domain_mapping

def preprocess_data(
    df: pd.DataFrame,
    cont_features: List[str],
    cat_features: List[str],
    labels: List[str],
    train: bool = True,
    feature_transformer=None,
    label_transformer=None,
    cat_encoders=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler, dict]:
    # Create transformers if not provided
    if feature_transformer is None:
        feature_transformer = StandardScaler()
    if label_transformer is None:
        label_transformer = StandardScaler()
    if cat_encoders is None:
        cat_encoders = {}

    # Process continuous features
    x_cont = df[cont_features].to_numpy()
    
    if train:
        feature_transformer.fit(x_cont)
    
    x_cont = feature_transformer.transform(x_cont).astype(np.float32)

    # Process categorical features
    x_cat = None
    if cat_features and len(cat_features) > 0:
        encoded_cols = []
        
        for col in cat_features:
            # Use existing encoder if provided, otherwise create and fit new one
            if col in cat_encoders:
                encoder = cat_encoders[col]
                codes = encoder.transform(df[col].to_numpy()).astype(np.int64)
            else:
                # Create new encoder and fit if training
                encoder = LabelEncoder()
                if train:
                    codes = encoder.fit_transform(df[col].to_numpy()).astype(np.int64)
                    cat_encoders[col] = encoder
                else:
                    # If not training but no encoder provided, this is an error
                    raise ValueError(f"No encoder provided for categorical feature '{col}' and train=False")
            
            encoded_cols.append(codes)

        x_cat = np.stack(encoded_cols, axis=1).astype(np.int64)
    else:
        # No categorical features: create an empty array for compatibility
        x_cat = np.empty((len(df), 0), dtype=np.int64)

    # Process labels
    y = df[labels].to_numpy()
    
    if train:
        label_transformer.fit(y)
    
    y = label_transformer.transform(y).astype(np.float32)

    return x_cont, x_cat, y, feature_transformer, label_transformer, cat_encoders


def save_scalers(feature_transformer, label_transformer, cat_encoders, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save feature scaler
    if feature_transformer is not None:
        joblib.dump(feature_transformer, os.path.join(save_dir, "feature_scaler.pkl"))
        print(f"Saved feature scaler to {os.path.join(save_dir, "feature_scaler.pkl")}")
    
    # Save label scaler
    if label_transformer is not None:
        joblib.dump(label_transformer, os.path.join(save_dir, "label_scaler.pkl"))
        print(f"Saved label scaler to {os.path.join(save_dir, "label_scaler.pkl")}")
    
    # Save categorical encoders
    if cat_encoders is not None and len(cat_encoders) > 0:
        joblib.dump(cat_encoders, os.path.join(save_dir, "cat_encoders.pkl"))
        print(f"Saved categorical encoders to {os.path.join(save_dir, "cat_encoders.pkl")}")