import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import json
import joblib
import os
from typing import List, Tuple

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
    """
    Preprocess data by scaling continuous features, encoding categorical features, and scaling labels.
    
    Args:
        df: Input DataFrame
        cont_features: List of continuous feature column names
        cat_features: List of categorical feature column names
        labels: List of label column names
        train: Whether this is training data (fit transformers) or test data (use existing transformers)
        feature_transformer: Existing StandardScaler for features (if train=False)
        label_transformer: Existing StandardScaler for labels (if train=False)
        cat_encoders: Existing categorical encoders dict (if train=False)
    
    Returns:
        Tuple of (x_cont, x_cat, y, feature_transformer, label_transformer, cat_encoders)
    """
    
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
        # No categorical features: return None for FTTransformer compatibility
        x_cat = None

    # Process labels
    y = df[labels].to_numpy()
    
    if train:
        label_transformer.fit(y)
    
    y = label_transformer.transform(y).astype(np.float32)

    return x_cont, x_cat, y, feature_transformer, label_transformer, cat_encoders


def get_loco_split(
    df: pd.DataFrame, 
    n_splits: int = 3, 
    klims: tuple = (3, 10), 
    n_folds: int = 10
) -> list:
    """
    Get the Leave-One-Cluster-Out (LOCO) splits 
    for cross-validation from the algorithm here: https://doi.org/10.1039/C8ME00012C.
    """
    pass


def save_scalers(feature_transformer, label_transformer, cat_encoders, save_dir: str):
    """
    Save the scalers to files for later reuse.
    
    Args:
        feature_transformer: StandardScaler for continuous features
        label_transformer: StandardScaler for labels
        cat_encoders: Dictionary of LabelEncoders for categorical features
        save_dir: Directory to save the scalers
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save feature scaler
    if feature_transformer is not None:
        joblib.dump(feature_transformer, os.path.join(save_dir, "feature_scaler.pkl"))
        print(f"Saved feature scaler to {os.path.join(save_dir, 'feature_scaler.pkl')}")
    
    # Save label scaler
    if label_transformer is not None:
        joblib.dump(label_transformer, os.path.join(save_dir, "label_scaler.pkl"))
        print(f"Saved label scaler to {os.path.join(save_dir, 'label_scaler.pkl')}")
    
    # Save categorical encoders
    if cat_encoders is not None and len(cat_encoders) > 0:
        joblib.dump(cat_encoders, os.path.join(save_dir, "cat_encoders.pkl"))
        print(f"Saved categorical encoders to {os.path.join(save_dir, 'cat_encoders.pkl')}")


def load_scalers(save_dir: str):
    """
    Load the saved scalers from files.
    
    Args:
        save_dir: Directory containing the saved scalers
        
    Returns:
        Tuple of (feature_transformer, label_transformer, cat_encoders)
    """
    feature_transformer = None
    label_transformer = None
    cat_encoders = None
    
    # Load feature scaler
    feature_scaler_path = os.path.join(save_dir, "feature_scaler.pkl")
    if os.path.exists(feature_scaler_path):
        feature_transformer = joblib.load(feature_scaler_path)
        print(f"Loaded feature scaler from {feature_scaler_path}")
    
    # Load label scaler
    label_scaler_path = os.path.join(save_dir, "label_scaler.pkl")
    if os.path.exists(label_scaler_path):
        label_transformer = joblib.load(label_scaler_path)
        print(f"Loaded label scaler from {label_scaler_path}")
    
    # Load categorical encoders
    cat_encoders_path = os.path.join(save_dir, "cat_encoders.pkl")
    if os.path.exists(cat_encoders_path):
        cat_encoders = joblib.load(cat_encoders_path)
        print(f"Loaded categorical encoders from {cat_encoders_path}")
    
    return feature_transformer, label_transformer, cat_encoders
    