from src.datasets import ERMDataset, EpisodicDataset
from .base import BaseDataModule
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class LOCODataModule(BaseDataModule):
    def __init__(self, data, cont_features, cat_features, labels, num_workers, batch_size, n_clusters, cluster_eval, dataset_type):
        super().__init__(data, cont_features, cat_features, labels, num_workers, batch_size, n_clusters, cluster_eval)

        self.dataset_type = dataset_type

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.n_clusters < 3:
            raise ValueError("n_clusters must be at least 3 to create train/val/test splits")
        
        feature_cols = [col for col in self.df.columns if col not in self.labels]
        
        X = self.df[feature_cols].copy()
        
        for cat_col in self.cat_features:
            if cat_col in X.columns:
                X[cat_col] = pd.Categorical(X[cat_col]).codes
        
        X = X.fillna(X.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=1, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        df_with_clusters = self.df.copy()
        df_with_clusters["domain_id"] = cluster_labels
        
        unique_clusters = np.unique(cluster_labels)
        cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]
        
        sorted_clusters = sorted(zip(unique_clusters, cluster_sizes), key=lambda x: x[1], reverse=True)
        sorted_cluster_ids = [x[0] for x in sorted_clusters]
        
        test_cluster = sorted_cluster_ids[0]  # Largest cluster for test
        val_cluster = sorted_cluster_ids[1]   # Second largest for validation
        train_clusters = sorted_cluster_ids[2:]  # Remaining for training
        
        test_df = df_with_clusters[df_with_clusters["domain_id"] == test_cluster]
        val_df = df_with_clusters[df_with_clusters["domain_id"] == val_cluster]
        train_df = df_with_clusters[df_with_clusters["domain_id"].isin(train_clusters)]
        
        print(f"Data split using {self.n_clusters} k-means clusters:")
        print(f"Train: {len(train_df)} samples ({len(train_clusters)} clusters)")
        print(f"Val: {len(val_df)} samples (1 cluster)")
        print(f"Test: {len(test_df)} samples (1 cluster)")
        
        return train_df, val_df, test_df
    
    def create_datasets(self, train_df, val_df, test_df):
        Dataset = ERMDataset#EpisodicDataset if self.dataset_type == "episodic" else 

        self.train_ds = Dataset(
            df=train_df,
            cont_features=self.cont_features,
            cat_features=self.cat_features,
            labels=self.labels,
            train=True
        )

        self.val_ds = Dataset(
            df=val_df,
            cont_features=self.cont_features,
            cat_features=self.cat_features,
            labels=self.labels,
            train=False,
            feature_transformer=self.train_ds.feature_transformer,
            label_transformer=self.train_ds.label_transformer,
            cat_encoders=self.train_ds.cat_encoders,
        )

        self.test_ds = Dataset(
            df=test_df,
            cont_features=self.cont_features,
            cat_features=self.cat_features,
            labels=self.labels,
            train=False,
            feature_transformer=self.train_ds.feature_transformer,
            label_transformer=self.train_ds.label_transformer,
            cat_encoders=self.train_ds.cat_encoders,
        )