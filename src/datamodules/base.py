import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.datasets import ERMDataset
from src.utils import load_domains

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, data, cont_features, cat_features, labels, num_workers, batch_size, n_clusters, cluster_eval):
        super().__init__()
        self.data = data
        self.cont_features = cont_features
        self.cat_features = cat_features
        self.labels = labels

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.n_clusters = n_clusters
        self.cluster_eval = cluster_eval

        self.df = pd.read_csv(self.data)

        #self.df, self.domain_mapping = load_domains(self.df, domain_cols=self.domain_cols)
        #print(f"Identified {len(self.domain_mapping)} unique domains based on columns {self.domain_cols}")

        train_df, val_df, test_df = self.split_data()

        self.create_datasets(train_df, val_df, test_df)

    def create_datasets(self, train_df, val_df, test_df):
        self.train_ds = ERMDataset(
            df=train_df,
            cont_features=self.cont_features,
            cat_features=self.cat_features,
            labels=self.labels,
            train=True
        )

        self.val_ds = ERMDataset(
            df=val_df,
            cont_features=self.cont_features,
            cat_features=self.cat_features,
            labels=self.labels,
            train=False,
            feature_transformer=self.train_ds.feature_transformer,
            label_transformer=self.train_ds.label_transformer,
            cat_encoders=self.train_ds.cat_encoders,
        )

        self.test_ds = ERMDataset(
            df=test_df,
            cont_features=self.cont_features,
            cat_features=self.cat_features,
            labels=self.labels,
            train=False,
            feature_transformer=self.train_ds.feature_transformer,
            label_transformer=self.train_ds.label_transformer,
            cat_encoders=self.train_ds.cat_encoders,
        )

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
        
        test_cluster = sorted_cluster_ids[self.cluster_eval+1]
        
        test_df = df_with_clusters[df_with_clusters["domain_id"] == test_cluster].copy()

        print(f"Data split using {self.n_clusters} k-means clusters:")
        
        train_val_df = df_with_clusters[df_with_clusters["domain_id"] != test_cluster].copy()
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.25,
            random_state=1
        )

        print(f"  Training clusters: {[c for c in sorted_cluster_ids if c != test_cluster]}")
        print(f"  Validation cluster: {[sorted_cluster_ids[self.cluster_eval]]}")
        print(f"  Test cluster: {[test_cluster]}")
        
        return train_df, val_df, test_df

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )