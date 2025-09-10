import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.datasets import EpisodicDataset, ERMDataset
from src.utils import load_domains

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, data, dataset_type, cont_features, cat_features, labels, domain_cols, num_workers, batch_size):
        super().__init__()
        self.data = data
        self.dataset_type = dataset_type
        self.cont_features = cont_features
        self.cat_features = cat_features
        self.labels = labels
        self.domain_cols = domain_cols

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.df = pd.read_csv(self.data)

        self.df, self.domain_mapping = load_domains(self.df, domain_cols=self.domain_cols)

        print(f"Identified {len(self.domain_mapping)} unique domains based on columns {self.domain_cols}")

        train_df, val_df, test_df = self.split_data()

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

    def split_data(self):
        train_val_df, test_df = train_test_split(self.df, test_size=0.2, random_state=1)
        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=1)

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