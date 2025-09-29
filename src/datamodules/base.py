import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from src.datasets import ERMDataset

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, data, cont_features, cat_features, labels, num_workers, batch_size, dataset_type):
        super().__init__()
        self.data = data
        self.cont_features = cont_features
        self.cat_features = cat_features
        self.labels = labels

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.dataset_type = dataset_type

        self.df = self.load_data()

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data)
    
    def init(self):
        self.fit_categorical_encoders()
        
        train_df, val_df, test_df = self.split_data()
        self.create_datasets(train_df, val_df, test_df)
    
    def fit_categorical_encoders(self):
        from sklearn.preprocessing import LabelEncoder
        
        self.cat_encoders = {}
        for cat_feature in self.cat_features:
            if cat_feature in self.df.columns:
                encoder = LabelEncoder()
                encoder.fit(self.df[cat_feature].to_numpy())
                self.cat_encoders[cat_feature] = encoder

    def create_datasets(self, train_df, val_df, test_df):
        Dataset = ERMDataset

        self.train_ds = Dataset(
            df=train_df,
            cont_features=self.cont_features,
            cat_features=self.cat_features,
            labels=self.labels,
            train=True,
            cat_encoders=self.cat_encoders  # Pass pre-fitted categorical encoders
        )

        self.val_ds = Dataset(
            df=val_df,
            cont_features=self.cont_features,
            cat_features=self.cat_features,
            labels=self.labels,
            train=False,
            feature_transformer=self.train_ds.feature_transformer,
            label_transformer=self.train_ds.label_transformer,
            cat_encoders=self.cat_encoders,  # Use the same pre-fitted encoders
        )

        if test_df is not None:
            self.test_ds = Dataset(
                df=test_df,
                cont_features=self.cont_features,
                cat_features=self.cat_features,
                labels=self.labels,
                train=False,
                feature_transformer=self.train_ds.feature_transformer,
                label_transformer=self.train_ds.label_transformer,
                cat_encoders=self.cat_encoders,  # Use the same pre-fitted encoders
            )

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = self.df[self.df["split"] == "train"]
        val_df = self.df[self.df["split"] == "test"]
        
        return train_df, val_df, None

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