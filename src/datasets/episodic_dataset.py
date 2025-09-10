import torch
from torch.utils.data import IterableDataset
from pandas import DataFrame
from ..utils import preprocess_data

class EpisodicDataset(IterableDataset):
    def __init__(
        self,
        df: DataFrame,
        cont_features,
        cat_features,
        labels,
        train,
        feature_transformer=None,
        label_transformer=None,
        cat_encoders=None
    ):
        self.n_samples = len(df)

        # Store domain information
        self.domain = torch.as_tensor(df["domain_id"].to_numpy(), dtype=torch.long)

        # Use the centralized preprocessing function
        x_cont, x_cat, y, self.feature_transformer, self.label_transformer, self.cat_encoders = preprocess_data(
            df=df,
            cont_features=cont_features,
            cat_features=cat_features,
            labels=labels,
            train=train,
            feature_transformer=feature_transformer,
            label_transformer=label_transformer,
            cat_encoders=cat_encoders,
        )

        # Convert to tensors
        self.x_cont = torch.as_tensor(x_cont, dtype=torch.float32)
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x_cont[index], self.x_cat[index], self.y[index], self.domain[index]

    def __len__(self):
        return self.n_samples