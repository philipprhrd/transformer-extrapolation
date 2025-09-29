import torch
from torch.utils.data import Dataset
from pandas import DataFrame
from src.utils import preprocess_data

def custom_collate_fn(batch):
    """Custom collate function to handle None values in categorical features"""
    x_cont_list, x_cat_list, y_list, domain_list = zip(*batch)
    
    # Stack continuous features and labels normally
    x_cont_batch = torch.stack(x_cont_list)
    y_batch = torch.stack(y_list)
    domain_batch = torch.stack(domain_list)
    
    # Handle categorical features - if all are None, return None, otherwise stack
    if all(x_cat is None for x_cat in x_cat_list):
        x_cat_batch = None
    else:
        x_cat_batch = torch.stack(x_cat_list)
    
    return x_cont_batch, x_cat_batch, y_batch, domain_batch

class ERMDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
        cont_features,
        cat_features,
        labels,
        train,
        feature_transformer=None,
        label_transformer=None,
        cat_encoders=None,
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
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long) if x_cat is not None else None
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def get_cardinalities(self):
        if self.cat_encoders is None:
            return []
        return [len(encoder.classes_) for encoder in self.cat_encoders.values()]

    def __getitem__(self, index):
        x_cat_item = self.x_cat[index] if self.x_cat is not None else None
        return self.x_cont[index], x_cat_item, self.y[index], self.domain[index]

    def __len__(self):
        return self.n_samples