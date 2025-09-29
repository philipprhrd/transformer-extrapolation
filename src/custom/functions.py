from src.datamodules import BaseDataModule

class CustomDataModule(BaseDataModule):
    def __init__(self, data, cont_features, cat_features, labels, num_workers, batch_size, n_clusters, cluster_eval, dataset_type):
        super().__init__(data, cont_features, cat_features, labels, num_workers, batch_size, n_clusters, cluster_eval, dataset_type)

    def load_data(self):
        df = super().load_data()
        df["domain_id"] = 0
        return df

    def split_data(self):
        train_df = self.df[self.df["split"] == "train"]
        val_df = self.df[self.df["split"] == "extrapolation"]

        return train_df, val_df, None
