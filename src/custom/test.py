from src.datamodules import BaseDataModule

class CustomDataModule(BaseDataModule):
    def __init__(self, data, dataset_type, cont_features, cat_features, labels, domain_cols, num_workers, batch_size):
        super().__init__(data, dataset_type, cont_features, cat_features, labels, domain_cols, num_workers, batch_size)

    def split_data(self):
        pass