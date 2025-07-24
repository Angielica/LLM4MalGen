import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

class EmbeddingDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()

        if isinstance(file_path, pd.DataFrame):
            self.data = file_path
        else:
            self.data = pd.read_parquet(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        x = torch.tensor(row['Embedding'])
        y = torch.tensor(row['Label']).float()

        return x, y