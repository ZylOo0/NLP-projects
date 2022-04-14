from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, header=None, sep="\t")
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.df.iloc[idx, 0]
        tgt = self.df.iloc[idx, 1]
        return src, tgt
