from torch.utils.data import Dataset
import pandas as pd


class Couplets(Dataset):
    def __init__(self, file_path):
        self.first = pd.read_table(f"{file_path}/in.txt", header=None)[0]
        self.second = pd.read_table(f"{file_path}/out.txt", header=None)[0]
        self.df = pd.DataFrame({"first": self.first, "second": self.second})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        return sample["first"], sample["second"]
