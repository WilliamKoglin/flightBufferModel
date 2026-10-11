# Change dataframs into a pytorch dataset

import torch
from torch.utils.data import Dataset

from features import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET


class FlightDelayDataset(Dataset):
    def __init__(self, df):
        cat_idx_cols = [c + "_idx" for c in CATEGORICAL_COLS]
        num_scaled_cols = [c + "_scaled" for c in NUMERICAL_COLS]

        # instance variables
        self.cat = torch.tensor(df[cat_idx_cols].values, dtype=torch.long)
        self.num = torch.tensor(df[num_scaled_cols].values, dtype=torch.float32)
        self.target = torch.tensor(df[TARGET].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        return self.cat[i], self.num[i], self.target[i]