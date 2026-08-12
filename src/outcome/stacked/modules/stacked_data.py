from this import d
from datetime import date

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from src.outcome.modules.data.strike_HBP_data import preprocess

PROB_COLUMNS = [
    "swing_prob",
    "ball_prob",
    "whiff_prob",
    "foul_prob",
    "bip_prob_0",
    "bip_prob_1",
    "bip_prob_2",
    "bip_prob_3",
    "bip_prob_4",
]
BIP_COLUMNS = [
    "bip_label_0",
    "bip_label_1",
    "bip_label_2",
    "bip_label_3",
    "bip_label_4",
]


class StackedData(Dataset):
    def __init__(self, df):
        self.probs = torch.from_numpy(
            df.select(PROB_COLUMNS).to_numpy().astype(np.float32)
        )
        
        self.swing = torch.from_numpy(df["swing_label"].to_numpy().astype(np.float32))
        self.ball = torch.from_numpy(df["ball_label"].to_numpy().astype(np.float32))
        self.whiff = torch.from_numpy(df["whiff_label"].to_numpy().astype(np.float32))
        self.foul = torch.from_numpy(df["foul_label"].to_numpy().astype(np.float32))
        self.bip = torch.from_numpy(
            df.select(BIP_COLUMNS).to_numpy().astype(np.float32)
        )

    def __len__(self):
        return len(self.swing)

    def __getitem__(self, idx):
        return {
            "probs": self.probs[idx],
            "swing": self.swing[idx],
            "ball": self.ball[idx],
            "whiff": self.whiff[idx],
            "foul": self.foul[idx],
            "bip": self.bip[idx],
        }


def split_loaders(df, cutoff: date, batch_size, prefetch, workers, pin=True):
    indexed = df.with_row_index("row_idx")
    trainval = indexed.filter(pl.col("game_date") < cutoff)
    test = indexed.filter(pl.col("game_date") >= cutoff)

    train = trainval.sample(fraction=0.8, seed=26)
    val = trainval.filter(~pl.col("row_idx").is_in(train["row_idx"].to_list()))
    dataset = StackedData(df)

    def loader(frame, shuffle):
        indices = frame["row_idx"].to_list()
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            prefetch_factor=prefetch if workers else None,
            pin_memory=pin,
        )

    return loader(train, True), loader(val, False), loader(test, False)
