import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


def preprocess(df, hbp=False):
    df = df.with_columns(pl.col("kmu").list.slice(0, 10))

    # if preprocessing for hbp
    if hbp:
        takes = ["called_strike", "ball", "blocked_ball", "intent_ball", "hit_by_pitch"]
        df = df.filter(pl.col("description").is_in(takes))
        df = df.with_columns(pl.col("description") == "hit_by_pitch")
        return df

    # ball strike
    takes = ["called_strike", "ball", "blocked_ball", "intent_ball"]
    df = df.filter(pl.col("description").is_in(takes))
    df = df.with_columns(
        ball=pl.col("description").is_in(["ball", "blocked_ball", "intent_ball"])
    )

    return df


class HBPStrikeData(Dataset):
    def __init__(self, df, label):
        super().__init__()

        # pitch embed
        kmu = df["kmu"].explode().to_numpy().astype(np.float32).reshape(-1, 10)
        self.pitch = torch.from_numpy(kmu)

        # label
        self.label = torch.from_numpy(df[label].to_numpy().astype(np.float32))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {"pitch": self.pitch[idx], "label": self.label[idx]}


def split_loaders(
    df,
    label,
    cutoff,
    batch_size,
    prefetch,
    workers,
    drop_last,
    pin=True,
):
    tv = df.with_row_index(name="ridx").filter(pl.col("game_date") < cutoff)

    train_subset = tv.sample(fraction=0.8, seed=26)
    val_subset = tv.filter(~pl.col("ridx").is_in(train_subset["ridx"]))
    test_subset = df.filter(pl.col("game_date") > cutoff)

    train_subset = train_subset.drop("ridx")
    val_subset = val_subset.drop("ridx")

    train_ds = HBPStrikeData(train_subset, label)
    val_ds = HBPStrikeData(val_subset, label)
    test_ds = HBPStrikeData(test_subset, label)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        prefetch_factor=prefetch,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
