from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class StackedDataset(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(
            np.stack(
                [
                    df["swingProb"].to_numpy(),
                    df["whiffProb"].to_numpy(),
                    df["foulProb"].to_numpy(),
                    df["flyball"].to_numpy(),
                    df["groundball"].to_numpy(),
                    df["popup"].to_numpy(),
                    df["linedrive"].to_numpy(),
                ],
                axis=1,
            ).astype(np.float32)
        )

        bip = np.stack(df["bipArr"].to_numpy()).astype(np.float32)
        self.y = torch.tensor(
            np.stack(
                [
                    df["swing"].to_numpy(),
                    df["whiff"].to_numpy(),
                    df["foul"].to_numpy(),
                    bip[:, 0],  # fly ball
                    bip[:, 1],  # ground ball
                    bip[:, 2],  # line drive
                    bip[:, 3],  # popup
                ],
                axis=1,
            ).astype(np.float32)
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


def split_loaders(dataset, dates, batch_size, cutoff=datetime(2025, 7, 1).date()):
    test_idx = [i for i, d in enumerate(dates) if d >= cutoff]
    pretrain_idx = [i for i, d in enumerate(dates) if d < cutoff]

    gen = torch.Generator().manual_seed(26)
    shuffled = torch.randperm(len(pretrain_idx), generator=gen).tolist()
    val_size = int(0.125 * len(pretrain_idx))
    val_idx = [pretrain_idx[i] for i in shuffled[:val_size]]
    train_idx = [pretrain_idx[i] for i in shuffled[val_size:]]

    return (
        DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        ),
        DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False),
        DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False),
    )
