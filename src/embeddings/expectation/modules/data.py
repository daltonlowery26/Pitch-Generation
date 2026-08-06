from datetime import date
from typing import NamedTuple

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from src.embeddings.expectation.modules.utils import repertoire_mask


class UsageData(NamedTuple):
    train_set: Subset
    val_set: Subset
    test_set: Subset
    mask: torch.Tensor
    pitchers: int
    batters: int
    pitches: int
    max_len: int


class UsageDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        # sort df to constant order and id
        df = df.sort(["pitchBatter", "atBatNumber", "pitch_number"])
        ids = torch.from_numpy(df["pitchBatter"].to_numpy())

        # load embeds
        emb = df["genPitch"].explode().to_numpy().astype(np.float32).reshape(-1, 10)
        pitch_group = df["pitch_group"].to_numpy().astype(np.int32)

        # train and labels
        self.embeds = torch.from_numpy(emb)
        self.labels = torch.from_numpy(pitch_group)

        # constants
        self.abNum = torch.from_numpy(df["atBatNumber"].to_numpy().astype(np.int32))
        self.count = torch.from_numpy(df["countArr"].to_numpy().astype(np.int32))
        self.pitcherID = torch.from_numpy(df["pitcher_year"].to_numpy().astype(np.int32))
        self.batterID = torch.from_numpy(df["batter_year"].to_numpy().astype(np.int32))
        self.at_bat_number = torch.from_numpy(df["at_bat_number"].to_numpy().astype(np.int32))
        self.pitch_number = torch.from_numpy(df["pitch_number"].to_numpy().astype(np.int32))

        # change in atbat
        changes = torch.where(ids[1:] != ids[:-1])[0] + 1
        self.boundaries = torch.cat(
            [torch.tensor([0]), changes, torch.tensor([len(ids)])]
        )

        # per-group start date
        date_series = df["game_date"]
        starts = self.boundaries[:-1].tolist()
        self.group_dates = [date_series[s] for s in starts]

    def __len__(self):
        return len(self.boundaries) - 1

    def __getitem__(self, idx):
        s, e = self.boundaries[idx].item(), self.boundaries[idx + 1].item()
        return {
            "embeds": self.embeds[s:e],
            "labels": self.labels[s:e],
            "abNum": self.abNum[s:e],
            "count": self.count[s:e],
            "pitcherID": self.pitcherID[s:e],
            "batterID": self.batterID[s:e],
            "at_bat_number": self.at_bat_number[s:e],
            "pitch_number": self.pitch_number[s:e],
        }


# collate fn
def collate_fn(batch):
    embeds = [b["embeds"] for b in batch]
    labels = [b["labels"] for b in batch]
    abNum = [b["abNum"] for b in batch]
    count = [b["count"] for b in batch]
    pitcherID = [b["pitcherID"] for b in batch]
    batterID = [b["batterID"] for b in batch]
    at_bat_number = [b["at_bat_number"] for b in batch]
    pitch_number = [b["pitch_number"] for b in batch]

    lengths = torch.tensor([e.shape[0] for e in embeds])

    embeds = nn.utils.rnn.pad_sequence(embeds, batch_first=True)

    abNum = nn.utils.rnn.pad_sequence(abNum, batch_first=True)
    count = nn.utils.rnn.pad_sequence(count, batch_first=True)
    pitcherID = nn.utils.rnn.pad_sequence(pitcherID, batch_first=True)
    batterID = nn.utils.rnn.pad_sequence(batterID, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
    at_bat_number = nn.utils.rnn.pad_sequence(
        at_bat_number, batch_first=True, padding_value=-1
    )
    pitch_number = nn.utils.rnn.pad_sequence(
        pitch_number, batch_first=True, padding_value=-1
    )

    return {
        "embeds": embeds,
        "labels": labels,
        "lengths": lengths,
        "abNum": abNum,
        "count": count,
        "pitcherID": pitcherID,
        "batterID": batterID,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
    }


def load_usage_data(path, val_start: date, val_end: date, test_cutoff: date):
    df = pl.read_parquet(path)

    # number of pitchers, batters, len
    pitchers = df["pitcher_year"].n_unique()
    batters = df["batter_year"].n_unique()
    pitches = df["pitch_group"].n_unique()
    max_len = df.group_by("pitchBatter").len().get_column("len").max()

    # create dataset and mask based on pitcher count
    dataset = UsageDataset(df)
    mask = repertoire_mask(df, num_pitchers=pitchers, num_pitches=pitches)
    group_dates = dataset.group_dates

    # seperate train and val indices
    train_indices = [
        i
        for i, d in enumerate(group_dates)
        if d < val_start or (val_end <= d < test_cutoff)
    ]
    val_indices = [i for i, d in enumerate(group_dates) if val_start <= d < val_end]
    test_indices = [i for i, d in enumerate(group_dates) if d >= test_cutoff]

    return UsageData(
        train_set=Subset(dataset, train_indices),
        val_set=Subset(dataset, val_indices),
        test_set=Subset(dataset, test_indices),
        mask=mask,
        pitchers=pitchers,
        batters=batters,
        pitches=pitches,
        max_len=max_len,
    )
