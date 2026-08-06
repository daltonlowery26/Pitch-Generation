import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


class OutcomeProb(Dataset):
    def __init__(self, df):
        df = df.sort(["pitchBatter", "atBatNumber", "pitch_number"])
        ids = torch.tensor(df["pitchBatter"].to_list())

        pitchEmb = df["pitchGen"].explode().to_numpy().astype(np.float32).reshape(-1, 10)
        expEmb = df["expEmb"].explode().to_numpy().astype(np.float32).reshape(-1, 4)
        bipLabel = (
            df["bipArr"]
            .explode()
            .fill_null(0)
            .to_numpy()
            .astype(np.int32)
            .reshape(-1, 5)
        )

        self.pitchEmb = torch.from_numpy(pitchEmb)
        self.expEmb = torch.from_numpy(expEmb)
        self.bipLabel = torch.from_numpy(bipLabel)

        self.batter_id = torch.from_numpy(df["batter_year"].to_numpy().astype(np.int32))
        self.pitcher_id = torch.from_numpy(
            df["pitcher_year"].to_numpy().astype(np.int32)
        )
        self.count = torch.from_numpy(df["countArr"].to_numpy().astype(np.int32))
        self.swing = torch.from_numpy(df["swing"].to_numpy().astype(np.int32))
        self.whiff = torch.from_numpy(df["whiff"].to_numpy().astype(np.int32))
        self.foul = torch.from_numpy(df["foul"].to_numpy().astype(np.int32))
        self.abNum = torch.from_numpy(df["atBatNumber"].to_numpy().astype(np.int32))

        changes = torch.where(ids[1:] != ids[:-1])[0] + 1
        self.boundaries = torch.cat(
            [torch.tensor([0]), changes, torch.tensor([len(ids)])]
        )

    def __len__(self):
        return len(self.boundaries) - 1

    def __getitem__(self, idx):
        s = self.boundaries[idx].item()
        e = self.boundaries[idx + 1].item()
        return {
            "pitchEmb": self.pitchEmb[s:e],
            "pitcherID": self.pitcher_id[s:e],
            "batterID": self.batter_id[s:e],
            "bipLabel": self.bipLabel[s:e],
            "expEmb": self.expEmb[s:e],
            "abNum": self.abNum[s:e],
            "swing": self.swing[s:e],
            "foul": self.foul[s:e],
            "count": self.count[s:e],
            "whiff": self.whiff[s:e],
        }


def collate_fn(batch):
    lengths = torch.tensor([b["pitchEmb"].shape[0] for b in batch])
    return {
        "pitchEmb": pad_sequence([b["pitchEmb"] for b in batch], batch_first=True),
        "batterID": pad_sequence([b["batterID"] for b in batch], batch_first=True),
        "pitcherID": pad_sequence([b["pitcherID"] for b in batch], batch_first=True),
        "expEmb": pad_sequence([b["expEmb"] for b in batch], batch_first=True),
        "abNum": pad_sequence([b["abNum"] for b in batch], batch_first=True),
        "count": pad_sequence([b["count"] for b in batch], batch_first=True),
        "bipLabel": pad_sequence(
            [b["bipLabel"] for b in batch], batch_first=True, padding_value=-1
        ),
        "swing": pad_sequence(
            [b["swing"] for b in batch], batch_first=True, padding_value=-1
        ),
        "foul": pad_sequence(
            [b["foul"] for b in batch], batch_first=True, padding_value=-1
        ),
        "whiff": pad_sequence(
            [b["whiff"] for b in batch], batch_first=True, padding_value=-1
        ),
        "lengths": lengths,
    }


class LenSampler(Sampler):
    def __init__(
        self,
        indices,
        lengths,
        batch_size,
        bucket_size,
        shuffle,
        drop_last,
    ):
        self.indices = np.asarray(indices)
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.indices)
        order = np.random.permutation(n) if self.shuffle else np.arange(n)
        batches = []

        for i in range(0, n, self.bucket_size):
            bucket = order[i : i + self.bucket_size]
            bucket = bucket[np.argsort(self.lengths[bucket], kind="stable")]
            for j in range(0, len(bucket), self.batch_size):
                b = bucket[j : j + self.batch_size]
                if self.drop_last and len(b) < self.batch_size:
                    continue
                batches.append(self.indices[b].tolist())
        if self.shuffle:
            np.random.shuffle(batches)
        yield from batches

    def __len__(self):
        n = len(self.indices)
        return (
            n // self.batch_size
            if self.drop_last
            else (n + self.batch_size - 1) // self.batch_size
        )


def split_loaders(
    data,
    df,
    batch_size,
    prefetch,
    workers,
    drop_last,
    cutoff,
    bucket_size=None,
    pin=True,
):
    if bucket_size is None:
        bucket_size = 250 * batch_size

    all_dates = df["game_date"]
    group_dates = [all_dates[data.boundaries[i].item()] for i in range(len(data))]

    test_idx = [i for i, d in enumerate(group_dates) if d >= cutoff]
    pretrain_idx = [i for i, d in enumerate(group_dates) if d < cutoff]

    gen = torch.Generator().manual_seed(26)
    shuffled = torch.randperm(len(pretrain_idx), generator=gen).tolist()
    val_size = int(0.125 * len(pretrain_idx))
    val_idx = [pretrain_idx[i] for i in shuffled[:val_size]]
    train_idx = [pretrain_idx[i] for i in shuffled[val_size:]]

    seq_lengths = torch.diff(data.boundaries).numpy()

    def make_loader(indices, shuffle):
        lengths = seq_lengths[indices]
        sampler = LenSampler(
            indices, lengths, batch_size, bucket_size, drop_last=drop_last, shuffle=shuffle
        )
        return DataLoader(
            data,
            num_workers=workers,
            prefetch_factor=prefetch,
            pin_memory=pin,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    return (
        make_loader(train_idx, True),
        make_loader(val_idx, False),
        make_loader(test_idx, False),
    )
