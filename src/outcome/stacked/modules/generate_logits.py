from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from src.outcome.modules.data.outcome_data import (
    BIP_CLASSES,
    LenSampler,
    OutcomeProb,
    collate_fn,
)
from src.sequencing.module.utils import load_ball_prob, load_outcome
from src.utils.modeling.config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
output_dir = Path("data/outcome")


def flatten_sequence_output(logits, row_idx, lengths):
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    row_idx = row_idx.to(logits.device)
    valid = torch.arange(logits.shape[1], device=logits.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    return row_idx[valid], logits[valid]


def add_labels(df):
    return df.with_columns(
        pl.col("swing").cast(pl.Int8).alias("swing_label"),
        pl.col("description")
        .is_in(["ball", "blocked_ball", "intent_ball"])
        .cast(pl.Int8)
        .alias("ball_label"),
        pl.col("whiff").cast(pl.Int8).alias("whiff_label"),
        pl.col("foul").cast(pl.Int8).alias("foul_label"),
        *[
            pl.col("bipArr").list.get(i).cast(pl.Int8).alias(f"bip_label_{i}")
            for i in range(BIP_CLASSES)
        ],
    )


def assemble_base_probabilities(df, paths, output):
    labels = [
        "row_idx",
        "game_date",
        "swing_label",
        "ball_label",
        "whiff_label",
        "foul_label",
        *[f"bip_label_{i}" for i in range(BIP_CLASSES)],
    ]
    result = add_labels(df).with_row_index("row_idx").select(labels).lazy()
    for path in paths.values():
        result = result.join(pl.scan_parquet(path), on="row_idx")

    probability_columns = [
        f"{name}_prob_{i}" if name == "bip" else f"{name}_prob"
        for name in paths
        for i in (range(BIP_CLASSES) if name == "bip" else [0])
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    (
        result.sort("row_idx")
        .drop("row_idx")
        .with_columns(pl.col(probability_columns).cast(pl.Float32))
        .collect()
        .write_parquet(output)
    )


class PitchData(Dataset):
    def __init__(self, df):
        kmu = np.asarray(df["kmu"].to_list(), dtype=np.float32)
        self.pitch = torch.from_numpy(kmu[:, :10])

    def __len__(self):
        return len(self.pitch)

    def __getitem__(self, idx):
        return self.pitch[idx], idx


def write_batches(path, batches):
    writer = None
    try:
        for rows, values, columns in batches:
            values = values.detach().cpu().numpy().astype(np.float32)
            data = {"row_idx": rows.cpu().numpy()}
            if values.ndim == 1:
                data[columns[0]] = values
            else:
                data.update({column: values[:, i] for i, column in enumerate(columns)})
            table = pa.table(data)
            if writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def outcome_loader(df, batch_size=4096, workers=4):
    data = OutcomeProb(df.with_row_index("row_idx"))
    sampler = LenSampler(
        np.arange(len(data)),
        torch.diff(data.boundaries).numpy(),
        batch_size,
        250 * batch_size,
        shuffle=False,
        drop_last=False,
    )
    kwargs = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }
    if workers:
        kwargs["prefetch_factor"] = 4
    return DataLoader(data, **kwargs)


def write_outcome_probabilities(df, name, model, path):
    loader = outcome_loader(df)
    columns = (
        [f"bip_prob_{i}" for i in range(BIP_CLASSES)]
        if name == "bip"
        else [f"{name}_prob"]
    )

    def batches():
        model.eval()
        with torch.no_grad():
            for batch in loader:
                args = (
                    batch["pitchEmb"].to(device, non_blocking=True),
                    batch["expEmb"].to(device, non_blocking=True),
                    batch["pitcherID"].to(device, non_blocking=True),
                    batch["batterID"].to(device, non_blocking=True),
                    batch["abNum"].to(device, non_blocking=True),
                    batch["count"].to(device, non_blocking=True),
                    batch["lengths"].to(device, non_blocking=True),
                )
                rows, logits = flatten_sequence_output(
                    model(*args), batch["row_idx"], args[-1]
                )
                probs = (
                    torch.softmax(logits, -1)
                    if name == "bip"
                    else torch.sigmoid(logits)
                )
                yield rows, probs, columns

    write_batches(path, batches())


def write_ball_probabilities(df, path):
    loader = DataLoader(PitchData(df), batch_size=8192, shuffle=False)
    model = load_ball_prob(load_config("config/prob/ball_prob.yml"))

    def batches():
        with torch.no_grad():
            for pitch, rows in loader:
                logits = model(pitch.to(device, non_blocking=True))
                yield rows, torch.sigmoid(logits).squeeze(-1), ["ball_prob"]

    write_batches(path, batches())


def load_data():
    return pl.read_parquet("data/pitch/pitch_exp.parquet").with_columns(
        pl.col("kmu").list.slice(0, 10),
        pl.col("bipArr").list.slice(0, BIP_CLASSES),
    )


def generate_dataset():
    df = load_data()
    paths = {
        "swing": output_dir / "swing_probs.parquet",
        "ball": output_dir / "ball_probs.parquet",
        "whiff": output_dir / "whiff_probs.parquet",
        "foul": output_dir / "foul_probs.parquet",
        "bip": output_dir / "bip_probs.parquet",
    }

    write_ball_probabilities(df, paths["ball"])
    for name in ("swing", "whiff", "foul", "bip"):
        cfg = load_config(
            f"config/prob/{'bip_type' if name == 'bip' else name + '_prob'}.yml"
        )
        model = load_outcome(
            cfg,
            output_dim=BIP_CLASSES if name == "bip" else 1,
            n_pitcher_seasons=df["pitcher_year"].n_unique(),
            n_batter_seasons=df["batter_year"].n_unique(),
        )
        write_outcome_probabilities(df, name, model, paths[name])
        del model

    assemble_base_probabilities(df, paths, output_dir / "base_probs.parquet")


if __name__ == "__main__":
    generate_dataset()
