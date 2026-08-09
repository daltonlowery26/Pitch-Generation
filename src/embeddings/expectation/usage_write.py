# %% batter expectation write: run inference over all splits, write parquet
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.embeddings.expectation.modules.data import UsageDataset, collate_fn
from src.embeddings.expectation.modules.transformer import BatterExp
from src.embeddings.expectation.modules.utils import inferForward, repertoire_mask
from src.embeddings.expectation.usage_embed import loadBatch
from src.utils.config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def write_expectation(data_path, weights_path, output_path, device=device):
    df = pl.read_parquet(data_path)

    # number of pitchers, batters, len
    pitchers = df["pitcher_year"].n_unique()
    batters = df["batter_year"].n_unique()
    pitches = df["pitch_name"].n_unique()
    max_len = df.group_by("pitchBatter").len().get_column("len").max()

    # usage dataset
    dataset = UsageDataset(df)
    mask = repertoire_mask(df, num_pitchers=pitchers, num_pitches=pitches).to(device)

    # load model
    cfg = load_config("expectation")
    arch = cfg["arch"]
    model = BatterExp(
        dim=arch["dim"],
        out_dim=pitches,
        nheads=arch["nheads"],
        layers=arch["layers"],
        attn_ffn_blocks=arch["attn_ffn_blocks"],
        dropout=arch["dropout"],
        max_len=max_len,
        horizons=tuple(arch["horizons"]),
        pitcher_dim=arch["pitcher_dim"],
        batter_dim=arch["batter_dim"],
        embed_dim=arch["embed_dim"],
        batter_seasons=batters,
        pitcher_seasons=pitchers,
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    # loop through batch and save
    all_dist = []
    with torch.inference_mode():
        loader = DataLoader(
            dataset,
            batch_size=4096,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn,
        )
        for feat in loader:
            batch = loadBatch(feat, device)
            _, _, probs, _, prior_probs = inferForward(
                model, batch, mask, device=device
            )
            lengths = batch["lengths"]
            for i in range(batch["embeds"].shape[0]):
                L = lengths[i].item()
                seq_probs = (
                    torch.cat([prior_probs[i : i + 1], probs[i, : L - 1]], dim=0)
                    .cpu()
                    .numpy()
                )
                all_dist.append(seq_probs)

    # to np array
    all_dist = np.concatenate(all_dist, axis=0)
    print("data written!")

    # assert len equivelence
    assert len(all_dist) == len(df)

    # write
    df = df.sort(["pitchBatter", "atBatNumber", "pitch_number"])
    df = df.with_columns(pl.Series("expEmb", all_dist.tolist()))
    df.write_parquet(output_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    write_expectation(
        data_path="data/pitch/pitch_map.parquet",
        weights_path="weights/usage.pth",
        output_path="data/pitch/pitch_exp.parquet",
    )
