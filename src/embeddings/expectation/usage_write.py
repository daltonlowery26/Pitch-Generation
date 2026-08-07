# %% batter expectation write: run inference over all splits, write parquet
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.embeddings.expectation.modules.data import UsageDataset, collate_fn
from src.embeddings.expectation.modules.transformer import BatterExp
from src.embeddings.expectation.modules.utils import inferForward, repertoire_mask
from src.embeddings.expectation.usage_embed import loadBatch

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


def write_expectation(data_path, weights_path, output_path, device=device):
    df = pl.read_parquet(data_path)

    # number of pitchers, batters, len
    pitchers = df["pitcher_year"].n_unique()
    batters = df["batter_year"].n_unique()
    pitches = df["pitch_group"].n_unique()
    max_len = df.group_by("pitchBatter").len().get_column("len").max()

    # usage dataset
    dataset = UsageDataset(df)
    mask = repertoire_mask(df, num_pitchers=pitchers, num_pitches=pitches).to(device)

    # load model
    model = BatterExp(
        dim=256,
        out_dim=pitches,
        nheads=8,
        layers=8,
        dropout=0.1,
        max_len=max_len,
        horizons=(1, 2, 3, 4),
        pitcher_dim=64,
        batter_dim=64,
        embed_dim=64,
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
            _, _, probs, _, prior_probs = inferForward(model, batch, mask, device=device)
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
        data_path="data/player/player.parquet",
        weights_path="weights/usage.pth",
        output_path="data/player/expectation.parquet",
    )
