import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


def generate_fakes(df, cvae_model, temperature, device):
    kmu = np.asarray(df["kmu"].to_list(), dtype=np.float32)
    pitcher = kmu[:, 10].astype(np.int64)
    pitch = kmu[:, 11].astype(np.int64)

    cvae_model.to(device)
    cvae_model.eval()

    fakes = []
    chunk_size = 4096
    with torch.no_grad():
        for i in range(0, len(df), chunk_size):
            end = min(i + chunk_size, len(df))
            n = end - i
            z = temperature * torch.randn(n, cvae_model.latent_dim, device=device)
            pitcher_batch = torch.from_numpy(pitcher[i:end]).to(device)
            pitch_batch = torch.from_numpy(pitch[i:end]).to(device)

            cond = cvae_model.cond_proj(
                torch.cat(
                    [
                        cvae_model.pitcherEmbed(pitcher_batch),
                        cvae_model.pitchEmbed(pitch_batch),
                    ],
                    dim=-1,
                )
            )
            gen_traits = cvae_model.decode(z, cond)
            fakes.append(gen_traits.cpu())

    return torch.cat(fakes).numpy().astype(np.float32)


class DiscrimDataset(Dataset):
    def __init__(self, real, fake):
        self.embed = torch.from_numpy(np.concatenate([real, fake]))
        self.label = torch.from_numpy(
            np.concatenate([np.ones(len(real)), np.zeros(len(fake))]).astype(np.float32)
        )

    def __len__(self):
        return len(self.label)

    def __getitems__(self, idx):
        return {
            "embed": self.embed[idx],
            "label": self.label[idx].unsqueeze(-1),
        }


def split_loaders(
    df,
    cvae_model,
    temperature,
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
    test_subset = df.filter(pl.col("game_date") >= cutoff)

    train_real = np.asarray(train_subset["kmu"].to_list(), dtype=np.float32)[:, :10]
    val_real = np.asarray(val_subset["kmu"].to_list(), dtype=np.float32)[:, :10]
    test_real = np.asarray(test_subset["kmu"].to_list(), dtype=np.float32)[:, :10]

    train_fake = generate_fakes(train_subset, cvae_model, temperature, "cuda")
    val_fake = generate_fakes(val_subset, cvae_model, temperature, "cuda")
    test_fake = generate_fakes(test_subset, cvae_model, temperature, "cuda")

    train_ds = DiscrimDataset(train_real, train_fake)
    val_ds = DiscrimDataset(val_real, val_fake)
    test_ds = DiscrimDataset(test_real, test_fake)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        prefetch_factor=prefetch,
        drop_last=drop_last,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch,
        drop_last=False,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch,
        drop_last=False,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader
