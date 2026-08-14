from datetime import datetime

import numpy as np
import polars as pl
import torch

from src.embeddings.pitch.modules.cvae import CVAE
from src.evaluate.modules.data import generate_fakes
from src.utils.modeling.config import load_config

device = "cuda" if torch.cuda.is_available() else "mps"


def _as_tensor(values):
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def _blockwise_mean_distance(left, right, block_size):
    total = 0.0
    for start in range(0, len(left), block_size):
        distances = torch.cdist(left[start : start + block_size], right)
        total += distances.sum().item()
    return total / (len(left) * len(right))


def _blockwise_mean_rbf(left, right, bandwidth, block_size):
    total = 0.0
    denominator = 2.0 * bandwidth * bandwidth
    for start in range(0, len(left), block_size):
        distances = torch.cdist(left[start : start + block_size], right)
        total += torch.exp(-(distances.square()) / denominator).sum().item()
    return total / (len(left) * len(right))


def rbf_mmd(observed, generated, bandwidth, block_size=1024):
    observed = _as_tensor(observed)
    generated = _as_tensor(generated)
    within_observed = _blockwise_mean_rbf(observed, observed, bandwidth, block_size)
    within_generated = _blockwise_mean_rbf(generated, generated, bandwidth, block_size)
    cross = _blockwise_mean_rbf(observed, generated, bandwidth, block_size)
    return max(within_observed + within_generated - 2.0 * cross, 0.0)


def energy_distance(observed, generated, block_size=1024):
    observed = _as_tensor(observed)
    generated = _as_tensor(generated)
    within_observed = _blockwise_mean_distance(observed, observed, block_size)
    within_generated = _blockwise_mean_distance(generated, generated, block_size)
    cross = _blockwise_mean_distance(observed, generated, block_size)
    return max(2.0 * cross - within_observed - within_generated, 0.0)


def estimate_bandwidth(reference, max_samples=4096, seed=0):
    reference = _as_tensor(reference)
    sample_count = min(len(reference), max_samples)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(reference), size=sample_count, replace=False)
    sample = reference[indices]
    return torch.pdist(sample).median().item()


def pitcher_distances(
    observed,
    generated,
    pitcher_year,
    pitcher_name,
    bandwidth,
    min_samples=100,
    max_samples=2048,
    seed=0,
    block_size=1024,
):
    observed = np.asarray(observed, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    pitcher_year = np.asarray(pitcher_year)
    pitcher_name = np.asarray(pitcher_name, dtype=object)

    rng = np.random.default_rng(seed)
    records = []
    for year in sorted(np.unique(pitcher_year).tolist()):
        group_indices = np.flatnonzero(pitcher_year == year)

        n_observed = len(group_indices)
        n_generated = len(group_indices)
        eligible = n_observed >= min_samples
        if not eligible:
            selected_indices = np.asarray([], dtype=np.intp)
        elif n_observed > max_samples:
            selected_indices = rng.choice(
                group_indices, size=max_samples, replace=False
            )
        else:
            selected_indices = group_indices

        if eligible:
            mmd_rbf = rbf_mmd(
                observed[selected_indices],
                generated[selected_indices],
                bandwidth,
                block_size=block_size,
            )
            group_energy_distance = energy_distance(
                observed[selected_indices],
                generated[selected_indices],
                block_size=block_size,
            )
            n_used = len(selected_indices)
        else:
            mmd_rbf = None
            group_energy_distance = None
            n_used = 0

        records.append(
            {
                "pitcher_year": year,
                "pitcher_name": pitcher_name[group_indices[0]],
                "n_observed": n_observed,
                "n_generated": n_generated,
                "n_used": n_used,
                "mmd_rbf": mmd_rbf,
                "energy_distance": group_energy_distance,
                "eligible": eligible,
            }
        )

    return pl.DataFrame(
        records,
        schema={
            "pitcher_year": pl.Int64,
            "pitcher_name": pl.String,
            "n_observed": pl.Int64,
            "n_generated": pl.Int64,
            "n_used": pl.Int64,
            "mmd_rbf": pl.Float64,
            "energy_distance": pl.Float64,
            "eligible": pl.Boolean,
        },
    )


def summarize_distances(metrics):
    eligible = metrics.filter(pl.col("eligible"))
    summary = {}

    for metric, macro_key, weighted_key in (
        ("mmd_rbf", "macro_mmd_rbf", "weighted_mmd_rbf"),
        ("energy_distance", "macro_energy_distance", "weighted_energy_distance"),
    ):
        scores = eligible[metric].to_list()
        counts = eligible["n_observed"].to_list()
        summary[macro_key] = sum(scores) / len(scores)
        summary[weighted_key] = sum(
            score * count for score, count in zip(scores, counts)
        ) / sum(counts)

    return summary


if __name__ == "__main__":
    cfg = load_config("config/test/eval_embed.yml")
    cvae_cfg = load_config("config/embed/cvae.yml")

    df = pl.read_parquet(cfg["data_path"])
    features = [
        "hra",
        "vra",
        "release_height",
        "release_x",
        "ay",
        "az",
        "ax",
        "vx0",
        "vy0",
        "vz0",
        "pitcher_id_year",
        "pitch_name",
    ]
    df = df.drop_nulls(subset=features)

    pitchers = df["pitcher_year"].n_unique()
    pitches = df["pitch_name"].n_unique()
    arch = cvae_cfg["arch"]
    cvae = CVAE(
        input_dim=arch["input_dim"],
        pitch_dim=arch["pitch_dim"],
        pitcher_dim=arch["pitcher_dim"],
        h_dim=arch["h_dim"],
        latent_dim=arch["latent_dim"],
        dropout=arch["dropout"],
        n_layers=arch["n_layers"],
        n_pitchers=pitchers,
        n_pitches=pitches,
    )
    state = torch.load(cfg["weight_path"], map_location=torch.device("cpu"))
    state = state.get("model_state", state)
    state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    cvae.load_state_dict(state)

    cutoff = datetime(2025, 7, 1).date()
    calibration = df.filter(pl.col("game_date") < cutoff)
    heldout = df.filter(pl.col("game_date") >= cutoff)
    reference = np.asarray(calibration["kmu"].to_list(), dtype=np.float32)[:, :10]
    observed = np.asarray(heldout["kmu"].to_list(), dtype=np.float32)[:, :10]
    generated = generate_fakes(heldout, cvae, cfg["temperature"], device)

    bandwidth = estimate_bandwidth(reference)

    metrics = pitcher_distances(
        observed,
        generated,
        heldout["pitcher_year"].to_numpy(),
        heldout["pitcher_name"].to_numpy(),
        bandwidth=bandwidth,
    )
    summary = summarize_distances(metrics)

    print(metrics)
    for key, value in summary.items():
        print(f"{key}={value:.6f}")
