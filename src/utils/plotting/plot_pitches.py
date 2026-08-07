import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.manifold import TSNE

from src.embeddings.pitch.modules.cvae import CVAE


def plot(data, model, n_samples_per_type=200, temperature=1.0):
    pitcher_counts = (
        data.group_by("pitcher_year")
        .agg(pl.n_unique("pitch_name").alias("n_types"), pl.len().alias("n_pitches"))
        .filter(pl.col("n_types") >= 3, pl.col("n_pitches") >= 200)
        .sort("n_pitches", descending=True)
    )

    if len(pitcher_counts) == 0:
        pitcher_counts = (
            data
            .filter(pl.col('pitcher_year') != 0)
            .group_by("pitcher_year")
            .agg(
                pl.n_unique("pitch_name").alias("n_types"),
                pl.len().alias("n_pitches"),
            )
            .filter(pl.col("n_types") >= 2, pl.col("n_pitches") >= 50)
            .sort("n_pitches", descending=True)
        )

    eligible = pitcher_counts["pitcher_year"].to_list() if len(pitcher_counts) > 0 else data["pitcher_year"].unique().to_list()
    target_pitcher = int(np.random.choice(eligible))

    pitcher_row = data.filter(pl.col("pitcher_year") == target_pitcher).head(1)
    pitcher_label = f"{pitcher_row['pitcher_name'].item(0)} ({pitcher_row['game_year'].item(0)})"

    if len(pitcher_counts) == 0:
        print(f"falling back to pitcher {pitcher_label}")
    else:
        print(f"using pitcher {pitcher_label}", flush=True)
        print(pitcher_counts.head(5))

    pitcher_data = data.filter(pl.col("pitcher_year") == target_pitcher)
    type_counts = pitcher_data.group_by("pitch_name").len()
    pitcher_types = sorted(type_counts.filter(pl.col("len") > 10)["pitch_name"].to_list())
    pitch_name_to_str = dict(data.select(["pitch_name", "pitch_str"]).unique().rows())

    real_traits = np.array(pitcher_data["kmu"].to_list())
    real_traits_10 = real_traits[:, :10]
    real_labels = pitcher_data["pitch_name"].to_numpy()

    all_gen_traits = []
    all_gen_labels = []

    for pg in pitcher_types:
        pitcher_embed = model.pitcherEmbed(torch.tensor([target_pitcher]))
        pitch_embed = model.pitchEmbed(torch.tensor([pg]))
        c = torch.cat([pitcher_embed, pitch_embed], dim=-1)
        generated = model.generate(c, n=n_samples_per_type, temperature=temperature).cpu().numpy()
        all_gen_traits.append(generated)
        all_gen_labels.extend([pg] * n_samples_per_type)

    gen_traits = np.concatenate(all_gen_traits, axis=0)
    gen_labels = np.array(all_gen_labels)

    combined = np.concatenate([real_traits_10, gen_traits], axis=0)
    print("fitting TSNE")

    reducer = TSNE(
        n_components=2,
        random_state=26,
    )
    embedding = reducer.fit_transform(combined)

    n_real = len(real_traits_10)
    real_embed = embedding[:n_real]
    gen_embed = embedding[n_real:]

    n_types = len(pitcher_types)
    cmap = plt.cm.tab10 if n_types <= 10 else plt.cm.tab20
    colors = [cmap(i / max(n_types - 1, 1)) for i in range(n_types)]

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, pg in enumerate(pitcher_types):
        color = colors[i]

        mask = real_labels == pg
        if mask.sum() > 0:
            ax.scatter(
                real_embed[mask, 0],
                real_embed[mask, 1],
                c=[color],
                marker="o",
                label=f"{pitch_name_to_str[pg]} (real, n={mask.sum()})",
                alpha=0.7,
                s=30,
                edgecolors="white",
                linewidth=0.3,
            )

        mask = gen_labels == pg
        ax.scatter(
            gen_embed[mask, 0],
            gen_embed[mask, 1],
            c=[color],
            marker="^",
            label=f"{pitch_name_to_str[pg]} (generated, n={mask.sum()})",
            alpha=0.5,
            s=40,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_title(f"TSNE: Real vs Generated Pitches — {pitcher_label}", fontsize=14)
    ax.set_xlabel("TSNE 1")
    ax.set_ylabel("TSNE 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9)
    fig.tight_layout()

    out_path = "data/plots/gen_pitch.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved to {out_path}")
    plt.show()


if __name__ == "__main__":
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
        "pitcher_year",
        "pitch_name",
    ]

    data = pl.read_parquet('data/cleaned/pitch_2015_2026.parquet').drop_nulls(subset=features)

    means = {f: data[f].mean() for f in features[:10]}
    stds = {f: data[f].std() for f in features[:10]}

    data = data.with_columns(
        kmu=pl.concat_list(
            [(pl.col(f) - means[f]) / stds[f] for f in features[:10]]
            + [pl.col(f) for f in features[10:]]
        )
    )

    pitchers = data["pitcher_year"].n_unique() 
    pitches = data["pitch_name"].n_unique()

    model = CVAE(
        input_dim=10,
        pitch_dim=16,
        pitcher_dim=16,
        h_dim=128,
        latent_dim=6,
        dropout=0.1,
        n_layers=4,
        n_pitchers=pitchers,
        n_pitches=pitches,
    )

    sd = torch.load('weights/cvae_map.pth', map_location=torch.device("cpu"))
    sd = sd.get("model_state", sd)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    plot(data, model, temperature=3.0)
