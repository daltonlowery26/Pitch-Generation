import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.manifold import TSNE

from src.embeddings.pitch.modules.cvae import CVAE

PITCH_TYPES = {
    0: {"name": "Fastball", "color": "#e74c3c"},
    1: {"name": "Breaking", "color": "#3498db"},
    2: {"name": "Offspeed", "color": "#2ecc71"},
    3: {"name": "Other", "color": "#7f8c8d"},
}


def plot(data, model, n_samples_per_type = 50):
    pitcher_counts = (
        data.group_by("pitcher_year")
        .agg(pl.n_unique("pitch_group").alias("n_types"), pl.len().alias("n_pitches"))
        .filter(pl.col("n_types") >= 3, pl.col("n_pitches") >= 200)
        .sort("n_pitches", descending=True)
    )

    if len(pitcher_counts) == 0:
        pitcher_counts = (
            data
            .filter(pl.col('pitcher_year') != 0)
            .group_by("pitcher_year")
            .agg(
                pl.n_unique("pitch_group").alias("n_types"),
                pl.len().alias("n_pitches"),
            )
            .filter(pl.col("n_types") >= 2, pl.col("n_pitches") >= 50)
            .sort("n_pitches", descending=True)
        )

    if len(pitcher_counts) == 0:
        target_pitcher = int(data["pitcher_year"].unique().item(0))
        print(f"falling back to pitcher {target_pitcher}")
    else:
        target_pitcher = int(pitcher_counts.item(1, "pitcher_year"))
        print(f"using pitcher {target_pitcher}", flush=True)
        print(pitcher_counts.head(5))

    pitcher_data = data.filter(pl.col("pitcher_year") == target_pitcher).sample(n=250)
    pitcher_types = sorted(pitcher_data["pitch_group"].unique().to_list())

    real_traits = np.array(pitcher_data["kmu"].to_list())
    real_traits_10 = real_traits[:, :10]
    real_labels = pitcher_data["pitch_group"].to_numpy()

    all_gen_traits = []
    all_gen_labels = []

    for pg in pitcher_types:
        pitcher_embed = model.pitcherEmbed(torch.tensor([target_pitcher]))
        pitch_embed = model.pitchEmbed(torch.tensor([pg]))
        c = torch.cat([pitcher_embed, pitch_embed], dim=-1)
        generated = model.generate(c, n=n_samples_per_type).cpu().numpy()
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

    fig, ax = plt.subplots(figsize=(12, 10))

    for pg in pitcher_types:
        info = PITCH_TYPES[pg]
        color = info["color"]
        name = info["name"]

        mask = real_labels == pg
        if mask.sum() > 0:
            ax.scatter(
                real_embed[mask, 0],
                real_embed[mask, 1],
                c=color,
                marker="o",
                label=f"{name} (real, n={mask.sum()})",
                alpha=0.7,
                s=30,
                edgecolors="white",
                linewidth=0.3,
            )

        mask = gen_labels == pg
        ax.scatter(
            gen_embed[mask, 0],
            gen_embed[mask, 1],
            c=color,
            marker="^",
            label=f"{name} (generated, n={mask.sum()})",
            alpha=0.5,
            s=40,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_title(f"TSNE: Real vs Generated Pitches — Pitcher {target_pitcher}", fontsize=14)
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
        "pitch_group",
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
    pitches = data["pitch_group"].n_unique()

    model = CVAE(
        input_dim=10,
        pitch_dim=24,
        pitcher_dim=24,
        h_dim=128,
        latent_dim=6,
        dropout=0.1,
        n_layers=5,
        n_pitchers=pitchers,
        n_pitches=pitches,
    )

    sd = torch.load('weights/cvae_map.pth', map_location=torch.device("cpu"))
    sd = sd.get("model_state", sd)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    plot(data, model)
