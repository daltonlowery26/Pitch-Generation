# package
import polars as pl
import torch

from src.embeddings.pitch.modules.cvae import CVAE
from src.utils.config import load_config


# encode each pitch, then decode from its latent as the generative step
def get_embeddings(model, data, device):
    model.to(device)
    model.eval()
    embeddings_list = []
    generated_list = []
    chunk_size = 65536

    with torch.no_grad():
        for i in range(0, len(data), chunk_size):
            batch = data[i : i + chunk_size]
            traits = batch[..., :10]
            pitcher = batch[..., 10].long()
            pitch = batch[..., 11].long()

            # encode, latent mean is the embedding
            mu_q, _ = model.encode(traits)

            # decode that latent under the pitch conditional
            cond = model.cond_proj(
                torch.concat(
                    [model.pitcherEmbed(pitcher), model.pitchEmbed(pitch)], dim=-1
                )
            )
            gen = model.decode(mu_q, cond)

            embeddings_list.append(mu_q.cpu())
            generated_list.append(gen.cpu())

    return torch.cat(embeddings_list).numpy(), torch.cat(generated_list).numpy()


if __name__ == "__main__":
    # full set
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
        "pitch_group",
    ]
    input = pl.read_parquet("data/source/pitch_2015_2026.parquet")
    input = input.drop_nulls(subset=features)

    pitchers = input["pitcher_year"].n_unique()
    pitches = input["pitch_group"].n_unique()

    # normalize
    input = input.with_columns(
        kmu=pl.concat_list(
            [(pl.col(f) - pl.col(f).mean()) / pl.col(f).std() for f in features[:10]]
            + [pl.col(f) for f in features[10:]]
        )
    )

    cfg = load_config("cvae")
    arch = cfg["arch"]
    model = CVAE(
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
    sd = torch.load("weights/cvae_map.pth", map_location=torch.device("cpu"))
    sd = sd.get("model_state", sd)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)

    fullData = torch.tensor(input["kmu"].to_list(), device="cuda")
    embed, gen = get_embeddings(model, fullData, "cuda")

    embedPL = pl.DataFrame({"embed": list(embed), "genPitch": list(gen)})
    input = pl.concat([input, embedPL], how="horizontal")
    input.write_parquet("data/pitch/pitch_map.parquet")
