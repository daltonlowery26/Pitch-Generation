import numpy as np
import polars as pl
import torch
import yaml

from src.embeddings.expectation.modules.transformer import BatterExp
from src.embeddings.expectation.modules.utils import inferForward
from src.embeddings.pitch.modules.cvae import CVAE
from src.outcome.modules.seq_transformer import Transformer
from src.sequencing.module.pitch_value import pitch_value

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def load_yaml(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_cvae(cfg, n_pitchers, n_pitches):
    arch = cfg["arch"]
    model = CVAE(
        input_dim=arch["input_dim"],
        pitch_dim=arch["pitch_dim"],
        pitcher_dim=arch["pitcher_dim"],
        h_dim=arch["h_dim"],
        latent_dim=arch["latent_dim"],
        dropout=arch["dropout"],
        n_layers=arch["n_layers"],
        n_pitchers=n_pitchers,
        n_pitches=n_pitches,
    )
    state = torch.load(cfg["weight_path"], map_location="cpu")
    state = state.get("model_state", state)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_expectation(cfg, n_pitcher_seasons, n_batter_seasons, n_pitch_groups, max_len):
    arch = cfg["arch"]
    model = BatterExp(
        dim=arch["dim"],
        out_dim=n_pitch_groups,
        nheads=arch["nheads"],
        layers=arch["layers"],
        dropout=arch["dropout"],
        max_len=max_len,
        horizons=tuple(arch["horizons"]),
        pitcher_dim=arch["pitcher_dim"],
        batter_dim=arch["batter_dim"],
        embed_dim=arch["embed_dim"],
        batter_seasons=n_batter_seasons,
        pitcher_seasons=n_pitcher_seasons,
    )
    state = torch.load(cfg["weight_path"], map_location="cpu")
    state = state.get("model_state", state)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_outcome(cfg, output_dim, n_pitcher_seasons, n_batter_seasons):
    arch = cfg["arch"]
    model = Transformer(
        dim=arch["dim"],
        layers=arch["layers"],
        dropout=arch["dropout"],
        nhead=arch["nhead"],
        output_dim=output_dim,
        countdim=arch["countdim"],
        abdim=arch["abdim"],
        embed_dim=arch["embed_dim"],
        exp_dim=arch["exp_dim"],
        exp_proj_dim=arch["exp_proj_dim"],
        pitcher_dim=arch["pitcher_dim"],
        batter_dim=arch["batter_dim"],
        n_pitcher_seasons=n_pitcher_seasons,
        n_batter_seasons=n_batter_seasons,
    )
    state = torch.load(cfg["weight_path"], map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def lookup_player(df, name, year, name_col, id_col):
    matches = df.filter((pl.col(name_col) == name) & (pl.col("game_year") == year))
    if matches.is_empty():
        names = df[name_col].unique().sort().to_list()
        names_str = ", ".join(str(n) for n in names[:20])
        more = "..." if len(names) > 20 else ""
        raise ValueError(
            f"No player '{name}' found in year {year}. Available: {names_str}{more}"
        )
    return matches[id_col][0]


def get_pitcher_repertoire(df, pitcher_id_year):
    return (
        df.filter(pl.col("pitcher_year") == pitcher_id_year)["pitch_group"]
        .unique()
        .sort()
        .to_list()
    )


def parse_count(count_str):
    balls, strikes = (int(x) for x in count_str.split("-"))
    if balls < 0 or balls > 3 or strikes < 0 or strikes > 2:
        raise ValueError(f"Invalid count: {count_str}. Balls 0-3, strikes 0-2.")
    count_arr = balls * 3 + strikes
    return count_str, count_arr


def get_prior_sequence(
    df, pitcher_id_year, batter_id_year, prior_seq_len, current_count_str
):
    matchup = df.filter(
        (pl.col("pitcher_year") == pitcher_id_year)
        & (pl.col("batter_year") == batter_id_year)
    ).sort(["game_date", "atBatNumber", "pitch_number"], descending=[True, True, True])

    if matchup.is_empty():
        return None

    pitch_batter = matchup["pitchBatter"][0]
    seq = df.filter(pl.col("pitchBatter") == pitch_batter).sort(
        ["atBatNumber", "pitch_number"]
    )

    if prior_seq_len > 0 and prior_seq_len < len(seq):
        seq = seq[-prior_seq_len:]

    pitch_emb = np.stack(seq["genPitch"].to_list()).astype(np.float32)
    counts = seq["countArr"].to_numpy().astype(np.int32)
    atbats = seq["atBatNumber"].to_numpy().astype(np.int32)

    c_str, c_arr = parse_count(current_count_str)

    return {
        "pitchEmb": pitch_emb,
        "count": counts,
        "atbat": atbats,
        "current_count": c_arr,
        "current_count_str": c_str,
        "seq_len": len(seq),
    }


def generate_cvae_samples(cvae, pitcher_id, pitch_type, n, temperature):
    pitcher_emb = cvae.pitcherEmbed(torch.tensor([pitcher_id], device=device))
    pitch_emb = cvae.pitchEmbed(torch.tensor([pitch_type], device=device))
    c = torch.cat([pitcher_emb, pitch_emb], dim=-1)
    return cvae.generate(c, n=n, temperature=temperature)


def compute_expectation(batter_exp, prior, rep_mask, pitcher_id_val, batter_id_val):
    if prior is None or prior["seq_len"] == 0:
        pitcher_id = torch.tensor([pitcher_id_val], device=device)
        batter_id = torch.tensor([batter_id_val], device=device)
        prior_logits = batter_exp.get_prior(pitcher_id, batter_id, rep_mask)
        prior_probs = torch.softmax(prior_logits, dim=-1)
        return None, prior_probs.squeeze(0)

    T_prior = prior["seq_len"]

    embed = torch.from_numpy(prior["pitchEmb"]).unsqueeze(0).to(device)
    ab_num = torch.from_numpy(prior["atbat"]).unsqueeze(0).to(device)
    count = torch.from_numpy(prior["count"]).unsqueeze(0).to(device)
    pitcher_id = torch.full((1, T_prior), prior["pitcher_id"], device=device)
    batter_id = torch.full((1, T_prior), prior["batter_id"], device=device)
    lengths = torch.tensor([T_prior], device=device)

    batch = {
        "embeds": embed,
        "abNum": ab_num,
        "count": count,
        "pitcherID": pitcher_id,
        "batterID": batter_id,
        "lengths": lengths,
    }

    _, _, probs, _, prior_probs = inferForward(
        batter_exp, batch, rep_mask, device=device
    )

    if T_prior == 1:
        prior_exp = prior_probs[0].unsqueeze(0)
        candidate_exp = prior_probs[0]
    else:
        prior_exp = torch.cat(
            [prior_probs[0].unsqueeze(0), probs[0, : T_prior - 1]], dim=0
        )
        candidate_exp = probs[0, T_prior - 1]

    return prior_exp, candidate_exp


def build_outcome_batch(
    prior,
    candidates,
    prior_exp,
    candidate_exp,
    pitcher_id_val,
    batter_id_val,
    current_count_arr,
):
    B = candidates.shape[0]

    if prior is None or prior["seq_len"] == 0:
        pitch_emb = candidates.unsqueeze(1)
        _, C, _D = pitch_emb.shape
    else:
        prior_emb = torch.from_numpy(prior["pitchEmb"]).to(device)
        pitch_emb_list = []
        for i in range(B):
            seq = torch.cat([prior_emb, candidates[i].unsqueeze(0)], dim=0)
            pitch_emb_list.append(seq)
        pitch_emb = torch.stack(pitch_emb_list)
        _, C, _D = pitch_emb.shape

    if prior_exp is not None:
        prior_exp_dev = prior_exp.to(device)
        exp_list = [
            torch.cat([prior_exp_dev, candidate_exp.unsqueeze(0)], dim=0)
            for _ in range(B)
        ]
        exp_emb = torch.stack(exp_list)
    else:
        exp_emb = candidate_exp.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

    if prior is None or prior["seq_len"] == 0:
        count = torch.full((B, 1), current_count_arr, device=device)
        atbat = torch.full((B, 1), 0, device=device)
    else:
        prior_count = torch.from_numpy(prior["count"]).to(device)
        count_list = [
            torch.cat([prior_count, torch.tensor([current_count_arr], device=device)])
            for _ in range(B)
        ]
        count = torch.stack(count_list)
        prior_atbat = torch.from_numpy(prior["atbat"]).to(device)
        atbat_last = prior_atbat[-1]
        atbat_list = [
            torch.cat([prior_atbat, torch.tensor([atbat_last], device=device)])
            for _ in range(B)
        ]
        atbat = torch.stack(atbat_list)

    pitcher_id = torch.full((B, C), pitcher_id_val, device=device, dtype=torch.long)
    batter_id = torch.full((B, C), batter_id_val, device=device, dtype=torch.long)
    lengths = torch.full((B,), C, device=device, dtype=torch.long)

    return pitch_emb, exp_emb, pitcher_id, batter_id, atbat, count, lengths


def run_outcome_models(swing_model, whiff_model, foul_model, bip_model, batch):
    pitch_emb, exp_emb, pitcher_id, batter_id, atbat, count, lengths = batch

    with torch.no_grad():
        swing_logits = swing_model(
            pitch_emb, exp_emb, pitcher_id, batter_id, atbat, count, lengths
        )
        swing_prob = torch.sigmoid(swing_logits[:, -1])

        whiff_logits = whiff_model(
            pitch_emb, exp_emb, pitcher_id, batter_id, atbat, count, lengths
        )
        whiff_prob = torch.sigmoid(whiff_logits[:, -1])

        foul_logits = foul_model(
            pitch_emb, exp_emb, pitcher_id, batter_id, atbat, count, lengths
        )
        foul_prob = torch.sigmoid(foul_logits[:, -1])

        bip_logits = bip_model(
            pitch_emb, exp_emb, pitcher_id, batter_id, atbat, count, lengths
        )
        bip_probs = torch.softmax(bip_logits[:, -1], dim=-1)

    return swing_prob, whiff_prob, foul_prob, bip_probs


def compute_run_values(
    swing_prob,
    whiff_prob,
    foul_prob,
    bip_probs,
    count_str,
    game_year,
    hbp_prob,
    strike_prob,
):
    n = swing_prob.shape[0]
    rv = np.empty(n, dtype=np.float32)

    bip_probs_np = bip_probs.cpu().numpy()

    for i in range(n):
        rv[i] = pitch_value(
            count=count_str,
            game_year=game_year,
            whiff_prob=whiff_prob[i].item(),
            foul_prob=foul_prob[i].item(),
            swing_prob=swing_prob[i].item(),
            fb_prob=bip_probs_np[i, 0],
            gb_prob=bip_probs_np[i, 1],
            ld_prob=bip_probs_np[i, 2],
            popup_prob=bip_probs_np[i, 3],
            hbp_prob=hbp_prob,
            strike_prob=strike_prob,
        )

    return rv
