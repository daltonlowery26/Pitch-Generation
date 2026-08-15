from collections import Counter

import numpy as np
import torch

from src.sequencing.module.inference import SequencingEngine
from src.sequencing.module.utils import lookup_player


def simulate_optimal_usage(
    pitch_map,
    pitcher,
    season,
    n_batters=None,
    seed=26,
    n_cvae_samples=10,
):
    engine = SequencingEngine(pitch_map)
    pitcher_id = lookup_player(
        engine.df, pitcher, season, "pitcher_name", "pitcher_year"
    )
    target_rows = engine.df.filter(
        (engine.df["pitcher_name"] == pitcher) & (engine.df["game_year"] == season)
    )
    repertoire = target_rows["pitch_name"].unique().sort().to_list()

    opponent_pool = (
        engine.df.filter(
            (engine.df["pitcher_name"] == pitcher) & (engine.df["game_year"] == season)
        )
        .select("batter_name")
        .unique()
        .drop_nulls()
        .sort("batter_name")["batter_name"]
        .to_list()
    )

    n_batters = len(opponent_pool) if n_batters is None else int(n_batters)
    rng = np.random.default_rng(seed)
    torch_generator = torch.Generator(device="cpu").manual_seed(seed)
    sampled_batters = rng.choice(opponent_pool, size=n_batters, replace=True)

    pitch_counts = Counter({pitch_type: 0 for pitch_type in repertoire})
    outcome_counts = Counter(
        {
            "swing": 0,
            "take": 0,
            "ball": 0,
            "called_strike": 0,
            "whiff": 0,
            "foul": 0,
            "fb": 0,
            "ld": 0,
            "gb": 0,
            "popup": 0,
        }
    )
    terminal_counts = Counter(
        {"strikeout": 0, "walk": 0, "fb": 0, "ld": 0, "gb": 0, "popup": 0}
    )

    # loop through batters
    for batter in sampled_batters:
        batter = batter.item() if hasattr(batter, "item") else batter
        batter_id = lookup_player(
            engine.df, batter, season, "batter_name", "batter_year"
        )
        prior = None
        # num of atbats per batter
        for atbat in range(2):
            balls = 0
            strikes = 0
            while True:
                count_str = f"{balls}-{strikes}"
                count_arr = balls * 3 + strikes
                scored = {}
                for pitch_type in repertoire:
                    scored[pitch_type] = engine.score_pitch_candidates(
                        pitcher_id=pitcher_id,
                        batter_id=batter_id,
                        pitch_type=pitch_type,
                        prior=prior,
                        current_count_str=count_str,
                        current_count_arr=count_arr,
                        current_atbat=atbat,
                        season=season,
                        n_cvae_samples=n_cvae_samples,
                        generator=torch_generator,
                    )
                # take the minimum run value pitch
                selected_type = min(
                    repertoire,
                    key=lambda pitch_type: float(
                        np.asarray(scored[pitch_type]["run_values"]).mean()
                    ),
                )
                selected = scored[selected_type]
                candidate_index = int(rng.integers(n_cvae_samples))
                pitch_counts[selected_type] += 1
                prior = _append_history(
                    prior,
                    selected["candidates"][candidate_index],
                    count_arr,
                    atbat,
                )
                probabilities = selected["probs"]

                if rng.random() >= float(probabilities["swing"][candidate_index]):
                    outcome_counts["take"] += 1
                    if rng.random() < float(probabilities["ball"][candidate_index]):
                        outcome_counts["ball"] += 1
                        balls += 1
                        if balls == 4:
                            terminal_counts["walk"] += 1
                            break
                    else:
                        outcome_counts["called_strike"] += 1
                        strikes += 1
                        if strikes == 3:
                            terminal_counts["strikeout"] += 1
                            break
                    continue

                outcome_counts["swing"] += 1
                if rng.random() < float(probabilities["whiff"][candidate_index]):
                    outcome_counts["whiff"] += 1
                    strikes += 1
                    if strikes == 3:
                        terminal_counts["strikeout"] += 1
                        break
                    continue

                if rng.random() < float(probabilities["foul"][candidate_index]):
                    outcome_counts["foul"] += 1
                    if strikes < 2:
                        strikes += 1
                        if strikes == 3:
                            terminal_counts["strikeout"] += 1
                            break
                    continue

                bip = np.asarray(probabilities["bip"][candidate_index], dtype=float)
                bip = bip / bip.sum()
                bip_name = ("fb", "ld", "gb", "popup")[int(rng.choice(4, p=bip))]
                outcome_counts[bip_name] += 1
                terminal_counts[bip_name] += 1
                break

    n_pitches = sum(pitch_counts.values())
    n_at_bats = n_batters * 2
    pitch_shares = {
        pitch_type: count / n_pitches for pitch_type, count in pitch_counts.items()
    }
    terminal_rates = {
        outcome: count / n_at_bats for outcome, count in terminal_counts.items()
    }
    outcome_rates = {
        outcome: count / n_pitches for outcome, count in outcome_counts.items()
    }
    return {
        "pitcher": pitcher,
        "season": season,
        "pitch_counts": dict(pitch_counts),
        "pitch_shares": pitch_shares,
        "outcome_counts": dict(outcome_counts),
        "outcome_rates": outcome_rates,
        "terminal_counts": dict(terminal_counts),
        "terminal_rates": terminal_rates,
        "n_batters": n_batters,
        "n_at_bats": n_at_bats,
        "n_pitches": n_pitches,
    }


def _append_history(prior, candidate, count, atbat):
    embedding = candidate.detach().cpu().numpy()
    if prior is None:
        prior = {
            "pitchEmb": embedding[None, :],
            "count": np.array([count], dtype=np.int32),
            "atbat": np.array([atbat], dtype=np.int32),
            "seq_len": 1,
        }
        return prior

    prior["pitchEmb"] = np.concatenate((prior["pitchEmb"], embedding[None, :]))
    prior["count"] = np.append(prior["count"], count).astype(np.int32)
    prior["atbat"] = np.append(prior["atbat"], atbat).astype(np.int32)
    prior["seq_len"] += 1
    return prior
