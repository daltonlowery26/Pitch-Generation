import polars as pl
import torch

from src.embeddings.expectation.modules.utils import repertoire_mask
from src.sequencing.module.pitch_value import _load_data
from src.sequencing.module.utils import (
    build_outcome_batch,
    compute_expectation,
    compute_run_values,
    device,
    generate_cvae_samples,
    get_pitcher_repertoire,
    load_cvae,
    load_expectation,
    load_outcome,
    load_strike_prob,
    lookup_player,
    run_outcome_models,
    sample_prior_pitches,
)
from src.utils.modeling.config import load_config


def optimal_pitch(
    pitch_map,
    pitcher,
    pitcher_year,
    batter,
    batter_year,
    prior_sequence=None,
):
    inf_cfg = load_config("config/test/pitch_sequencing.yml")["inference"]
    n_samples = inf_cfg["n_cvae_samples"]
    temperature = inf_cfg["temperature"]
    hbp_prob = inf_cfg["hbp_prob"]
    game_year = inf_cfg["game_year"]

    df = pl.read_parquet(pitch_map)

    n_pitcher_seasons = df["pitcher_year"].n_unique()
    n_batter_seasons = df["batter_year"].n_unique()
    n_pitches = df["pitch_name"].n_unique()
    n_pitches_cvae = df["pitch_name"].n_unique()
    max_len = df.group_by("pitchBatter").len().get_column("len").max()

    pitcher_id_year = lookup_player(
        df, pitcher, pitcher_year, "pitcher_name", "pitcher_year"
    )
    batter_id_year = lookup_player(
        df, batter, batter_year, "batter_name", "batter_year"
    )

    repertoire = get_pitcher_repertoire(df, pitcher_id_year)

    if not repertoire:
        raise ValueError(
            f"Pitcher '{pitcher}' ({pitcher_year}) has no pitch repertoire."
        )

    rep_mask = repertoire_mask(df, n_pitcher_seasons, n_pitches).to(device)

    cvae = load_cvae(
        load_config("config/embed/cvae.yml"), n_pitcher_seasons, n_pitches_cvae
    )
    batter_exp = load_expectation(
        load_config("config/embed/expectation.yml"),
        n_pitcher_seasons,
        n_batter_seasons,
        n_pitches,
        max_len,
    )
    swing_model = load_outcome(
        load_config("config/prob/swing_prob.yml"),
        1,
        n_pitcher_seasons,
        n_batter_seasons,
    )
    whiff_model = load_outcome(
        load_config("config/prob/whiff_prob.yml"),
        1,
        n_pitcher_seasons,
        n_batter_seasons,
    )
    foul_model = load_outcome(
        load_config("config/prob/foul_prob.yml"), 1, n_pitcher_seasons, n_batter_seasons
    )
    bip_model = load_outcome(
        load_config("config/prob/bip_type.yml"), 5, n_pitcher_seasons, n_batter_seasons
    )
    strike_model = load_strike_prob(load_config("config/prob/strike_prob.yml"))

    _load_data()

    prior = None
    prior_info = None
    count_str = "0-0"
    count_arr = 0

    if prior_sequence:
        prior, count_str, count_arr = sample_prior_pitches(
            df, strike_model, prior_sequence
        )
        prior["pitcher_id"] = pitcher_id_year
        prior["batter_id"] = batter_id_year
        prior_info = {
            "pitch_type": prior["pitch_type"],
            "description": prior["description"],
            "counts": [
                dict(zip(["balls", "strikes"], divmod(c, 3)))
                for c in prior["count"].tolist()
            ],
        }

    prior_exp, candidate_exp = compute_expectation(
        batter_exp, prior, rep_mask, pitcher_id_year, batter_id_year
    )

    results = {}
    probs = {}
    for pitch_type in repertoire:
        candidates = generate_cvae_samples(
            cvae, pitcher_id_year, pitch_type, n_samples, temperature
        )
        batch = build_outcome_batch(
            prior,
            candidates,
            prior_exp,
            candidate_exp,
            pitcher_id_year,
            batter_id_year,
            count_arr,
        )
        swing_prob, whiff_prob, foul_prob, bip_probs = run_outcome_models(
            swing_model, whiff_model, foul_model, bip_model, batch
        )
        with torch.no_grad():
            strike_logits = strike_model(candidates).squeeze(-1)
            strike_probs = 1.0 - torch.sigmoid(strike_logits)
        run_values = compute_run_values(
            swing_prob,
            whiff_prob,
            foul_prob,
            bip_probs,
            count_str,
            game_year,
            hbp_prob,
            strike_probs,
        )
        results[pitch_type] = float(run_values.mean())
        probs[pitch_type] = {
            "swing": float(swing_prob.mean()),
            "whiff": float(whiff_prob.mean()),
            "foul": float(foul_prob.mean()),
            "strike": float(strike_probs.mean()),
            "bip": [float(b) for b in bip_probs.mean(dim=0)],
        }

    return results, prior_info, count_str, probs


def build_prior_sequence_tui(df, pitcher, pitcher_year, batter, batter_year):
    pitcher_id_year = lookup_player(
        df, pitcher, pitcher_year, "pitcher_name", "pitcher_year"
    )
    repertoire = get_pitcher_repertoire(df, pitcher_id_year)
    pitch_name_to_str = dict(
        df.select(["pitch_name", "pitch_type"]).unique().rows()
    )

    numbered_repertoire = {}
    for i, name in enumerate(sorted(repertoire), 1):
        numbered_repertoire[i] = name

    print(f"\nPitcher: {pitcher} ({pitcher_year})")
    print(f"Batter:  {batter} ({batter_year})\n")
    print("Repertoire:")
    for num, name in numbered_repertoire.items():
        label = pitch_name_to_str.get(name, str(name))
        print(f"  {num}. {label}")
    print()

    sequence = []
    balls = 0
    strikes = 0

    # allow user to specify sequence
    while True:
        count_str = f"{balls}-{strikes}"
        print(f"Count: {count_str}  |  Pitches: {len(sequence)}")
        if sequence:
            items = ", ".join(
                f"{pitch_name_to_str.get(n, str(n))} \u2192 {r}"
                for n, r in sequence
            )
            print(f"  [{items}]")

        try:
            user_input = input(
                "\nEnter: <#> <ball|strike> or Enter to analyze: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            break

        parts = user_input.split()
        if len(parts) != 2:
            print("  Format: <number> <ball|strike>")
            continue

        try:
            num = int(parts[0])
        except ValueError:
            print("  Number must be an integer.")
            continue

        result = parts[1].lower()
        if result not in ("ball", "strike"):
            print("  Result must be 'ball' or 'strike'.")
            continue

        if num not in numbered_repertoire:
            print(f"  Number must be 1-{len(numbered_repertoire)}.")
            continue

        if result == "ball" and balls + 1 > 3:
            print("  Walk \u2014 at-bat ends here. Press Enter to analyze.\n")
            continue
        if result == "strike" and strikes + 1 > 2:
            print("  Strikeout \u2014 at-bat ends here. Press Enter to analyze.\n")
            continue

        pitch_name = numbered_repertoire[num]
        sequence.append((pitch_name, result))
        if result == "ball":
            balls += 1
        else:
            strikes += 1

        print()

    final_count = f"{balls}-{strikes}"
    if sequence:
        print(f"\n--- Analyzing at count {final_count} ---")
    else:
        print("Analyzing at count 0-0...")

    return sequence, final_count, pitch_name_to_str


if __name__ == "__main__":
    PITCH_MAP = "data/pitch/pitch_exp.parquet"
    PITCHER = "Warren, Will"
    PITCHER_YEAR = 2025
    BATTER = "Harper, Bryce"
    BATTER_YEAR = 2025

    df = pl.read_parquet(PITCH_MAP)

    sequence, final_count, pitch_name_to_str = build_prior_sequence_tui(
        df, PITCHER, PITCHER_YEAR, BATTER, BATTER_YEAR
    )

    results, prior_info, _, probs = optimal_pitch(
        pitch_map=PITCH_MAP,
        pitcher=PITCHER,
        pitcher_year=PITCHER_YEAR,
        batter=BATTER,
        batter_year=BATTER_YEAR,
        prior_sequence=sequence if sequence else None
    )

    if prior_info is not None and prior_info["pitch_type"]:
        print("\nSampled prior pitches:")
        for i, (p_str, desc, cnt) in enumerate(
            zip(
                prior_info["pitch_type"],
                prior_info["description"],
                prior_info["counts"],
            ),
            1,
        ):
            print(f"  {i}. {p_str:<16} \u2192 {desc} ({cnt['balls']}-{cnt['strikes']})")

    print(f"\nResults at {final_count}:")
    print(f"{'Pitch Type':<20} {'Avg Run Value':>14}")
    print("-" * 35)

    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for pitch_type, rv in sorted_results:
        name = pitch_name_to_str.get(pitch_type, str(pitch_type))
        print(f"  {name:<18} {rv:>14.4f}")

    best = sorted_results[0]
    best_name = pitch_name_to_str.get(best[0], str(best[0]))
    best_probs = probs[best[0]]
    print(f"\nBest: {best_name} ({best[1]:.4f})")
    print(f"  Swing: {best_probs['swing']:.4f}  |  Whiff: {best_probs['whiff']:.4f}  |  Foul: {best_probs['foul']:.4f}  |  Strike: {best_probs['strike']:.4f}")
    print(f"  BIP:   LD {best_probs['bip'][2]:.4f}  |  GB {best_probs['bip'][1]:.4f}  |  FB {best_probs['bip'][0]:.4f}  |  Popup {best_probs['bip'][3]:.4f}")
