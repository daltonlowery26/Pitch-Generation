import polars as pl

from src.embeddings.expectation.modules.utils import repertoire_mask
from src.sequencing.module.pitch_value import _load_data
from src.sequencing.module.utils import (
    build_outcome_batch,
    compute_expectation,
    compute_run_values,
    device,
    generate_cvae_samples,
    get_pitcher_repertoire,
    get_prior_sequence,
    load_cvae,
    load_expectation,
    load_outcome,
    lookup_player,
    parse_count,
    run_outcome_models,
)
from src.utils.modeling.config import load_config


def optimal_pitch(
    pitch_map,
    pitcher,
    pitcher_year,
    batter,
    batter_year,
    count,
    prior_seq_len=5,
    n_samples=None,
):
    inf_cfg = load_config("pitch_sequencing")["inference"]
    if n_samples is None:
        n_samples = inf_cfg["n_cvae_samples"]
    temperature = inf_cfg["temperature"]
    hbp_prob = inf_cfg["hbp_prob"]
    strike_prob = inf_cfg["strike_prob"]
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

    cvae = load_cvae(load_config("cvae"), n_pitcher_seasons, n_pitches_cvae)
    batter_exp = load_expectation(
        load_config("expectation"),
        n_pitcher_seasons,
        n_batter_seasons,
        n_pitches,
        max_len,
    )
    swing_model = load_outcome(
        load_config("swing_prob"), 1, n_pitcher_seasons, n_batter_seasons
    )
    whiff_model = load_outcome(
        load_config("whiff_prob"), 1, n_pitcher_seasons, n_batter_seasons
    )
    foul_model = load_outcome(
        load_config("foul_prob"), 1, n_pitcher_seasons, n_batter_seasons
    )
    bip_model = load_outcome(
        load_config("bip_type"), 5, n_pitcher_seasons, n_batter_seasons
    )

    _load_data()

    count_str, count_arr = parse_count(count)

    prior = get_prior_sequence(
        df, pitcher_id_year, batter_id_year, prior_seq_len, count
    )

    if prior is not None:
        prior["pitcher_id"] = pitcher_id_year
        prior["batter_id"] = batter_id_year

    prior_exp, candidate_exp = compute_expectation(
        batter_exp, prior, rep_mask, pitcher_id_year, batter_id_year
    )

    results = {}
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
        run_values = compute_run_values(
            swing_prob,
            whiff_prob,
            foul_prob,
            bip_probs,
            count_str,
            game_year,
            hbp_prob,
            strike_prob,
        )
        results[pitch_type] = float(run_values.mean())

    return results


if __name__ == "__main__":
    PITCH_MAP = "data/pitch/pitch_exp.parquet"
    PITCHER = "Warren, Will"
    PITCHER_YEAR = 2025
    BATTER = "Harper, Bryce"
    BATTER_YEAR = 2025
    COUNT = "1-0"
    PRIOR_SEQ = 0
    N_SAMPLES = None

    results = optimal_pitch(
        pitch_map=PITCH_MAP,
        pitcher=PITCHER,
        pitcher_year=PITCHER_YEAR,
        batter=BATTER,
        batter_year=BATTER_YEAR,
        count=COUNT,
        prior_seq_len=PRIOR_SEQ,
        n_samples=N_SAMPLES,
    )

    df = pl.read_parquet(PITCH_MAP)
    pitch_name_to_str = dict(
        df.select(["pitch_name", "pitch_str"]).unique().rows()
    )

    print(f"Pitcher: {PITCHER} ({PITCHER_YEAR})")
    print(f"Batter:  {BATTER} ({BATTER_YEAR})")
    print(f"Count:   {COUNT}")
    print(f"{'Pitch Type':<12} {'Avg Run Value':>14}")

    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for pitch_type, rv in sorted_results:
        name = pitch_name_to_str.get(pitch_type, str(pitch_type))
        print(f"  {name:<16} {rv:>14.4f}")

    best = sorted_results[0]
    best_name = pitch_name_to_str.get(best[0], str(best[0]))
    print(f"\n  Best: {best_name} ({best[1]:.4f})")
