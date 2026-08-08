import polars as pl

if __name__ == "__main__":
    df = (
        pl.scan_csv("data/raw/pitch_2015_2026.csv").select(
            [
                "game_year",
                "events",
                "description",
                "inning",
                "game_pk",
                "inning_topbot",
                "bb_type",
                "on_2b",
                "on_3b",
                "on_1b",
                "home_score",
                "away_score",
                "balls",
                "strikes",
                "outs_when_up",
                "at_bat_number",
                "pitch_number",
            ]
        )
    ).collect()
    df = df.sort(["game_year", "game_pk", "inning", "at_bat_number", "pitch_number"])

    # filter extras and invalid states
    df = df.filter(
        (
            (pl.col("inning") < 9)
            | ((pl.col("inning_topbot") != "bot") & (pl.col("inning") == 9))
        )
        & (pl.col("strikes") <= 2)
        & (pl.col("balls") <= 3)
    )

    # make counts and matrices
    df = df.with_columns(
        on_3b=pl.col("on_3b").is_not_null().cast(pl.Int8),
        on_2b=pl.col("on_2b").is_not_null().cast(pl.Int8),
        on_1b=pl.col("on_1b").is_not_null().cast(pl.Int8),
        strikeout=pl.col("events") == "strikeout",
        walk=pl.col("events") == "walk",
    )

    df = df.with_columns(
        endOfInning=(
            (pl.col("game_pk") != pl.col("game_pk").shift(1))
            | (pl.col("inning") != pl.col("inning").shift(1))
            | (pl.col("inning_topbot") != pl.col("inning_topbot").shift(1))
            | pl.col("game_pk").shift(1).is_null()
        )
    )
    df = df.with_columns(inning_group=pl.col("endOfInning").cast(pl.Int32).cum_sum())

    # batting-team pre-pitch score
    bat_score = (
        pl.when(pl.col("inning_topbot") == "Top")
        .then(pl.col("away_score"))
        .otherwise(pl.col("home_score"))
    )

    # next pitch's batting-team score, within same game
    next_bat_score = (
        pl.when(pl.col("inning_topbot") == "Top")
        .then(pl.col("away_score").shift(-1).over("game_pk"))
        .otherwise(pl.col("home_score").shift(-1).over("game_pk"))
    )

    # runs scored on this play
    df = df.with_columns(
        play_runs=(next_bat_score - bat_score).fill_null(0).cast(pl.Int32)
    )

    # total runs scored in the remainder of the half-inning
    df = df.with_columns(
        runs_remaining=(
            pl.col("play_runs").reverse().cum_sum().reverse().over("inning_group")
        )
    )

    # run_diff from this state to end of half
    df = df.with_columns(run_diff=pl.col("runs_remaining"))

    # count value
    countValues = df.group_by(["balls", "strikes", "game_year"]).agg(
        rv=pl.col("run_diff").mean()
    )
    countValues.write_csv("data/cleaned/run_exp/ball_strike_year.csv")

    # state value
    stateValue = df.group_by(
        ["outs_when_up", "on_3b", "on_1b", "on_2b", "game_year"]
    ).agg(rv=pl.col("run_diff").mean())
    stateValue.write_csv("data/cleaned/run_exp/out_state_year.csv")

    # event run value
    eventRV = df.select(
        [
            "on_3b",
            "on_1b",
            "on_2b",
            "bb_type",
            "walk",
            "strikeout",
            "outs_when_up",
            "game_year",
            "inning",
            "inning_topbot",
            "inning_group",
            "game_pk",
            "play_runs",
        ]
    )
    eventRV = eventRV.join(
        stateValue, on=["outs_when_up", "on_3b", "on_1b", "on_2b", "game_year"]
    )
    rv_next = pl.col("rv").shift(-1).over("inning_group").fill_null(0)

    eventRV = eventRV.with_columns(
        changeValue=pl.col("play_runs") + (rv_next - pl.col("rv"))
    )

    # bip_value
    bb_value = (
        eventRV.filter(
            pl.col("bb_type").is_in(["popup", "fly_ball", "ground_ball", "line_drive"])
        )
        .group_by(["game_year", "bb_type"])
        .agg(pl.col("changeValue").mean())
        .sort(["game_year", "bb_type"])
    )
    bb_value.write_csv("data/cleaned/run_exp/bipvalue.csv")

    # strikeout walk value
    bbK = eventRV.group_by(["game_year"]).agg(
        bb=pl.col("changeValue").filter(pl.col("walk")).mean(),
        k=pl.col("changeValue").filter(pl.col("strikeout")).mean(),
    )
    bbK.write_csv("data/cleaned/run_exp/bb_k_value.csv")

    # transation values
    pitch_trans = df.select(
        [
            "game_year",
            "balls",
            "strikes",
            "description",
            "inning_group",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ]
    )

    df = df.join(
        stateValue, on=["outs_when_up", "on_3b", "on_1b", "on_2b", "game_year"]
    )

    rv_next = pl.col("rv").shift(-1).over("inning_group").fill_null(0)
    df = df.with_columns(changeValue=pl.col("play_runs") + (rv_next - pl.col("rv")))

    strike_descs = [
        "called_strike",
        "swinging_strike",
        "swinging_strike_blocked",
        "foul_tip",
        "missed_bunt",
        "bunt_foul_tip",
    ]
    foul_descs = ["foul", "foul_bunt"]
    ball_descs = ["ball", "blocked_ball", "pitchout"]

    transition = (
        pl.when(pl.col("description").is_in(strike_descs))
        .then(pl.lit("strike"))
        .when(pl.col("description").is_in(foul_descs))
        .then(pl.lit("foul"))
        .when(pl.col("description").is_in(ball_descs))
        .then(pl.lit("ball"))
        .otherwise(pl.lit(None))
    )

    df = df.with_columns(transition=transition)

    # linear weights for non-terminal count transitions
    countTransValue = (
        df.filter(pl.col("transition").is_not_null())
        .group_by(["game_year", "balls", "strikes", "transition"])
        .agg(rv=pl.col("changeValue").mean(), n=pl.col("changeValue").count())
    )
    countTransValue.write_csv("data/cleaned/run_exp/count_change_run.csv")
