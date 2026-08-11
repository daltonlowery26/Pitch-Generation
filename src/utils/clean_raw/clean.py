# packages and setup
import numpy as np
import polars as pl

# constants
cols_to_remove = [
    "",
    "spin_rate_deprecated",
    "des",
    "break_angle_deprecated",
    "break_length_deprecated",
    "spin_dir",
    "hit_location",
    "game_type",
    "on_3b",
    "on_2b",
    "on_1b",
    "outs_when_up",
    "inning_topbot",
    "tfs_deprecated",
    "tfs_zulu_deprecated",
    "umpire",
    "fielder_2",
    "fielder_2",
    "fielder_3",
    "fielder_4",
    "fielder_5",
    "fielder_6",
    "fielder_7",
    "fielder_8",
    "fielder_9",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "post_away_score",
    "post_home_score",
    "sv_id",
    "post_bat_score",
    "post_fld_score",
    "home_score_diff",
    "bat_score_diff",
    "age_pit_legacy",
    "home_win_exp",
    "woba_denom",
    "if_fielding_alignment",
    "babip_value",
    "launch_speed_angle",
    "iso_value",
    "of_fielding_alignment",
    "delta_home_win_exp",
    "bat_win_exp",
    "age_pit_legacy",
    "age_bat_legacy",
    "hyper_speed",
    "age_pit",
    "age_bat",
    "pitcher_days_since_prev_game",
    "batter_days_since_prev_game",
    "pitcher_days_until_next_game",
    "batter_days_until_next_game",
    "delta_pitcher_run_exp",
]

# group of pitches
pitch_groups = {
    "fastball": {"4-Seam Fastball", "Sinker", "Cutter"},
    "breaking": {
        "Knuckle Curve",
        "Screwball",
        "Slow Curve",
        "Curveball",
        "Slider",
        "Slurve",
        "Sweeper",
    },
    "offspeed": {"Changeup", "Split-Finger", "Forkball"},
    "others": {
        "Other",
        "Unknown",
        "None",
        "Knuckleball",
        "Eephus",
        "Intentional Ball",
        "Pitch Out",
    },
}
pitch_to_group = {
    pitch: group for group, pitches in pitch_groups.items() for pitch in pitches
}


# release-angle helpers (m_rosen kirby index)
def calculate_VRA(vy0, ay, release_extension, vz0, az):
    d_release = 60.5 - release_extension - 50
    vy_s = -((vy0**2 + 2 * ay * d_release).sqrt())
    t_s = (vy0 - vy_s) / ay
    vz_s = vz0 - az * t_s
    VRA = -(vz_s / vy_s).arctan() * (180 / np.pi)
    return VRA


def calculate_HRA(vy0, ay, release_extension, vx0, ax):
    d_release = 60.5 - release_extension - 50
    vy_s = -((vy0**2 + 2 * ay * d_release).sqrt())
    t_s = (vy0 - vy_s) / ay
    vx_s = vx0 - ax * t_s
    HRA = -(vx_s / vy_s).arctan() * (180 / np.pi)
    return HRA


if __name__ == "__main__":
    # load, filter, drop cols, count, rename
    unclean = (
        pl.scan_csv("data/raw/pitch_2015_2026.csv")
        .filter(pl.col("game_type") == "R")
        .select(pl.exclude(cols_to_remove))
        .with_columns(
            pl.concat_str(pl.col("balls"), pl.col("strikes"), separator="-").alias(
                "count"
            )
        )
        .rename(
            {
                "batter": "batter_id",
                "pitcher": "pitcher_id",
                "player_name": "pitcher_name",
                "release_pos_z": "release_height",
                "release_pos_x": "release_x",
                "stand": "b_stand",
                "estimated_woba_using_speedangle": "xwoba",
                "intercept_ball_minus_batter_pos_x_inches": "intercept_x",
                "intercept_ball_minus_batter_pos_y_inches": "intercept_y",
            }
        )
    ).collect(engine="streaming")

    # join batter names (fangraphs merge)
    bat_names = (
        pl.scan_csv("data/raw/batter_names.csv")
        .rename({"last_name, first_name": "batter_name"})
        .select(pl.exclude("year"))
        .unique()
    ).collect(engine="streaming")

    unclean = unclean.join(
        bat_names, left_on="batter_id", right_on="player_id", how="left"
    )

    # labels and identifiers
    unclean = unclean.with_columns(
        (
            pl.col("plate_z").is_between(pl.col("sz_bot"), pl.col("sz_top"))
            & pl.col("plate_x").is_between(-0.7083, 0.7083)
        ).alias("abs_strike")
    )

    # pitch group
    unclean = unclean.with_columns(
        pitch_group=pl.col("pitch_name").replace_strict(
            pitch_to_group, default="others"
        )
    )
    unclean = unclean.filter(pl.col("pitch_group") != "others")
    print(len(unclean), flush=True)
    unclean = unclean.with_columns(pitch_group=pl.col("pitch_group").rank("dense") - 1)
    des = unclean["description"].unique()

    # pitch name
    unclean = unclean.with_columns(
        pitch_type=pl.col("pitch_name"),
        pitch_name=pl.col("pitch_name").rank("dense") - 1,
    )

    # outcome labels + sequence identifiers
    unclean = unclean.with_columns(
        atBatNumber=pl.col("at_bat_number")
        .rank("dense")
        .over(["game_pk", "pitcher_id", "batter_id"]),
        pitchBatter=pl.struct(["game_pk", "pitcher_id", "batter_id"]).rank("dense"),
        countArr=pl.col("count").str.split("-").list.get(0).cast(pl.Int32) * 3
        + pl.col("count").str.split("-").list.get(1).cast(pl.Int32),
        swing=pl.col("description").is_in(
            [
                "hit_into_play",
                "foul",
                "foul_bunt",
                "foul_pitchout",
                "swinging_pitchout",
                "missed_bunt",
                "swinging_strike",
                "foul_tip",
                "swinging_strike_blocked",
                "bunt_foul_tip",
            ]
        ),
        foul=pl.col("description").is_in(
            ["foul_tip", "foul", "foul_bunt", "foul_pitchout", "bunt_foul_tip"]
        ),
        whiff=pl.col("description").is_in(
            [
                "swinging_strike_blocked",
                "swinging_pitchout",
                "swinging_strike",
                "missed_bunt",
            ]
        ),
        bipArr=pl.concat_list(
            (pl.col("bb_type") == "fly_ball"),
            (pl.col("bb_type") == "ground_ball"),
            (pl.col("bb_type") == "line_drive"),
            (pl.col("bb_type") == "popup"),
            (pl.col("bb_type").is_null()),
        ),
    )

    # remove non-continuous pitch sequences + any pitchBatter containing them
    unclean = unclean.with_columns(
        pl.len().over(["game_pk", "at_bat_number"]).alias("_n_pitches"),
        pl.col("pitch_number")
        .min()
        .over(["game_pk", "at_bat_number"])
        .alias("_min_pitch"),
        pl.col("pitch_number")
        .max()
        .over(["game_pk", "at_bat_number"])
        .alias("_max_pitch"),
        pl.col("pitch_number")
        .n_unique()
        .over(["game_pk", "at_bat_number"])
        .alias("_n_unique"),
    )

    unclean = unclean.with_columns(
        pl.when(
            (pl.col("_min_pitch") == 1)
            & (pl.col("_max_pitch") == pl.col("_n_pitches"))
            & (pl.col("_n_unique") == pl.col("_n_pitches"))
        )
        .then(0)
        .otherwise(1)
        .alias("_bad_atbat")
    )

    unclean = unclean.with_columns(
        pl.col("_bad_atbat").max().over("pitchBatter").alias("_bad_matchup")
    )

    unclean = unclean.filter(pl.col("_bad_matchup") == 0).drop(
        [
            "_n_pitches",
            "_min_pitch",
            "_max_pitch",
            "_n_unique",
            "_bad_atbat",
            "_bad_matchup",
        ]
    )
    print(len(unclean), flush=True)

    # pitcher / batter year identifiers
    unclean = unclean.with_columns(
        pitcher_year=pl.struct(["pitcher_id", "game_year"]).rank("dense") - 1,
        batter_year=pl.struct(["batter_id", "game_year"]).rank("dense") - 1,
    )
    unclean = unclean.with_columns(
        pitcher_year=pl.when(pl.len().over("pitcher_year") < 50)
        .then(0)
        .otherwise(pl.col("pitcher_year").rank("dense"))
    ).with_columns(pitcher_year=pl.col("pitcher_year").rank("dense") - 1)
    unclean = unclean.with_columns(
        batter_year=pl.when(pl.len().over("batter_year") < 50)
        .then(0)
        .otherwise(pl.col("batter_year").rank("dense"))
    ).with_columns(batter_year=pl.col("batter_year").rank("dense") - 1)

    # typing date
    unclean = unclean.with_columns(
        game_date=pl.col("game_date").str.to_date("%Y-%m-%d")
    )

    # physics-derived features
    vy_f = -(pl.col("vy0").pow(2) - (2 * pl.col("ay") * (50 - (17 / 12)))).sqrt()
    t = (vy_f - pl.col("vy0")) / pl.col("ay")
    vz_f = pl.col("vz0") + (pl.col("az") * t)
    unclean = unclean.with_columns(vaa=-(vz_f / vy_f).arctan() * (180 / np.pi))

    vx_f = pl.col("vx0") + pl.col("ax") * t
    unclean = unclean.with_columns(haa=-(vx_f / vy_f).arctan() * (180 / np.pi))

    # vra / hra (release_extension filled first)
    unclean = unclean.with_columns(
        pl.col("release_extension").fill_null(pl.col("release_extension").mean())
    )
    unclean = unclean.with_columns(
        calculate_VRA(
            pl.col("vy0"),
            pl.col("ay"),
            pl.col("release_extension"),
            pl.col("vz0"),
            pl.col("az"),
        ).alias("vra"),
        calculate_HRA(
            pl.col("vy0"),
            pl.col("ay"),
            pl.col("release_extension"),
            pl.col("vx0"),
            pl.col("ax"),
        ).alias("hra"),
    )

    # column ordering (names at start)
    start = ["game_date", "pitcher_name", "pitcher_id", "batter_name", "batter_id"]
    unclean = unclean.select(
        pl.col(start),
        pl.exclude(start),  # get rest of cols
    )

    # write cleaned parquet
    unclean.write_parquet("data/cleaned/pitch_2015_2026.parquet")
