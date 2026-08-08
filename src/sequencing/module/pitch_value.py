import numpy as np
import polars as pl

_data = None


# cached dataloading function to minimize repeated loading
def _load_data():
    global _data
    if _data is not None:
        return _data
    bipvalue = (
        pl.read_csv("cleaned_data/metrics/re/bipvalue.csv")
        .pivot(on="bb_type", index="game_year", values="changeValue")
        .rename(
            {
                col: f"{col}_value"
                for col in ["line_drive", "ground_ball", "popup", "fly_ball"]
            }
        )
    )
    bb_strikeout = pl.read_csv("cleaned_data/metrics/re/bbK.csv")
    count_trans_value = pl.read_csv("cleaned_data/metrics/re/counttransvalue.csv")
    _data = (bipvalue, bb_strikeout, count_trans_value)
    return _data


# calculate the run value of the given pitch
def pitch_value(
    count,
    game_year,
    whiff_prob,
    foul_prob,
    swing_prob,
    fb_prob,
    ld_prob,
    gb_prob,
    popup_prob,
    hbp_prob,
    strike_prob,
):

    balls, strikes = (int(x) for x in count.split("-"))
    if balls >= 4 or strikes >= 3:
        raise ValueError(f"Invalid count: {count}")

    bipvalue, bb_strikeout, count_trans_value = _load_data()

    bip_row = bipvalue.filter(pl.col("game_year") == game_year)
    ld_value = bip_row["line_drive_value"][0]
    gb_value = bip_row["ground_ball_value"][0]
    popup_value = bip_row["popup_value"][0]
    fb_value = bip_row["fly_ball_value"][0]

    bbk_row = bb_strikeout.filter(pl.col("game_year") == game_year)
    bb_value = bbk_row["bb"][0]
    k_value = bbk_row["k"][0]

    def _lookup_transition(transition, default):
        result = count_trans_value.filter(
            (pl.col("game_year") == game_year)
            & (pl.col("balls") == balls)
            & (pl.col("strikes") == strikes)
            & (pl.col("transition") == transition)
        )
        if result.is_empty():
            return default
        return result["rv"][0]

    if_strike_val = _lookup_transition("strike", k_value)
    if_ball_val = _lookup_transition("ball", bb_value)
    if_foul_val = _lookup_transition("foul", np.nan)

    takeprob = 1 - swing_prob
    ballprob = 1 - strike_prob
    takevalue = strike_prob * if_strike_val + ballprob * if_ball_val
    take_branch = (takeprob - hbp_prob) * takevalue + hbp_prob * bb_value

    whiffvalue = whiff_prob * if_strike_val
    foulvalue = (1 - whiff_prob) * foul_prob * if_foul_val
    bipProb = (1 - whiff_prob) * (1 - foul_prob)
    bipValue = bipProb * (
        (ld_prob * ld_value)
        + (gb_prob * gb_value)
        + (popup_prob * popup_value)
        + (fb_prob * fb_value)
    )

    return take_branch + swing_prob * (whiffvalue + foulvalue + bipValue)
