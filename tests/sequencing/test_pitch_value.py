import polars as pl
import pytest

from src.sequencing.module import pitch_value


def test_pitch_value_uses_ball_probability_for_transition_weights(monkeypatch):
    bipvalue = pl.DataFrame(
        {
            "game_year": [2024],
            "line_drive_value": [0.0],
            "ground_ball_value": [0.0],
            "popup_value": [0.0],
            "fly_ball_value": [0.0],
        }
    )
    bb_strikeout = pl.DataFrame({"game_year": [2024], "bb": [0.0], "k": [0.0]})
    count_transitions = pl.DataFrame(
        {
            "game_year": [2024, 2024, 2024],
            "balls": [0, 0, 0],
            "strikes": [0, 0, 0],
            "transition": ["strike", "ball", "foul"],
            "rv": [2.0, 5.0, 0.0],
        }
    )
    monkeypatch.setattr(
        pitch_value, "_load_data", lambda: (bipvalue, bb_strikeout, count_transitions)
    )

    value = pitch_value.pitch_value(
        count="0-0",
        game_year=2024,
        whiff_prob=0.0,
        foul_prob=0.0,
        swing_prob=0.0,
        fb_prob=0.0,
        ld_prob=0.0,
        gb_prob=0.0,
        popup_prob=0.0,
        hbp_prob=0.0,
        ball_prob=0.2,
    )

    assert value == pytest.approx(0.8 * 2.0 + 0.2 * 5.0)
