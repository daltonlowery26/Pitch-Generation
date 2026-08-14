import polars as pl
import pytest

from src.sequencing.module.inference import SequencingEngine


@pytest.fixture
def pitch_map():
    return pl.DataFrame(
        {
            "game_pk": [100, 100, 100, 101, 101, 200, 200, 201, 201],
            "at_bat_number": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "pitch_number": [1, 2, 3, 1, 2, 1, 2, 1, 2],
            "pitch_name": [0, 1, 2, 0, 1, 0, 2, 1, 0],
            "pitch_type": [
                "FF",
                "SL",
                "CH",
                "FF",
                "SL",
                "FF",
                "CH",
                "SL",
                "FF",
            ],
            "description": [
                "called_strike",
                "ball",
                "foul",
                "called_strike",
                "ball",
                "ball",
                "called_strike",
                "called_strike",
                "ball",
            ],
            "count": ["0-0", "0-1", "1-1", "0-0", "0-1", "0-0", "1-0", "0-0", "0-1"],
            "game_year": [2024, 2024, 2024, 2024, 2024, 2025, 2025, 2025, 2025],
            "pitcher_name": [
                "P1",
                "P1",
                "P1",
                "P1",
                "P1",
                "P1",
                "P1",
                "P2",
                "P2",
            ],
            "batter_name": [
                "B1",
                "B1",
                "B1",
                "B2",
                "B2",
                "B1",
                "B1",
                "B1",
                "B1",
            ],
            "delta_run_exp": [0.10, 0.20, -0.30, 0.15, 0.40, 0.05, 0.50, 0.25, -0.60],
        }
    )


@pytest.fixture
def engine(pitch_map):
    engine = SequencingEngine.__new__(SequencingEngine)
    engine.df = pitch_map
    return engine


def test_sample_scenarios_is_deterministic_for_a_seed(engine):
    first = engine.sample_scenarios(n=2, seed=7)
    second = engine.sample_scenarios(n=2, seed=7)
    different_seed = engine.sample_scenarios(n=2, seed=8)

    assert len(first) == 2
    assert len(different_seed) == 2
    assert first == second
    assert {scenario["observed_run_value"] for scenario in first} != {
        scenario["observed_run_value"] for scenario in different_seed
    }


def test_sample_scenarios_filters_pitcher_batter_and_year(engine):
    pitcher_scenarios = engine.sample_scenarios(
        n=12,
        pitcher="P2",
        pitcher_year=2025,
        seed=7,
    )
    other_pitcher_scenarios = engine.sample_scenarios(
        n=12,
        pitcher="P1",
        pitcher_year=2025,
        seed=7,
    )
    earlier_year_scenarios = engine.sample_scenarios(
        n=12,
        pitcher_year=2024,
        seed=7,
    )
    later_year_scenarios = engine.sample_scenarios(
        n=12,
        pitcher_year=2025,
        seed=7,
    )
    batter_scenarios = engine.sample_scenarios(
        n=12,
        batter="B2",
        batter_year=2024,
        seed=7,
    )
    other_batter_scenarios = engine.sample_scenarios(
        n=12,
        batter="B1",
        batter_year=2024,
        seed=7,
    )
    earlier_batter_year_scenarios = engine.sample_scenarios(
        n=12,
        batter_year=2024,
        seed=7,
    )
    later_batter_year_scenarios = engine.sample_scenarios(
        n=12,
        batter_year=2025,
        seed=7,
    )
    combined_scenarios = engine.sample_scenarios(
        n=12,
        pitcher="P1",
        pitcher_year=2024,
        batter="B1",
        batter_year=2024,
        seed=7,
    )

    assert len(pitcher_scenarios) == 1
    assert pitcher_scenarios[0]["pitcher"] == "P2"
    assert pitcher_scenarios[0]["pitcher_year"] == 2025
    assert pitcher_scenarios[0]["observed_run_value"] == pytest.approx(-0.60)
    assert other_pitcher_scenarios[0]["pitcher"] == "P1"
    assert {scenario["observed_run_value"] for scenario in pitcher_scenarios} != {
        scenario["observed_run_value"] for scenario in other_pitcher_scenarios
    }

    assert {scenario["observed_run_value"] for scenario in earlier_year_scenarios} != {
        scenario["observed_run_value"] for scenario in later_year_scenarios
    }
    assert {scenario["pitcher_year"] for scenario in earlier_year_scenarios} == {2024}
    assert {scenario["pitcher_year"] for scenario in later_year_scenarios} == {2025}

    assert len(batter_scenarios) == 1
    assert batter_scenarios[0]["batter"] == "B2"
    assert batter_scenarios[0]["observed_run_value"] == pytest.approx(0.40)
    assert other_batter_scenarios[0]["batter"] == "B1"
    assert {scenario["observed_run_value"] for scenario in batter_scenarios} != {
        scenario["observed_run_value"] for scenario in other_batter_scenarios
    }

    assert {
        scenario["observed_run_value"] for scenario in earlier_batter_year_scenarios
    } != {scenario["observed_run_value"] for scenario in later_batter_year_scenarios}
    assert {scenario["batter_year"] for scenario in earlier_batter_year_scenarios} == {
        2024
    }
    assert {scenario["batter_year"] for scenario in later_batter_year_scenarios} == {
        2025
    }

    assert len(combined_scenarios) == 1
    assert {
        (
            scenario["pitcher"],
            scenario["pitcher_year"],
            scenario["batter"],
            scenario["batter_year"],
        )
        for scenario in combined_scenarios
    } == {("P1", 2024, "B1", 2024)}


@pytest.mark.parametrize(
    ("kwargs", "missing_year"),
    [
        ({"pitcher": "P1"}, "pitcher_year"),
        ({"batter": "B1"}, "batter_year"),
    ],
)
def test_sample_scenarios_requires_a_year_for_named_players(
    engine, kwargs, missing_year
):
    with pytest.raises(ValueError, match=missing_year):
        engine.sample_scenarios(**kwargs)


def test_sample_scenarios_returns_fewer_than_twelve_when_pool_is_smaller(engine):
    scenarios = engine.sample_scenarios(n=12, seed=7)

    assert len(scenarios) == 4


def test_sample_scenarios_raises_for_an_empty_pool(engine):
    with pytest.raises(ValueError, match="No eligible scenarios"):
        engine.sample_scenarios(
            pitcher="Nobody",
            pitcher_year=2024,
            batter="B1",
            batter_year=2024,
        )


def test_sample_scenarios_converts_prior_pitches_to_pitch_result_pairs(engine):
    scenarios = engine.sample_scenarios(
        pitcher="P1",
        pitcher_year=2024,
        batter="B1",
        batter_year=2024,
        seed=7,
    )
    target = next(
        scenario
        for scenario in scenarios
        if scenario["observed_run_value"] == pytest.approx(-0.30)
    )

    assert target["prior_sequence"] == [(0, "strike"), (1, "ball")]
    assert target["observed_pitch"] == "CH"


def test_sample_scenarios_preserves_missed_bunt_as_a_strike(engine):
    engine.df = engine.df.with_columns(
        pl.when((pl.col("game_pk") == 100) & (pl.col("pitch_number") == 1))
        .then(pl.lit("missed_bunt"))
        .otherwise(pl.col("description"))
        .alias("description")
    )

    scenarios = engine.sample_scenarios(
        pitcher="P1",
        pitcher_year=2024,
        batter="B1",
        batter_year=2024,
        seed=7,
    )
    target = next(
        scenario
        for scenario in scenarios
        if scenario["observed_run_value"] == pytest.approx(-0.30)
    )

    assert target["prior_sequence"] == [(0, "strike"), (1, "ball")]


def test_sample_scenarios_rejects_an_unsupported_prior_description(engine):
    engine.df = engine.df.with_columns(
        pl.when((pl.col("game_pk") == 100) & (pl.col("pitch_number") == 1))
        .then(pl.lit("foul_pitchout"))
        .otherwise(pl.col("description"))
        .alias("description")
    )

    scenarios = engine.sample_scenarios(n=12, seed=7)

    assert len(scenarios) == 3
    assert all(
        scenario["observed_run_value"] != pytest.approx(-0.30) for scenario in scenarios
    )


def test_sample_scenarios_preserves_observed_delta_run_values(engine):
    scenarios = engine.sample_scenarios(n=12, seed=7)

    observed_by_identity = {
        (
            scenario["pitcher"],
            scenario["pitcher_year"],
            scenario["batter"],
            scenario["batter_year"],
        ): scenario["observed_run_value"]
        for scenario in scenarios
    }
    expected_values = {
        ("P1", 2024, "B1", 2024): -0.30,
        ("P1", 2024, "B2", 2024): 0.40,
        ("P1", 2025, "B1", 2025): 0.50,
        ("P2", 2025, "B1", 2025): -0.60,
    }

    assert set(observed_by_identity) == set(expected_values)
    for identity, expected_value in expected_values.items():
        assert observed_by_identity[identity] == pytest.approx(expected_value)
