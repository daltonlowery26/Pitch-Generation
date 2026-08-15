import numpy as np
import polars as pl
import pytest
import torch

from src.sequencing.module import inference
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
    assert target["count_str"] == "1-1"


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


def test_sequencing_engine_loads_four_class_bip_model(monkeypatch):
    pitch_map = pl.DataFrame(
        {
            "pitcher_year": [0],
            "batter_year": [0],
            "pitch_name": [0],
            "pitchBatter": [0],
            "pitch_type": ["FF"],
        }
    )
    outcome_calls = []

    def fake_load_config(path):
        if path == "config/test/pitch_sequencing.yml":
            return {
                "inference": {
                    "n_cvae_samples": 1,
                    "temperature": 1.0,
                    "game_year": 2024,
                    "hbp_prob": 0.01,
                },
                "data": {"pitch_map": "ignored"},
            }
        return {"path": path}

    monkeypatch.setattr(inference, "load_config", fake_load_config)
    monkeypatch.setattr(inference.pl, "read_parquet", lambda _: pitch_map)
    monkeypatch.setattr(
        inference, "repertoire_mask", lambda *_args: torch.ones(1, 1, dtype=torch.bool)
    )
    monkeypatch.setattr(inference, "load_cvae", lambda *_args: object())
    monkeypatch.setattr(inference, "load_expectation", lambda *_args: object())
    monkeypatch.setattr(
        inference,
        "load_outcome",
        lambda cfg, output_dim, *_args: (
            outcome_calls.append((cfg, output_dim)) or object()
        ),
    )
    monkeypatch.setattr(inference, "load_ball_prob", lambda *_args: object())

    SequencingEngine()

    assert outcome_calls[-1] == ({"path": "config/prob/bip_type.yml"}, 4)


def test_evaluate_exposes_ball_probability_and_canonical_bip_probabilities(
    monkeypatch, pitch_map
):
    engine = SequencingEngine.__new__(SequencingEngine)
    engine.df = pitch_map
    engine.inference_cfg = {
        "n_cvae_samples": 2,
        "temperature": 1.0,
        "game_year": 2024,
        "hbp_prob": 0.01,
    }
    engine.rep_mask = torch.ones(1, 1, dtype=torch.bool)
    engine.cvae = object()
    engine.batter_exp = object()
    engine.swing_model = object()
    engine.whiff_model = object()
    engine.foul_model = object()
    engine.bip_model = object()
    ball_logit = torch.logit(torch.tensor(0.2)).item()
    engine.ball_model = lambda candidates: torch.full(
        (candidates.shape[0], 1), ball_logit
    )
    engine.pitch_labels = {}

    candidates = torch.zeros(2, 3)
    raw_bip_probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    run_value_bip_probs = []
    run_value_ball_probs = []
    batch_count_arrs = []
    run_value_count_strs = []

    monkeypatch.setattr(inference, "lookup_player", lambda *_args: 0)
    monkeypatch.setattr(inference, "get_pitcher_repertoire", lambda *_args: [0])
    monkeypatch.setattr(
        inference, "compute_expectation", lambda *_args: (None, torch.zeros(1))
    )
    monkeypatch.setattr(inference, "generate_cvae_samples", lambda *_args: candidates)

    def fake_build_outcome_batch(*args):
        batch_count_arrs.append(args[6])
        return "batch"

    monkeypatch.setattr(inference, "build_outcome_batch", fake_build_outcome_batch)
    monkeypatch.setattr(
        inference,
        "run_outcome_models",
        lambda *_args: (
            torch.full((2,), 0.5),
            torch.full((2,), 0.2),
            torch.full((2,), 0.1),
            raw_bip_probs,
        ),
    )

    def fake_compute_run_values(*args):
        run_value_bip_probs.append(args[3])
        run_value_count_strs.append(args[4])
        run_value_ball_probs.append(args[7])
        return np.zeros(2, dtype=np.float32)

    monkeypatch.setattr(inference, "compute_run_values", fake_compute_run_values)

    result = engine.evaluate(
        {
            "pitcher": "P1",
            "pitcher_year": 2024,
            "batter": "B1",
            "batter_year": 2024,
            "count_str": "1-2",
        }
    )

    assert result["probs"][0]["bip"].shape == (2, 4)
    assert result["probs"][0]["ball"].tolist() == pytest.approx([0.2, 0.2])
    assert "strike" not in result["probs"][0]
    np.testing.assert_allclose(
        result["probs"][0]["bip"],
        [
            [0.1, 0.3, 0.2, 0.4],
            [0.4, 0.2, 0.3, 0.1],
        ],
    )
    assert torch.equal(run_value_bip_probs[0], raw_bip_probs)
    assert run_value_bip_probs[0].shape == (2, 4)
    assert run_value_ball_probs[0].tolist() == pytest.approx([0.2, 0.2])
    assert batch_count_arrs == [5]
    assert run_value_count_strs == ["1-2"]
