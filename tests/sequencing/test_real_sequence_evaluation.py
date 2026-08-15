import numpy as np
import polars as pl
import pytest
import torch

from src.sequencing.module import inference


def _scenario_engine(df):
    engine = inference.SequencingEngine.__new__(inference.SequencingEngine)
    engine.df = df
    engine.inference_cfg = {
        "n_cvae_samples": 1,
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
    engine.ball_model = lambda candidates: torch.zeros(candidates.shape[0], 1)
    engine.pitch_labels = {}
    return engine


def test_sample_scenarios_preserves_real_prior_embeddings():
    pitch_map = pl.DataFrame(
        {
            "game_pk": [100, 100, 100],
            "at_bat_number": [1, 1, 1],
            "pitch_number": [1, 2, 3],
            "pitch_name": [0, 1, 2],
            "pitch_type": ["FF", "SL", "CH"],
            "description": ["called_strike", "ball", "hit_into_play"],
            "count": ["0-0", "0-1", "1-1"],
            "game_year": [2024, 2024, 2024],
            "pitcher_name": ["P1", "P1", "P1"],
            "batter_name": ["B1", "B1", "B1"],
            "delta_run_exp": [0.1, 0.2, -0.3],
            "kmu": [
                list(range(10)),
                list(range(10, 20)),
                list(range(20, 30)),
            ],
        }
    )
    engine = _scenario_engine(pitch_map)

    scenario = engine.sample_scenarios(n=1, seed=7)[0]

    assert scenario["prior_sequence"] == [(0, "strike"), (1, "ball")]
    assert scenario["prior"]["pitchEmb"] == [list(range(10)), list(range(10, 20))]
    assert scenario["prior"]["count"] == [0, 1]
    assert scenario["prior"]["pitch_type"] == ["FF", "SL"]


def test_sampled_scenario_evaluates_with_its_real_prior(monkeypatch):
    pitch_map = pl.DataFrame(
        {
            "game_pk": [100, 100, 100],
            "at_bat_number": [1, 1, 1],
            "pitch_number": [1, 2, 3],
            "pitch_name": [0, 1, 2],
            "pitch_type": ["FF", "SL", "CH"],
            "description": ["called_strike", "ball", "hit_into_play"],
            "count": ["0-0", "0-1", "1-1"],
            "game_year": [2024, 2024, 2024],
            "pitcher_name": ["P1", "P1", "P1"],
            "batter_name": ["B1", "B1", "B1"],
            "delta_run_exp": [0.1, 0.2, -0.3],
            "kmu": [
                list(range(10)),
                list(range(10, 20)),
                list(range(20, 30)),
            ],
        }
    )
    engine = _scenario_engine(pitch_map)
    prior_seen = {}

    monkeypatch.setattr(inference, "lookup_player", lambda *_args: 0)
    monkeypatch.setattr(inference, "get_pitcher_repertoire", lambda *_args: [0])

    def fake_expectation(_model, prior, *_args):
        prior_seen["prior"] = prior
        return None, torch.zeros(1)

    monkeypatch.setattr(
        inference,
        "compute_expectation",
        fake_expectation,
    )
    monkeypatch.setattr(
        inference,
        "generate_cvae_samples",
        lambda *_args: torch.zeros(1, 10),
    )
    monkeypatch.setattr(inference, "build_outcome_batch", lambda *_args: "batch")
    monkeypatch.setattr(
        inference,
        "run_outcome_models",
        lambda *_args: (
            torch.full((1,), 0.5),
            torch.full((1,), 0.2),
            torch.full((1,), 0.1),
            torch.full((1, 4), 0.25),
        ),
    )
    monkeypatch.setattr(
        inference,
        "compute_run_values",
        lambda *_args: np.zeros(1, dtype=np.float32),
    )
    monkeypatch.setattr(
        inference,
        "sample_prior_pitches",
        lambda *_args: pytest.fail("real scenarios must not sample prior pitches"),
    )

    scenario = engine.sample_scenarios(n=1, seed=7)[0]
    engine.evaluate(scenario)

    np.testing.assert_allclose(
        prior_seen["prior"]["pitchEmb"], [list(range(10)), list(range(10, 20))]
    )


def test_evaluate_uses_real_prior_without_sampling(monkeypatch):
    pitch_map = pl.DataFrame(
        {
            "pitcher_name": ["P1"],
            "batter_name": ["B1"],
            "game_year": [2024],
            "pitcher_year": [0],
            "batter_year": [0],
            "pitch_name": [0],
        }
    )
    engine = _scenario_engine(pitch_map)
    prior_seen = {}

    monkeypatch.setattr(inference, "lookup_player", lambda *_args: 0)
    monkeypatch.setattr(inference, "get_pitcher_repertoire", lambda *_args: [0])

    def fake_expectation(_model, prior, *_args):
        prior_seen["prior"] = prior
        return None, torch.zeros(1)

    monkeypatch.setattr(inference, "compute_expectation", fake_expectation)
    monkeypatch.setattr(
        inference,
        "generate_cvae_samples",
        lambda *_args: torch.zeros(1, 10),
    )
    monkeypatch.setattr(inference, "build_outcome_batch", lambda *_args: "batch")
    monkeypatch.setattr(
        inference,
        "run_outcome_models",
        lambda *_args: (
            torch.full((1,), 0.5),
            torch.full((1,), 0.2),
            torch.full((1,), 0.1),
            torch.full((1, 4), 0.25),
        ),
    )
    monkeypatch.setattr(
        inference,
        "compute_run_values",
        lambda *_args: np.zeros(1, dtype=np.float32),
    )
    monkeypatch.setattr(
        inference,
        "sample_prior_pitches",
        lambda *_args: pytest.fail("real scenarios must not sample prior pitches"),
    )

    scenario = {
        "pitcher": "P1",
        "pitcher_year": 2024,
        "batter": "B1",
        "batter_year": 2024,
        "prior_sequence": [(0, "strike")],
        "prior": {
            "pitchEmb": [[1.0] * 10],
            "count": [0],
            "atbat": [0],
            "pitch_type": ["FF"],
            "description": ["called_strike"],
            "seq_len": 1,
        },
        "count_str": "0-1",
    }

    engine.evaluate(scenario)

    np.testing.assert_allclose(prior_seen["prior"]["pitchEmb"], [[1.0] * 10])
    assert prior_seen["prior"]["count"].tolist() == [0]
    assert prior_seen["prior"]["pitcher_id"] == 0
    assert prior_seen["prior"]["batter_id"] == 0


def test_evaluate_samples_prior_when_requested(monkeypatch):
    pitch_map = pl.DataFrame(
        {
            "pitcher_name": ["P1"],
            "batter_name": ["B1"],
            "game_year": [2024],
            "pitcher_year": [0],
            "batter_year": [0],
            "pitch_name": [0],
        }
    )
    engine = _scenario_engine(pitch_map)
    sampled = {
        "pitchEmb": np.ones((1, 10), dtype=np.float32),
        "count": np.array([0], dtype=np.int32),
        "atbat": np.array([0], dtype=np.int32),
        "pitch_type": ["FF"],
        "description": ["called_strike"],
        "seq_len": 1,
    }
    sample_calls = []

    monkeypatch.setattr(inference, "lookup_player", lambda *_args: 0)
    monkeypatch.setattr(inference, "get_pitcher_repertoire", lambda *_args: [0])
    monkeypatch.setattr(
        inference,
        "sample_prior_pitches",
        lambda *_args: (sampled, "0-1", 1),
    )
    monkeypatch.setattr(
        inference,
        "compute_expectation",
        lambda _model, prior, *_args: (
            sample_calls.append(prior) or (None, torch.zeros(1))
        ),
    )
    monkeypatch.setattr(
        inference,
        "generate_cvae_samples",
        lambda *_args: torch.zeros(1, 10),
    )
    monkeypatch.setattr(inference, "build_outcome_batch", lambda *_args: "batch")
    monkeypatch.setattr(
        inference,
        "run_outcome_models",
        lambda *_args: (
            torch.full((1,), 0.5),
            torch.full((1,), 0.2),
            torch.full((1,), 0.1),
            torch.full((1, 4), 0.25),
        ),
    )
    monkeypatch.setattr(
        inference,
        "compute_run_values",
        lambda *_args: np.zeros(1, dtype=np.float32),
    )

    engine.evaluate(
        {
            "pitcher": "P1",
            "pitcher_year": 2024,
            "batter": "B1",
            "batter_year": 2024,
            "prior_sequence": [(0, "strike")],
        },
        sample_sequence=True,
    )

    assert len(sample_calls) == 1
    assert sample_calls[0]["pitcher_id"] == 0
