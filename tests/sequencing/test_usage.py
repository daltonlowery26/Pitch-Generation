import numpy as np
import pytest
import torch

from src.embeddings.pitch.modules.cvae import CVAE
from src.sequencing import usage
from src.sequencing.module import inference, utils

utils.device = torch.device("cpu")
inference.device = torch.device("cpu")


def test_cvae_generation_is_repeatable_with_a_torch_generator():
    model = CVAE(
        input_dim=3,
        pitcher_dim=2,
        pitch_dim=2,
        h_dim=4,
        latent_dim=2,
        dropout=0.0,
        n_layers=1,
        n_pitchers=2,
        n_pitches=2,
    ).eval()
    condition = torch.zeros(1, 4)

    first = model.generate(condition, n=3, generator=torch.Generator().manual_seed(7))
    second = model.generate(condition, n=3, generator=torch.Generator().manual_seed(7))

    assert torch.equal(first, second)


def test_generate_cvae_samples_forwards_generator():
    class FakeCVAE:
        def __init__(self):
            self.pitcherEmbed = lambda ids: torch.stack(
                [ids.float(), ids.float() + 1], dim=-1
            )
            self.pitchEmbed = lambda ids: torch.stack(
                [ids.float() + 10, ids.float() + 11, ids.float() + 12], dim=-1
            )
            self.calls = []

        def generate(self, condition, **kwargs):
            self.calls.append((condition, kwargs))
            return torch.zeros(kwargs["n"], 4)

    model = FakeCVAE()
    generator = torch.Generator().manual_seed(7)

    utils.generate_cvae_samples(model, 1, 2, 10, 1.5, generator=generator)

    assert torch.equal(
        model.calls[0][0],
        torch.tensor([[1.0, 2.0, 12.0, 13.0, 14.0]]),
    )
    assert model.calls[0][1]["generator"] is generator
    assert model.calls[0][1]["n"] == 10
    assert model.calls[0][1]["temperature"] == 1.5


def test_usage_generator_uses_model_device():
    model = CVAE(
        input_dim=3,
        pitcher_dim=2,
        pitch_dim=2,
        h_dim=4,
        latent_dim=2,
        dropout=0.0,
        n_layers=1,
        n_pitchers=2,
        n_pitches=2,
    )

    generator = usage._make_torch_generator(model, seed=7)

    assert generator.device == torch.device("cpu")


def test_build_outcome_batch_uses_explicit_current_atbat():
    prior = {
        "pitchEmb": np.zeros((2, 3), dtype=np.float32),
        "atbat": np.array([0, 0], dtype=np.int32),
        "count": np.array([0, 1], dtype=np.int32),
        "seq_len": 2,
    }
    candidates = torch.zeros(4, 3)
    prior_exp = torch.zeros(2, 5)
    candidate_exp = torch.zeros(5)

    batch = utils.build_outcome_batch(
        prior,
        candidates,
        prior_exp,
        candidate_exp,
        pitcher_id_val=1,
        batter_id_val=2,
        current_count_arr=0,
        current_atbat=1,
    )

    assert batch[4].tolist() == [[0, 0, 1]] * 4


def test_aggregate_bip_probs_maps_four_model_classes_to_consumer_order():
    bip_probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    aggregated = utils.aggregate_bip_probs(bip_probs)

    assert aggregated.shape == (1, 4)
    assert aggregated[0].tolist() == pytest.approx([0.1, 0.3, 0.2, 0.4])


def test_aggregate_bip_probs_rejects_inputs_with_fewer_than_four_classes():
    with pytest.raises(ValueError, match="at least four classes"):
        utils.aggregate_bip_probs(torch.zeros(1, 3))


def test_aggregate_bip_probs_ignores_fifth_model_class():
    bip_probs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.9]])

    aggregated = utils.aggregate_bip_probs(bip_probs)

    assert aggregated.shape == (1, 4)
    assert aggregated[0].tolist() == pytest.approx([0.1, 0.3, 0.2, 0.4])


def test_compute_run_values_passes_ball_probability_and_four_class_bip_probabilities(
    monkeypatch,
):
    calls = []

    def fake_pitch_value(**kwargs):
        calls.append(kwargs)
        return 1.0

    monkeypatch.setattr(utils, "pitch_value", fake_pitch_value)
    run_values = utils.compute_run_values(
        torch.tensor([0.5]),
        torch.tensor([0.2]),
        torch.tensor([0.1]),
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "1-2",
        2024,
        0.01,
        ball_probs=torch.tensor([0.6]),
    )

    assert run_values.tolist() == [1.0]
    assert calls[0]["ball_prob"] == pytest.approx(0.6)
    assert calls[0]["fb_prob"] == pytest.approx(0.1)
    assert calls[0]["ld_prob"] == pytest.approx(0.3)
    assert calls[0]["gb_prob"] == pytest.approx(0.2)
    assert calls[0]["popup_prob"] == pytest.approx(0.4)


def test_score_pitch_candidates_returns_state_conditioned_candidates(monkeypatch):
    calls = {}

    class FakeCVAE:
        pass

    class FakeExpectation:
        max_len = 20

    engine = inference.SequencingEngine.__new__(inference.SequencingEngine)
    engine.inference_cfg = {"n_cvae_samples": 10, "temperature": 2.0, "hbp_prob": 0.01}
    engine.cvae = FakeCVAE()
    engine.batter_exp = FakeExpectation()
    engine.rep_mask = torch.ones(6, 3, dtype=torch.bool)
    engine.ball_model = lambda candidates: torch.zeros(candidates.shape[0], 1)
    engine.swing_model = object()
    engine.whiff_model = object()
    engine.foul_model = object()
    engine.bip_model = object()

    candidates = torch.zeros(10, 4)
    prior = {
        "pitchEmb": np.zeros((1, 4), dtype=np.float32),
        "atbat": np.array([1], dtype=np.int32),
        "count": np.array([0], dtype=np.int32),
        "seq_len": 1,
    }

    def fake_generate(*args, **kwargs):
        calls["generate"] = (args, kwargs)
        return candidates

    def fake_expectation(*args):
        calls["expectation"] = args
        assert args[1]["pitcher_id"] == 4
        assert args[1]["batter_id"] == 5
        return None, torch.zeros(3)

    def fake_batch(*args, **kwargs):
        calls["batch"] = (args, kwargs)
        return "batch"

    def fake_outcomes(*args):
        return (
            torch.full((10,), 0.2),
            torch.full((10,), 0.3),
            torch.full((10,), 0.1),
            torch.full((10, 4), 0.25),
        )

    def fake_run_values(*args):
        calls["run_values"] = args
        return np.arange(10, dtype=np.float32)

    monkeypatch.setattr(inference, "generate_cvae_samples", fake_generate)
    monkeypatch.setattr(inference, "compute_expectation", fake_expectation)
    monkeypatch.setattr(inference, "build_outcome_batch", fake_batch)
    monkeypatch.setattr(inference, "run_outcome_models", fake_outcomes)
    monkeypatch.setattr(inference, "compute_run_values", fake_run_values)

    generator = torch.Generator().manual_seed(3)
    result = engine.score_pitch_candidates(
        pitcher_id=4,
        batter_id=5,
        pitch_type=2,
        prior=prior,
        current_count_str="1-2",
        current_count_arr=5,
        current_atbat=1,
        season=2024,
        generator=generator,
    )

    assert result["candidates"] is candidates
    assert result["run_values"].tolist() == list(range(10))
    assert result["probs"]["swing"].shape == (10,)
    assert result["probs"]["ball"].tolist() == pytest.approx([0.5] * 10)
    assert "strike" not in result["probs"]
    assert result["probs"]["bip"].shape == (10, 4)
    assert result["probs"]["bip"][0].tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert calls["generate"][1]["generator"] is generator
    assert calls["batch"][1]["current_atbat"] == 1
    assert calls["run_values"][4] == "1-2"
    assert calls["run_values"][5] == 2024


def test_score_pitch_candidates_rejects_maximum_history():
    class FakeExpectation:
        max_len = 2

    engine = inference.SequencingEngine.__new__(inference.SequencingEngine)
    engine.batter_exp = FakeExpectation()

    prior = {"seq_len": 2}
    with pytest.raises(ValueError, match="maximum"):
        engine.score_pitch_candidates(
            pitcher_id=1,
            batter_id=2,
            pitch_type=3,
            prior=prior,
            season=2024,
        )


def test_simulate_optimal_usage_samples_two_history_preserving_at_bats(
    monkeypatch,
):
    pitch_map = {
        "pitcher_name": ["P1"] * 4,
        "game_year": [2024] * 4,
        "pitcher_year": [7] * 4,
        "pitch_name": [10, 20, 10, 20],
        "batter_name": ["B1", "B1", "B2", "B2"],
        "batter_year": [11, 11, 12, 12],
    }
    calls = []

    class FakeEngine:
        def __init__(self, _pitch_map):
            self.df = __import__("polars").DataFrame(pitch_map)
            self.cvae = torch.nn.Linear(1, 1)

        def score_pitch_candidates(self, **kwargs):
            calls.append(
                {
                    **kwargs,
                    "prior": None
                    if kwargs["prior"] is None
                    else {
                        key: value.copy() if hasattr(value, "copy") else value
                        for key, value in kwargs["prior"].items()
                    },
                }
            )
            pitch_type = kwargs["pitch_type"]
            run_value = 2.0 if pitch_type == 10 else -1.0
            return {
                "candidates": torch.zeros(10, 2),
                "probs": {
                    "swing": np.zeros(10),
                    "whiff": np.zeros(10),
                    "foul": np.zeros(10),
                    "ball": np.ones(10),
                    "bip": np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
                },
                "run_values": np.full(10, run_value),
            }

    monkeypatch.setattr(usage, "SequencingEngine", FakeEngine)

    result = usage.simulate_optimal_usage(
        pitch_map,
        pitcher="P1",
        season=2024,
        n_batters=3,
        seed=7,
    )

    assert result["pitch_counts"] == {10: 0, 20: 24}
    assert result["pitch_shares"] == {10: 0.0, 20: 1.0}
    assert result["outcome_counts"]["ball"] == 24
    assert result["terminal_counts"]["walk"] == 6
    assert len(calls) == 48
    assert {call["pitch_type"] for call in calls} == {10, 20}
    assert {call["current_atbat"] for call in calls} == {0, 1}
    assert any(call["prior"] is not None for call in calls)
    assert calls[0]["prior"] is None
    assert calls[24]["current_count_str"] == "0-0"
    assert calls[24]["current_atbat"] == 1
    for batter_id in {call["batter_id"] for call in calls}:
        batter_calls = [call for call in calls if call["batter_id"] == batter_id]
        first_atbat = next(call for call in batter_calls if call["current_atbat"] == 0)
        second_atbat = next(call for call in batter_calls if call["current_atbat"] == 1)
        assert first_atbat["prior"] is None
        assert second_atbat["prior"]["seq_len"] == 4


def test_simulate_optimal_usage_uses_target_observed_repertoire_for_pooled_id(
    monkeypatch,
):
    import polars as pl

    pitch_map = pl.DataFrame(
        {
            "pitcher_name": ["P1", "P2"],
            "game_year": [2024, 2024],
            "pitcher_year": [0, 0],
            "pitch_name": [10, 20],
            "batter_name": ["B1", "B2"],
            "batter_year": [0, 1],
        }
    )
    calls = []

    class FakeEngine:
        def __init__(self, _pitch_map):
            self.df = pitch_map
            self.cvae = torch.nn.Linear(1, 1)

        def score_pitch_candidates(self, **kwargs):
            calls.append(kwargs)
            return {
                "candidates": torch.zeros(10, 2),
                "probs": {
                    "swing": np.zeros(10),
                    "whiff": np.zeros(10),
                    "foul": np.zeros(10),
                    "ball": np.zeros(10),
                    "bip": np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
                },
                "run_values": np.full(10, -1.0 if kwargs["pitch_type"] == 20 else 0.0),
            }

    monkeypatch.setattr(usage, "SequencingEngine", FakeEngine)

    result = usage.simulate_optimal_usage(pitch_map, pitcher="P1", season=2024, seed=7)

    assert {call["pitcher_id"] for call in calls} == {0}
    assert {call["pitch_type"] for call in calls} == {10}
    assert result["pitch_counts"] == {10: 6}
    assert result["outcome_counts"]["called_strike"] == 6
    assert result["terminal_counts"]["strikeout"] == 2


def test_simulate_optimal_usage_selects_by_mean_and_uses_selected_candidate(
    monkeypatch,
):
    import polars as pl

    pitch_map = pl.DataFrame(
        {
            "pitcher_name": ["P1", "P1"],
            "game_year": [2024, 2024],
            "pitcher_year": [0, 0],
            "pitch_name": [10, 20],
            "batter_name": ["B1", "B1"],
            "batter_year": [0, 0],
        }
    )
    calls = []

    class FixedRNG:
        def choice(self, values, size=None, replace=True, p=None):
            if size is not None:
                return np.repeat(values[0], size)
            return 0

        def integers(self, _high):
            return 7

        def random(self):
            return 0.0

    class FakeEngine:
        def __init__(self, _pitch_map):
            self.df = pitch_map
            self.cvae = torch.nn.Linear(1, 1)

        def score_pitch_candidates(self, **kwargs):
            calls.append(
                {
                    "pitch_type": kwargs["pitch_type"],
                    "prior": None
                    if kwargs["prior"] is None
                    else {
                        key: value.copy() if hasattr(value, "copy") else value
                        for key, value in kwargs["prior"].items()
                    },
                }
            )
            pitch_type = kwargs["pitch_type"]
            candidates = torch.tensor(
                [[pitch_type, candidate_index] for candidate_index in range(10)],
                dtype=torch.float32,
            )
            whiff = np.zeros(10)
            whiff[7] = 1.0
            run_values = np.array([-100.0] + [20.0] * 9)
            if pitch_type == 20:
                run_values = np.zeros(10)
            return {
                "candidates": candidates,
                "probs": {
                    "swing": np.ones(10),
                    "whiff": whiff,
                    "foul": np.zeros(10),
                    "ball": np.zeros(10),
                    "bip": np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
                },
                "run_values": run_values,
            }

    monkeypatch.setattr(usage, "SequencingEngine", FakeEngine)
    monkeypatch.setattr(usage.np.random, "default_rng", lambda _seed: FixedRNG())

    result = usage.simulate_optimal_usage(
        pitch_map, pitcher="P1", season=2024, n_batters=1, seed=7
    )

    assert result["pitch_counts"] == {10: 0, 20: 6}
    assert result["outcome_counts"]["whiff"] == 6
    assert result["terminal_counts"]["strikeout"] == 2
    prior_calls = [call for call in calls if call["prior"] is not None]
    assert prior_calls
    assert prior_calls[0]["prior"]["pitchEmb"].tolist() == [[20.0, 7.0]]


def test_simulate_optimal_usage_whiff_at_two_strikes_is_terminal(monkeypatch):
    import polars as pl

    pitch_map = pl.DataFrame(
        {
            "pitcher_name": ["P1"],
            "game_year": [2024],
            "pitcher_year": [0],
            "pitch_name": [10],
            "batter_name": ["B1"],
            "batter_year": [0],
        }
    )

    class FakeEngine:
        def __init__(self, _pitch_map):
            self.df = pitch_map
            self.cvae = torch.nn.Linear(1, 1)

        def score_pitch_candidates(self, **kwargs):
            return {
                "candidates": torch.zeros(10, 2),
                "probs": {
                    "swing": np.ones(10),
                    "whiff": np.ones(10),
                    "foul": np.zeros(10),
                    "ball": np.zeros(10),
                    "bip": np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
                },
                "run_values": np.zeros(10),
            }

    monkeypatch.setattr(usage, "SequencingEngine", FakeEngine)
    result = usage.simulate_optimal_usage(pitch_map, "P1", 2024, seed=7)

    assert result["outcome_counts"]["whiff"] == 6
    assert result["terminal_counts"] == {
        "strikeout": 2,
        "walk": 0,
        "fb": 0,
        "ld": 0,
        "gb": 0,
        "popup": 0,
    }


def test_simulate_optimal_usage_resolves_called_strikes_and_counts_zero_use_pitches(
    monkeypatch,
):
    import polars as pl

    pitch_map = pl.DataFrame(
        {
            "pitcher_name": ["P1"],
            "game_year": [2024],
            "pitcher_year": [7],
            "pitch_name": [10],
            "batter_name": ["B1"],
            "batter_year": [11],
        }
    )

    class FakeEngine:
        def __init__(self, _pitch_map):
            self.df = pitch_map
            self.cvae = torch.nn.Linear(1, 1)

        def score_pitch_candidates(self, **kwargs):
            return {
                "candidates": torch.zeros(10, 2),
                "probs": {
                    "swing": np.zeros(10),
                    "whiff": np.zeros(10),
                    "foul": np.zeros(10),
                    "ball": np.zeros(10),
                    "bip": np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
                },
                "run_values": np.zeros(10),
            }

    monkeypatch.setattr(usage, "SequencingEngine", FakeEngine)
    result = usage.simulate_optimal_usage(pitch_map, "P1", 2024, seed=7)

    assert result["pitch_counts"] == {10: 6}
    assert result["outcome_counts"]["called_strike"] == 6
    assert result["terminal_counts"]["strikeout"] == 2
    assert result["terminal_rates"]["strikeout"] == 1.0
    assert result["pitcher"] == "P1"
    assert result["season"] == 2024
    assert set(result) == {
        "pitcher",
        "season",
        "pitch_counts",
        "pitch_shares",
        "outcome_counts",
        "outcome_rates",
        "terminal_counts",
        "terminal_rates",
        "n_batters",
        "n_at_bats",
        "n_pitches",
    }
    terminal_keys = {"strikeout", "walk", "fb", "ld", "gb", "popup"}
    assert set(result["terminal_counts"]) == terminal_keys
    assert set(result["terminal_rates"]) == terminal_keys
    assert "terminal_outcome_counts" not in result
    assert "terminal_outcome_rates" not in result
    assert result["terminal_counts"] == {
        "strikeout": 2,
        "walk": 0,
        "fb": 0,
        "ld": 0,
        "gb": 0,
        "popup": 0,
    }
    assert result["outcome_counts"]["fb"] == 0
    assert result["outcome_counts"]["ld"] == 0
    assert result["outcome_counts"]["gb"] == 0
    assert result["outcome_counts"]["popup"] == 0
    assert (
        result["outcome_rates"]["called_strike"]
        == result["outcome_counts"]["called_strike"] / 6
    )
    for outcome, count in result["terminal_counts"].items():
        assert result["terminal_rates"][outcome] == count / 2
    assert sum(result["pitch_counts"].values()) == result["n_pitches"] == 6
    assert result["outcome_counts"]["swing"] + result["outcome_counts"]["take"] == 6
    assert (
        result["outcome_counts"]["called_strike"] + result["outcome_counts"]["ball"]
        == result["outcome_counts"]["take"]
    )
    assert (
        result["outcome_counts"]["whiff"]
        + result["outcome_counts"]["foul"]
        + result["outcome_counts"]["fb"]
        + result["outcome_counts"]["ld"]
        + result["outcome_counts"]["gb"]
        + result["outcome_counts"]["popup"]
        == result["outcome_counts"]["swing"]
    )
    assert sum(result["terminal_counts"].values()) == result["n_at_bats"] == 2


def test_simulate_optimal_usage_samples_and_aggregates_popups(monkeypatch):
    import polars as pl

    pitch_map = pl.DataFrame(
        {
            "pitcher_name": ["P1"],
            "game_year": [2024],
            "pitcher_year": [7],
            "pitch_name": [10],
            "batter_name": ["B1"],
            "batter_year": [11],
        }
    )

    class FakeEngine:
        def __init__(self, _pitch_map):
            self.df = pitch_map
            self.cvae = torch.nn.Linear(1, 1)

        def score_pitch_candidates(self, **kwargs):
            return {
                "candidates": torch.zeros(10, 2),
                "probs": {
                    "swing": np.ones(10),
                    "whiff": np.zeros(10),
                    "foul": np.zeros(10),
                    "ball": np.zeros(10),
                    "bip": np.tile([0.0, 0.0, 0.0, 1.0], (10, 1)),
                },
                "run_values": np.zeros(10),
            }

    monkeypatch.setattr(usage, "SequencingEngine", FakeEngine)
    result = usage.simulate_optimal_usage(pitch_map, "P1", 2024, seed=7)

    assert result["outcome_counts"]["popup"] == result["n_pitches"]
    assert result["terminal_counts"]["popup"] == 2
    assert result["outcome_rates"]["popup"] == 1.0
    assert result["terminal_rates"]["popup"] == 2 / result["n_at_bats"]


def test_simulate_optimal_usage_keeps_two_strike_fouls_in_count(monkeypatch):
    import polars as pl

    pitch_map = pl.DataFrame(
        {
            "pitcher_name": ["P1"],
            "game_year": [2024],
            "pitcher_year": [7],
            "pitch_name": [10],
            "batter_name": ["B1"],
            "batter_year": [11],
        }
    )
    count_calls = []
    count_occurrences = {}

    class FakeEngine:
        def __init__(self, _pitch_map):
            self.df = pitch_map
            self.cvae = torch.nn.Linear(1, 1)

        def score_pitch_candidates(self, **kwargs):
            count = kwargs["current_count_str"]
            count_calls.append(count)
            key = (kwargs["current_atbat"], count)
            count_occurrences[key] = count_occurrences.get(key, 0) + 1
            foul = count in {"0-0", "0-1", "0-2"} and not (
                count == "0-2" and count_occurrences[key] > 1
            )
            return {
                "candidates": torch.zeros(10, 2),
                "probs": {
                    "swing": np.ones(10),
                    "whiff": np.zeros(10),
                    "foul": np.full(10, foul),
                    "ball": np.zeros(10),
                    "bip": np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
                },
                "run_values": np.zeros(10),
            }

    monkeypatch.setattr(usage, "SequencingEngine", FakeEngine)
    result = usage.simulate_optimal_usage(pitch_map, "P1", 2024, seed=7)

    assert count_calls == ["0-0", "0-1", "0-2", "0-2"] * 2
    assert result["outcome_counts"]["foul"] == 6
    assert result["terminal_counts"]["fb"] == 2


def test_simulate_optimal_usage_is_seeded_and_samples_opponents_with_replacement(
    monkeypatch,
):
    import polars as pl

    pitch_map = pl.DataFrame(
        {
            "pitcher_name": ["P1", "P1"],
            "game_year": [2024, 2024],
            "pitcher_year": [7, 7],
            "pitch_name": [10, 10],
            "batter_name": ["B1", "B2"],
            "batter_year": [11, 12],
        }
    )

    def run(seed):
        sampled_batters = []

        class FakeEngine:
            def __init__(self, _pitch_map):
                self.df = pitch_map
                self.cvae = torch.nn.Linear(1, 1)

            def score_pitch_candidates(self, **kwargs):
                if (
                    kwargs["current_atbat"] == 0
                    and kwargs["current_count_str"] == "0-0"
                ):
                    sampled_batters.append(kwargs["batter_id"])
                return {
                    "candidates": torch.zeros(10, 2),
                    "probs": {
                        "swing": np.array([0.0] * 5 + [1.0] * 5),
                        "whiff": np.ones(10),
                        "foul": np.zeros(10),
                        "ball": np.zeros(10),
                        "bip": np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
                    },
                    "run_values": np.zeros(10),
                }

        monkeypatch.setattr(usage, "SequencingEngine", FakeEngine)
        result = usage.simulate_optimal_usage(
            pitch_map,
            "P1",
            2024,
            n_batters=5,
            seed=seed,
        )
        return result, sampled_batters

    first, first_batters = run(7)
    second, second_batters = run(7)
    different, _ = run(8)

    assert first == second
    assert first_batters == second_batters
    assert first != different
    assert len(first_batters) == 5
    assert len(set(first_batters)) < len(first_batters)


def test_simulate_optimal_usage_propagates_maximum_history_error(monkeypatch):
    import polars as pl

    pitch_map = pl.DataFrame(
        {
            "pitcher_name": ["P1"],
            "game_year": [2024],
            "pitcher_year": [7],
            "pitch_name": [10],
            "batter_name": ["B1"],
            "batter_year": [11],
        }
    )

    class FakeEngine:
        def __init__(self, _pitch_map):
            self.df = pitch_map
            self.cvae = torch.nn.Linear(1, 1)

        def score_pitch_candidates(self, **kwargs):
            if kwargs["prior"] is not None:
                raise ValueError(
                    "Current pitch history is at the expectation model maximum"
                )
            return {
                "candidates": torch.zeros(10, 2),
                "probs": {
                    "swing": np.ones(10),
                    "whiff": np.zeros(10),
                    "foul": np.zeros(10),
                    "ball": np.zeros(10),
                    "bip": np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
                },
                "run_values": np.zeros(10),
            }

    monkeypatch.setattr(usage, "SequencingEngine", FakeEngine)
    with pytest.raises(ValueError, match="maximum"):
        usage.simulate_optimal_usage(pitch_map, "P1", 2024, seed=7)
