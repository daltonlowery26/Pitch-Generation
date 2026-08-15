import numpy as np

from src.utils.plotting import plot_run_values


def test_plot_run_values_does_not_sample_supplied_real_scenarios(monkeypatch):
    calls = []

    class FakeEngine:
        def __init__(self):
            pass

        def evaluate(self, scenario, *, sample_sequence=False):
            calls.append((scenario, sample_sequence))
            return {
                "FF": np.array([0.1, 0.2]),
                "probs": {},
                "prior_info": None,
                "count_str": "0-0",
                "pitch_labels": {"FF": "Fastball"},
            }

    monkeypatch.setattr(plot_run_values, "SequencingEngine", FakeEngine)
    monkeypatch.setattr(plot_run_values.plt, "show", lambda: None)

    scenario = {
        "pitcher": "P1",
        "pitcher_year": 2024,
        "batter": "B1",
        "batter_year": 2024,
        "prior": {"pitchEmb": [[1.0] * 10]},
    }

    plot_run_values.plot_run_values(scenarios=[scenario])

    assert calls == [(scenario, False)]
