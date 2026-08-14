from unittest.mock import Mock, call, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.utils.plotting.plot_run_values import plot_run_values


def _scenario(observed_pitch="FF", observed_run_value=0.25, count="1-2"):
    return {
        "pitcher": "Pitcher",
        "pitcher_year": 2025,
        "batter": "Batter",
        "batter_year": 2025,
        "observed_pitch": observed_pitch,
        "observed_run_value": observed_run_value,
        "prior_sequence": [],
        "count_str": count,
    }


def _evaluation(scenario):
    return {
        "FF": np.array([-0.4, 0.0, 0.2, 0.4]),
        "SL": np.array([-0.1, 0.1, 0.3, 0.5]),
        "pitch_labels": {"FF": "Fastball", "SL": "Slider"},
        "count_str": scenario.get("count_str", "0-0"),
        "probs": {},
    }


def test_plots_explicit_scenarios_with_density_histograms_and_observed_marker():
    scenarios = [_scenario(), _scenario(observed_pitch="SL", observed_run_value=-0.2)]
    engine = Mock()
    engine.evaluate.side_effect = _evaluation

    with (
        patch(
            "src.utils.plotting.plot_run_values.SequencingEngine", return_value=engine
        ),
        patch.object(
            Figure,
            "savefig",
            side_effect=AssertionError("plot_run_values must not save figures"),
        ),
        patch(
            "src.utils.plotting.plot_run_values.plt.savefig",
            side_effect=AssertionError("plot_run_values must not save figures"),
        ),
        patch(
            "src.utils.plotting.plot_run_values.plt.show",
            side_effect=AssertionError("plot_run_values must not show figures"),
        ),
    ):
        figure = plot_run_values("pitch-map.parquet", scenarios=scenarios, bins=4)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 3
    assert not figure.axes[-1].get_visible()
    assert engine.evaluate.call_args_list[0].args == (scenarios[0],)
    assert engine.evaluate.call_args_list[1].args == (scenarios[1],)
    assert len(figure.axes[0].patches) == 8
    assert all(
        isinstance(patch, Rectangle) and patch.get_height() >= 0
        for patch in figure.axes[0].patches
    )
    for axis in figure.axes[: len(scenarios)]:
        for histogram in axis.containers:
            rectangles = [patch for patch in histogram if isinstance(patch, Rectangle)]
            if rectangles:
                assert sum(
                    rectangle.get_height() * rectangle.get_width()
                    for rectangle in rectangles
                ) == pytest.approx(1.0)
    assert figure.axes[0].get_lines()[0].get_linestyle() == "--"
    assert figure.axes[0].get_lines()[0].get_label() == "Observed Fastball"
    for axis, scenario in zip(figure.axes, scenarios):
        observed_lines = [
            line for line in axis.get_lines() if line.get_linestyle() == "--"
        ]
        assert len(observed_lines) == 1
        assert np.asarray(observed_lines[0].get_xdata())[0] == pytest.approx(
            scenario["observed_run_value"]
        )
    assert "Pitcher (2025)" in figure.axes[0].get_title()
    assert "Batter (2025)" in figure.axes[0].get_title()
    assert "1-2" in figure.axes[0].get_title()

    plt.close(figure)


def test_samples_scenarios_using_filters_and_hides_unused_axes():
    scenarios = [
        _scenario(),
        _scenario(observed_pitch="SL", observed_run_value=-0.2),
        _scenario(observed_pitch="FF", observed_run_value=0.1),
        _scenario(observed_pitch="SL", observed_run_value=0.4),
    ]
    engine = Mock()
    engine.sample_scenarios.return_value = scenarios
    engine.evaluate.side_effect = _evaluation

    with patch(
        "src.utils.plotting.plot_run_values.SequencingEngine", return_value=engine
    ):
        figure = plot_run_values(
            "pitch-map.parquet",
            n_scenarios=4,
            pitcher="Pitcher",
            pitcher_year=2025,
            batter="Batter",
            batter_year=2025,
            seed=7,
        )

    engine.sample_scenarios.assert_called_once_with(
        n=4,
        pitcher="Pitcher",
        pitcher_year=2025,
        batter="Batter",
        batter_year=2025,
        seed=7,
    )
    assert len(figure.axes) == 6
    assert engine.evaluate.call_args_list == [call(scenario) for scenario in scenarios]
    assert all(axis.get_visible() for axis in figure.axes[: len(scenarios)])
    assert all(not axis.get_visible() for axis in figure.axes[len(scenarios) :])

    plt.close(figure)
