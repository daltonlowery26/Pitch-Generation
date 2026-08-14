import numpy as np
import polars as pl
import pytest

from src.evaluate import distribution_distance
from src.evaluate.distribution_distance import (
    energy_distance,
    estimate_bandwidth,
    pitcher_distances,
    rbf_mmd,
    summarize_distances,
)


def test_evaluator_has_no_cli_wrapper():
    assert not hasattr(distribution_distance, "main")
    assert not hasattr(distribution_distance, "_arguments")


def test_identical_samples_have_zero_distance():
    samples = np.asarray([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5]], dtype=np.float32)

    assert rbf_mmd(samples, samples.copy(), bandwidth=1.0) == pytest.approx(0.0)
    assert energy_distance(samples, samples.copy()) == pytest.approx(0.0)


def test_translation_increases_both_distances():
    observed = np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    identical = observed.copy()
    shifted = observed + np.asarray([3.0, 0.0], dtype=np.float32)

    assert rbf_mmd(observed, shifted, bandwidth=1.0) > rbf_mmd(
        observed, identical, bandwidth=1.0
    )
    assert energy_distance(observed, shifted) > energy_distance(observed, identical)


def test_rbf_detects_equal_mean_shape_change():
    observed = np.asarray(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=np.float32
    )
    changed = np.asarray(
        [[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]], dtype=np.float32
    )

    assert rbf_mmd(observed, changed, bandwidth=0.75) > 0.0


def test_bandwidth_estimation_is_seeded_and_positive():
    reference = np.arange(60, dtype=np.float32).reshape(20, 3)

    first = estimate_bandwidth(reference, max_samples=10, seed=7)
    second = estimate_bandwidth(reference, max_samples=10, seed=7)

    assert first == pytest.approx(second)
    assert first > 0.0


def test_metrics_are_finite_and_nonnegative():
    generator = np.random.default_rng(4)
    observed = generator.normal(size=(32, 4)).astype(np.float32)
    generated = generator.normal(size=(32, 4)).astype(np.float32)

    assert np.isfinite(rbf_mmd(observed, generated, bandwidth=1.0))
    assert np.isfinite(energy_distance(observed, generated))
    assert rbf_mmd(observed, generated, bandwidth=1.0) >= 0.0
    assert energy_distance(observed, generated) >= 0.0


def test_pitcher_distances_preserves_names_and_filters_small_groups():
    observed = np.asarray(
        [[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]], dtype=np.float32
    )
    generated = observed.copy()
    pitcher_year = np.asarray([10, 10, 20, 20])
    pitcher_name = np.asarray(["A. Pitcher", "A. Pitcher", "B. Pitcher", "B. Pitcher"])

    result = pitcher_distances(
        observed,
        generated,
        pitcher_year,
        pitcher_name,
        bandwidth=1.0,
        min_samples=2,
        max_samples=2,
    )

    assert result.select("pitcher_name").to_series().to_list() == [
        "A. Pitcher",
        "B. Pitcher",
    ]
    assert result["eligible"].to_list() == [True, True]
    assert result["n_used"].to_list() == [2, 2]

    ineligible = pitcher_distances(
        observed[:1],
        generated[:1],
        pitcher_year[:1],
        pitcher_name[:1],
        bandwidth=1.0,
        min_samples=2,
        max_samples=2,
    )
    assert ineligible["eligible"].to_list() == [False]
    assert ineligible["mmd_rbf"].to_list() == [None]


def test_pitcher_distances_subsampling_is_deterministic():
    values = np.arange(40, dtype=np.float32).reshape(20, 2)
    groups = np.repeat([10, 20], 10)
    names = np.where(groups == 10, "A. Pitcher", "B. Pitcher")

    first = pitcher_distances(
        values,
        values + 0.25,
        groups,
        names,
        bandwidth=1.0,
        min_samples=2,
        max_samples=4,
        seed=9,
    )
    second = pitcher_distances(
        values,
        values + 0.25,
        groups,
        names,
        bandwidth=1.0,
        min_samples=2,
        max_samples=4,
        seed=9,
    )

    assert first.equals(second)
    assert first["n_used"].to_list() == [4, 4]


def test_pitcher_distances_forwards_explicit_block_size_to_both_metrics(monkeypatch):
    calls = []

    def fake_rbf_mmd(observed, generated, bandwidth, block_size=1024):
        calls.append(("rbf_mmd", block_size))
        return 0.0

    def fake_energy_distance(observed, generated, block_size=1024):
        calls.append(("energy_distance", block_size))
        return 0.0

    monkeypatch.setattr(distribution_distance, "rbf_mmd", fake_rbf_mmd)
    monkeypatch.setattr(distribution_distance, "energy_distance", fake_energy_distance)

    pitcher_distances(
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([10, 10]),
        np.asarray(["A. Pitcher", "A. Pitcher"]),
        bandwidth=1.0,
        min_samples=1,
        max_samples=2,
        block_size=3,
    )

    assert calls == [("rbf_mmd", 3), ("energy_distance", 3)]


def test_summaries_include_macro_and_observation_weighted_scores():
    metrics = pl.DataFrame(
        {
            "pitcher_year": [10, 20, 30],
            "pitcher_name": ["A", "B", "C"],
            "n_observed": [2, 4, 8],
            "n_generated": [2, 4, 8],
            "n_used": [2, 4, 8],
            "mmd_rbf": [1.0, 3.0, None],
            "energy_distance": [2.0, 4.0, None],
            "eligible": [True, True, False],
        }
    )

    summary = summarize_distances(metrics)

    assert summary["macro_mmd_rbf"] == pytest.approx(2.0)
    assert summary["macro_energy_distance"] == pytest.approx(3.0)
    assert summary["weighted_mmd_rbf"] == pytest.approx((1.0 * 2 + 3.0 * 4) / 6)
    assert summary["weighted_energy_distance"] == pytest.approx((2.0 * 2 + 4.0 * 4) / 6)
