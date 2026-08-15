import numpy as np
import pytest

from src.sequencing import seq_pitch


def test_optimal_pitch_returns_bip_probabilities_in_canonical_order(monkeypatch):
    canonical_bip_probs = np.array(
        [
            [0.10, 0.30, 0.20, 0.40],
            [0.50, 0.25, 0.15, 0.10],
        ]
    )

    class FakeSequencingEngine:
        def __init__(self, _pitch_map):
            pass

        def evaluate(self, request, *, sample_sequence=False):
            assert sample_sequence is True
            assert request["prior_sequence"] == [(1, "strike")]
            return {
                "FF": np.array([0.0, 1.0]),
                "probs": {
                    "FF": {
                        "swing": np.array([0.2, 0.4]),
                        "whiff": np.array([0.1, 0.3]),
                        "foul": np.array([0.05, 0.15]),
                        "ball": np.array([0.6, 0.8]),
                        "bip": canonical_bip_probs,
                    }
                },
                "prior_info": None,
                "count_str": "0-0",
                "pitch_labels": {},
            }

    monkeypatch.setattr(seq_pitch, "SequencingEngine", FakeSequencingEngine)

    _, _, _, probs = seq_pitch.optimal_pitch(
        pitch_map="ignored",
        pitcher="P1",
        pitcher_year=2024,
        batter="B1",
        batter_year=2024,
        prior_sequence=[(1, "strike")],
    )

    assert probs["FF"]["bip"] == pytest.approx([0.30, 0.275, 0.175, 0.25])
    assert probs["FF"]["ball"] == pytest.approx(0.7)
