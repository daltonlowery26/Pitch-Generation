from datetime import date

import polars as pl
import pytest
import torch

from src.outcome.stacked.modules.generate_logits import (
    assemble_base_probabilities,
    flatten_sequence_output,
)
from src.outcome.stacked.modules.stacked_data import StackedData, split_loaders
from src.outcome.stacked.modules.stacked_model import StackedCalibrator, brier_loss


def test_stacked_calibrator_returns_task_head_shapes():
    model = StackedCalibrator(inp=9, dim=16, blocks=1).eval()
    inputs = torch.rand(4, 9)

    with torch.no_grad():
        swing, ball, whiff, foul, bip = model(inputs)

    assert swing.shape == (4, 1)
    assert ball.shape == (4, 1)
    assert whiff.shape == (4, 1)
    assert foul.shape == (4, 1)
    assert bip.shape == (4, 5)


def test_brier_loss_uses_conditional_ground_truth_masks():
    outputs = tuple(torch.zeros(4, width) for width in (1, 1, 1, 1, 5))
    batch = {
        "swing": torch.tensor([0, 1, 1, 1]),
        "ball": torch.tensor([0, 1, 0, 1]),
        "whiff": torch.tensor([0, 1, 0, 0]),
        "foul": torch.tensor([0, 0, 1, 0]),
        "bip": torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ],
            dtype=torch.float32,
        ),
    }

    loss = brier_loss(outputs, batch)
    binary = torch.tensor(0.25)
    bip = torch.tensor(((0.2 - 1) ** 2 + 4 * 0.2**2) / 5)

    assert torch.isclose(
        loss, torch.stack([binary, binary, binary, binary, bip]).mean()
    )


def stacked_frame(n=3):
    return pl.DataFrame(
        {
            "game_date": ["2025-06-01", "2025-06-02", "2025-07-02"][:n],
            "swing_prob": [0.1, 0.2, 0.3][:n],
            "ball_prob": [0.4, 0.5, 0.6][:n],
            "whiff_prob": [0.7, 0.8, 0.9][:n],
            "foul_prob": [0.2, 0.3, 0.4][:n],
            "bip_prob_0": [0.1, 0.2, 0.3][:n],
            "bip_prob_1": [0.2, 0.3, 0.4][:n],
            "bip_prob_2": [0.3, 0.4, 0.5][:n],
            "bip_prob_3": [0.2, 0.1, 0.1][:n],
            "bip_prob_4": [0.2, 0.0, 0.0][:n],
            "swing_label": [0, 1, 1][:n],
            "ball_label": [1, 0, 0][:n],
            "whiff_label": [0, 1, 0][:n],
            "foul_label": [0, 0, 0][:n],
            "bip_label_0": [0, 0, 0][:n],
            "bip_label_1": [0, 0, 0][:n],
            "bip_label_2": [0, 0, 1][:n],
            "bip_label_3": [0, 0, 0][:n],
            "bip_label_4": [0, 0, 0][:n],
        }
    ).with_columns(pl.col("game_date").str.to_date())


def test_stacked_data_returns_flat_batch_fields():
    item = StackedData(stacked_frame())[1]

    assert item["probs"].shape == (9,)
    assert item["probs"].dtype == torch.float32
    assert item["bip"].shape == (5,)
    assert set(item) == {"probs", "swing", "ball", "whiff", "foul", "bip"}


def test_split_loaders_keep_test_after_cutoff():
    train_loader, val_loader, test_loader = split_loaders(
        stacked_frame(), date(2025, 7, 1), batch_size=2, prefetch=4, workers=0
    )

    assert len(train_loader.dataset) + len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 1


def test_flatten_sequence_output_discards_padding():
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    row_idx = torch.tensor([[10, 11, 12], [20, 21, -1]])
    lengths = torch.tensor([3, 2])

    rows, values = flatten_sequence_output(logits, row_idx, lengths)

    assert torch.equal(rows, torch.tensor([10, 11, 12, 20, 21]))
    assert torch.equal(values, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_assemble_base_probabilities_joins_predictions_and_labels(tmp_path):
    frame = pl.DataFrame(
        {
            "game_date": ["2025-06-01", "2025-06-02"],
            "description": ["called_strike", "ball"],
            "swing": [False, False],
            "whiff": [False, False],
            "foul": [False, False],
            "bipArr": [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]],
        }
    ).with_columns(pl.col("game_date").str.to_date())
    swing_path = tmp_path / "swing_probs.parquet"
    pl.DataFrame({"row_idx": [0, 1], "swing_prob": [0.2, 0.3]}).write_parquet(
        swing_path
    )
    ball_path = tmp_path / "ball_probs.parquet"
    pl.DataFrame({"row_idx": [0, 1], "ball_prob": [0.4, 0.5]}).write_parquet(ball_path)

    output = tmp_path / "base_probs.parquet"
    assemble_base_probabilities(
        frame,
        {"swing": swing_path, "ball": ball_path},
        output,
    )
    result = pl.read_parquet(output).sort("game_date")

    assert result["swing_prob"].dtype == pl.Float32
    assert result["ball_label"].to_list() == [False, True]
    assert result["bip_label_0"].to_list() == [1, 0]
    assert result["bip_label_4"].to_list() == [0, 1]
    assert result["swing_prob"].to_list() == pytest.approx([0.2, 0.3])
