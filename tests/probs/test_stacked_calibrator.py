import torch

from src.outcome.stacked.modules.stacked_model import StackedCalibrator, brier_loss


def test_stacked_calibrator_returns_task_head_shapes():
    model = StackedCalibrator(inp=8, dim=16, blocks=1).eval()
    inputs = torch.rand(4, 8)

    with torch.no_grad():
        swing, ball, whiff, foul, bip = model(inputs)

    assert swing.shape == (4, 1)
    assert ball.shape == (4, 1)
    assert whiff.shape == (4, 1)
    assert foul.shape == (4, 1)
    assert bip.shape == (4, 4)


def test_brier_loss_uses_conditional_ground_truth_masks():
    outputs = tuple(torch.zeros(4, width) for width in (1, 1, 1, 1, 4))
    batch = {
        "swing": torch.tensor([0, 1, 1, 1]),
        "ball": torch.tensor([0, 1, 0, 1]),
        "whiff": torch.tensor([0, 1, 0, 0]),
        "foul": torch.tensor([0, 0, 1, 0]),
        "bip": torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        ),
    }

    loss = brier_loss(outputs, batch)
    binary = torch.tensor(0.25)
    bip = torch.tensor(((0.25 - 1) ** 2 + 3 * 0.25**2) / 4)

    assert torch.isclose(
        loss, torch.stack([binary, binary, binary, binary, bip]).mean()
    )
