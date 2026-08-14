import polars as pl
import torch
from torch import nn

from src.outcome.event_probablity import ball_prob
from src.outcome.modules.data.ball_HBP_data import BallHBPData
from src.outcome.modules.models.mlp import MLP
from src.sequencing.module import utils
from src.utils.modeling.config import load_config


def test_ball_probability_config_declares_ball_checkpoint():
    cfg = load_config("config/prob/ball_prob.yml")

    assert cfg["weight_path"] == "weights/probs/ball_prob.pth"


def test_ball_hbp_data_exposes_renamed_dataset_contract():
    data = pl.DataFrame(
        {
            "kmu": [[0.0] * 10],
            "label": [1.0],
        }
    )

    dataset = BallHBPData(data, "label")

    assert len(dataset) == 1
    assert dataset[0]["pitch"].shape == (10,)


def test_load_ball_prob_uses_configured_checkpoint(monkeypatch):
    cfg = {
        "arch": {"inp": 10, "out": 1, "dim": 8, "blocks": 1},
        "weight_path": "custom/ball_prob.pth",
    }
    expected = MLP(**cfg["arch"])
    loaded = {}

    def fake_load(path, **kwargs):
        loaded["path"] = path
        loaded["kwargs"] = kwargs
        return {"model_state_dict": expected.state_dict()}

    monkeypatch.setattr(utils.torch, "load", fake_load)

    model = utils.load_ball_prob(cfg)

    assert loaded == {
        "path": "custom/ball_prob.pth",
        "kwargs": {"map_location": utils.device, "weights_only": True},
    }
    assert all(
        torch.equal(actual.cpu(), expected_value.cpu())
        for actual, expected_value in zip(
            model.state_dict().values(), expected.state_dict().values(), strict=True
        )
    )
    assert not model.training


def test_ball_training_saves_existing_checkpoint_format_to_configured_path(monkeypatch):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    saved = {}

    def fake_forward_pass(model, _batch, _loss_fn, label_smoothing):
        parameter = next(model.parameters())
        return model(torch.ones(1, 1, device=parameter.device)).sum()

    def fake_save(checkpoint, path):
        saved["checkpoint"] = checkpoint
        saved["path"] = path

    monkeypatch.setattr(ball_prob, "forward_pass", fake_forward_pass)
    monkeypatch.setattr(ball_prob.torch, "save", fake_save)

    ball_prob.train(
        model=model,
        trainloader=[{}],
        valloader=[{}],
        epochs=1,
        label_smoothing=0.0,
        lossfn=nn.BCEWithLogitsLoss(),
        adam_opti=optimizer,
        muon_opti=optimizer,
        adam_sched=scheduler,
        muon_sched=scheduler,
        weight_path="custom/ball_prob.pth",
    )

    assert saved["path"] == "custom/ball_prob.pth"
    assert set(saved["checkpoint"]) == {"model_state_dict"}
