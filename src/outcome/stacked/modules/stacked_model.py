import torch
from torch import nn

from src.outcome.modules.models.mlp import MLP


class StackedCalibrator(nn.Module):
    def __init__(self, inp, dim, blocks):
        super().__init__()
        self.trunk = MLP(inp=inp, out=dim, dim=dim, blocks=blocks)
        self.swing = nn.Linear(dim, 1)
        self.ball = nn.Linear(dim, 1)
        self.whiff = nn.Linear(dim, 1)
        self.foul = nn.Linear(dim, 1)
        self.bip = nn.Linear(dim, 5)

    def forward(self, probs):
        features = self.trunk(probs)
        return (
            self.swing(features),
            self.ball(features),
            self.whiff(features),
            self.foul(features),
            self.bip(features),
        )


def brier_loss(outputs, batch):
    swing_logits, ball_logits, whiff_logits, foul_logits, bip_logits = outputs
    swing = batch["swing"].float()
    ball = batch["ball"].float()
    whiff = batch["whiff"].float()
    foul = batch["foul"].float()
    bip = batch["bip"].float()

    swing_loss = (torch.sigmoid(swing_logits).squeeze(-1) - swing).square().mean()
    ball_loss = (
        (torch.sigmoid(ball_logits).squeeze(-1)[swing == 0] - ball[swing == 0])
        .square()
        .mean()
    )
    whiff_loss = (
        (torch.sigmoid(whiff_logits).squeeze(-1)[swing == 1] - whiff[swing == 1])
        .square()
        .mean()
    )
    contact = (swing == 1) & (whiff == 0)
    foul_loss = (
        (torch.sigmoid(foul_logits).squeeze(-1)[contact] - foul[contact])
        .square()
        .mean()
    )
    fair = contact & (foul == 0)
    bip_loss = (torch.softmax(bip_logits, dim=-1)[fair] - bip[fair]).square().mean()

    return (swing_loss + ball_loss + whiff_loss + foul_loss + bip_loss) / 5
