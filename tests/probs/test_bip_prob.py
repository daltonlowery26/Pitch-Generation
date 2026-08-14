import torch
import torch.nn.functional as F

from src.outcome.event_probablity import bip_type
from src.outcome.modules.models.transformer import Transformer

DEVICE = "cpu"
PITCH_EMB_DIM = 10
EXP_EMB_DIM = 4
COUNT_CARD = 12
ATBAT_CARD = 8
N_PITCHER_SEASONS = 1200
N_BATTER_SEASONS = 1200
BIP_CLASSES = 4


def tiny_model(dropout=0.0):
    return Transformer(
        dim=32,
        layers=2,
        dropout=dropout,
        nhead=2,
        attn_fnn_block=2,
        embed_dim=PITCH_EMB_DIM,
        countdim=4,
        exp_dim=EXP_EMB_DIM,
        abdim=4,
        output_dim=BIP_CLASSES,
        exp_proj_dim=16,
        batter_dim=8,
        n_pitcher_seasons=N_PITCHER_SEASONS,
        n_batter_seasons=N_BATTER_SEASONS,
    ).to(DEVICE)


def toy_batch(B=8, T=6, seed=0):
    g = torch.Generator().manual_seed(seed)

    def to(x):
        return x.to(DEVICE)

    bip_classes = torch.randint(0, BIP_CLASSES, (B, T), generator=g)
    return {
        "pitchEmb": to(torch.randn(B, T, PITCH_EMB_DIM, generator=g)),
        "expEmb": to(torch.randn(B, T, EXP_EMB_DIM, generator=g)),
        "pitcherID": to(torch.randint(0, N_PITCHER_SEASONS, (B, T), generator=g)),
        "batterID": to(torch.randint(0, N_BATTER_SEASONS, (B, T), generator=g)),
        "abNum": to(torch.randint(0, ATBAT_CARD, (B, T), generator=g)),
        "count": to(torch.randint(0, COUNT_CARD, (B, T), generator=g)),
        "lengths": to(torch.full((B,), T, dtype=torch.long)),
        "swing": to(torch.ones(B, T)),
        "whiff": to(torch.zeros(B, T)),
        "foul": to(torch.zeros(B, T)),
        "bipLabel": to(F.one_hot(bip_classes, num_classes=BIP_CLASSES)),
    }


def _forward(model, b):
    return model(
        b["pitchEmb"],
        b["expEmb"],
        b["pitcherID"],
        b["batterID"],
        b["abNum"],
        b["count"],
        b["lengths"],
    )


def test_forward_shapes():
    model = tiny_model().eval()
    B, T = 8, 6
    b = toy_batch(B=B, T=T)
    with torch.no_grad():
        logits = _forward(model, b)
    assert logits.shape == (B, T, BIP_CLASSES), logits.shape
    assert torch.isfinite(logits).all()


def test_forward_masks_non_bip_and_padding_rows():
    batch = toy_batch(B=1, T=6)
    batch["bipLabel"][0, 0] = 0
    batch["bipLabel"][0, 1] = -1
    batch["swing"][0, 2] = 0
    batch["whiff"][0, 3] = 1
    batch["foul"][0, 4] = 1

    class DummyModel:
        def __call__(self, *args):
            return torch.zeros(1, 6, BIP_CLASSES)

    _, _, mask = bip_type.forward(batch, DummyModel(), DEVICE)

    assert mask.tolist() == [[False, False, False, False, False, True]]


def test_cross_entropy_uses_one_hot_class_indices():
    logits = torch.randn(3, BIP_CLASSES)
    labels = F.one_hot(torch.tensor([2, 0, 1]), num_classes=BIP_CLASSES)
    mask = torch.tensor([True, False, True])
    captured = {}

    def loss_fn(actual_logits, targets):
        captured["targets"] = targets
        return F.cross_entropy(actual_logits, targets)

    bip_type.compute_loss(logits, labels, mask, loss_fn)

    assert captured["targets"].tolist() == [2, 1]


def test_overfit_single_batch():
    torch.manual_seed(0)
    model = tiny_model(dropout=0.0).train()
    b = toy_batch()
    mask = (
        (b["bipLabel"].sum(dim=-1) > 0)
        & (b["bipLabel"] != -1).all(dim=-1)
        & (b["swing"] == 1)
        & (b["whiff"] != 1)
        & (b["foul"] != 1)
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    first = last = None
    for step in range(400):
        opt.zero_grad(set_to_none=True)
        logits = _forward(model, b)
        loss = F.cross_entropy(logits[mask], b["bipLabel"][mask].argmax(dim=-1))
        loss.backward()
        opt.step()
        first = loss.item() if step == 0 else first
        last = loss.item()

    assert last < 0.05, f"did not overfit: start={first:.3f} end={last:.3f}"
    assert last < first / 5, f"loss barely moved: {first:.3f} -> {last:.3f}"
