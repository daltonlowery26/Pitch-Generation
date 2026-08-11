import torch
from torch import nn

from src.evaluate.modules.discriminator import Discriminator

DEVICE = "cpu"
INPUT_DIM = 10


def tiny_model(dim=16, blocks=1):
    return Discriminator(inp=INPUT_DIM, dim=dim, blocks=blocks).to(DEVICE)


def toy_batch(B=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    embed = torch.randn(B, INPUT_DIM, generator=g).to(DEVICE)
    label = torch.randint(0, 2, (B, 1), generator=g).float().to(DEVICE)
    return {"embed": embed, "label": label}


def _forward(model, b):
    return model(b["embed"])


def test_forward_shapes():
    model = tiny_model().eval()
    B = 32
    b = toy_batch(B=B)
    with torch.no_grad():
        logits = _forward(model, b)
    assert logits.shape == (B, 1), logits.shape
    assert torch.isfinite(logits).all()


def test_overfit_single_batch():
    torch.manual_seed(0)
    model = tiny_model(dim=16, blocks=1).train()
    b = toy_batch(B=512)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    first_loss = last_loss = None
    for step in range(400):
        opt.zero_grad(set_to_none=True)
        logits = _forward(model, b)
        loss = loss_fn(logits, b["label"])
        loss.backward()
        opt.step()
        first_loss = loss.item() if step == 0 else first_loss
        last_loss = loss.item()

    assert last_loss < 0.05, (
        f"did not overfit: start loss={first_loss:.4f} end={last_loss:.4f}"
    )
    assert last_loss < first_loss / 5, (
        f"loss barely moved: {first_loss:.4f} -> {last_loss:.4f}"
    )
