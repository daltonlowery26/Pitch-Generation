import torch
import torch.nn.functional as F

from src.outcome.modules.seq_transformer import Transformer

DEVICE = "cpu"
PITCH_EMB_DIM = 10
EXP_EMB_DIM = 4
COUNT_CARD = 12
ATBAT_CARD = 8
N_PITCHER_SEASONS = 1200
N_BATTER_SEASONS = 1200


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
        output_dim=1,
        exp_proj_dim=16,
        batter_dim=8,
        n_pitcher_seasons=N_PITCHER_SEASONS,
        n_batter_seasons=N_BATTER_SEASONS,
    ).to(DEVICE)


def toy_batch(B=8, T=6, seed=0):
    g = torch.Generator().manual_seed(seed)

    def to(x):
        return x.to(DEVICE)

    return {
        "pitchEmb": to(torch.randn(B, T, PITCH_EMB_DIM, generator=g)),
        "expEmb": to(torch.randn(B, T, EXP_EMB_DIM, generator=g)),
        "pitcherID": to(torch.randint(0, N_PITCHER_SEASONS, (B, T), generator=g)),
        "batterID": to(torch.randint(0, N_BATTER_SEASONS, (B, T), generator=g)),
        "abNum": to(torch.randint(0, ATBAT_CARD, (B, T), generator=g)),
        "count": to(torch.randint(0, COUNT_CARD, (B, T), generator=g)),
        "lengths": to(torch.full((B,), T, dtype=torch.long)),
        "swing": to(torch.ones(B, T)),
        "whiff": to(torch.randint(0, 2, (B, T), generator=g).float()),
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
    logits = (
        logits.squeeze(-1) if logits.dim() == 3 and logits.size(-1) == 1 else logits
    )
    assert logits.shape == (B, T), logits.shape
    assert torch.isfinite(logits).all()


def test_overfit_single_batch():
    torch.manual_seed(0)
    model = tiny_model(dropout=0.0).train()
    b = toy_batch()
    mask = (b["whiff"] != -1) & (b["swing"] == 1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    first = last = None
    for step in range(400):
        opt.zero_grad(set_to_none=True)
        logits = _forward(model, b)
        logits = (
            logits.squeeze(-1) if logits.dim() == 3 and logits.size(-1) == 1 else logits
        )
        loss = F.binary_cross_entropy_with_logits(logits[mask], b["whiff"][mask])
        loss.backward()
        opt.step()
        first = loss.item() if step == 0 else first
        last = loss.item()

    assert last < 0.05, f"did not overfit: start={first:.3f} end={last:.3f}"
    assert last < first / 5, f"loss barely moved: {first:.3f} -> {last:.3f}"
