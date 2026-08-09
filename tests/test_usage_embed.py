import torch
import torch.nn.functional as F

from src.embeddings.expectation.modules.transformer import BatterExp

# --- toy dims ---
DEVICE = "cpu"
C = 4
PITCHER_SEASONS = 1000
BATTER_SEASONS = 1000
HORIZONS = (1, 2, 3, 4)
EMBED_IN_DIM = 10
COUNT_CARD = 15


def tiny_model(dropout=0.0, max_len=32):
    return BatterExp(
        dim=32,
        out_dim=C,
        nheads=2,
        layers=2,
        attn_ffn_blocks=2,
        dropout=dropout,
        max_len=max_len,
        horizons=HORIZONS,
        pitcher_dim=8,
        batter_dim=8,
        embed_dim=10,
        batter_seasons=BATTER_SEASONS,
        pitcher_seasons=PITCHER_SEASONS,
    ).to(DEVICE)


def toy_batch(B=8, T=6, seed=0):
    g = torch.Generator().manual_seed(seed)

    def to(x):
        return x.to(DEVICE)

    return {
        "embeds": to(torch.randn(B, T, EMBED_IN_DIM, generator=g)),
        "atbat": to(torch.zeros(B, T, dtype=torch.long)),
        "count": to(torch.randint(0, COUNT_CARD, (B, T), generator=g)),
        "pitcherID": to(torch.randint(0, PITCHER_SEASONS, (B, T), generator=g)),
        "batterID": to(torch.randint(0, BATTER_SEASONS, (B, T), generator=g)),
        "lengths": to(torch.full((B,), T, dtype=torch.long)),
        "labels": to(torch.randint(0, C, (B, T), generator=g)),
    }


def _forward(model, b, repmask):
    return model(
        b["embeds"],
        b["atbat"],
        b["count"],
        b["pitcherID"],
        b["batterID"],
        repmask,
        b["lengths"],
    )


def test_forward_shapes():
    model = tiny_model().eval()
    B, T = 8, 6
    b = toy_batch(B=B, T=T)
    repMask = torch.ones(
        PITCHER_SEASONS, model.out_dim, dtype=torch.bool, device=DEVICE
    )
    with torch.no_grad():
        logits, _ = _forward(model, b, repMask)
    assert logits.shape == (B, len(model.horizons), T, model.out_dim), logits.shape
    assert torch.isfinite(logits).all()


def test_prior_shape():
    model = tiny_model().eval()
    b = toy_batch(B=8)
    repMask = torch.ones(
        PITCHER_SEASONS, model.out_dim, dtype=torch.bool, device=DEVICE
    )
    with torch.no_grad():
        prior = model.get_prior(b["pitcherID"][:, 0], b["batterID"][:, 0], repMask)
    assert prior.shape == (8, model.out_dim), prior.shape


def test_overfit_single_batch():
    torch.manual_seed(0)
    model = tiny_model(dropout=0.0).train()
    repMask = torch.ones(
        PITCHER_SEASONS, model.out_dim, dtype=torch.bool, device=DEVICE
    )
    b = toy_batch()
    i1 = model.horizons.index(1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    first = last = None
    for step in range(400):
        opt.zero_grad(set_to_none=True)
        logits, _ = _forward(model, b, repMask)
        h1 = logits[:, i1]
        loss = F.cross_entropy(h1.reshape(-1, C), b["labels"].reshape(-1))
        loss.backward()
        opt.step()
        first = loss.item() if step == 0 else first
        last = loss.item()

    assert last < 0.1, f"did not overfit: start={first:.3f} end={last:.3f}"
    assert last < first / 5, f"loss barely moved: {first:.3f} -> {last:.3f}"
