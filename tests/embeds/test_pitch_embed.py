import torch
import torch.nn.functional as F

from src.embeddings.pitch.modules.cvae import CVAE, cvae_loss

DEVICE = "cpu"
INPUT_DIM = 10
LATENT_DIM = 8
N_PITCHES = 20
N_PITCHERS = 1000


def tiny_model(dropout=0.0):
    return CVAE(
        input_dim=INPUT_DIM,
        pitch_dim=8,
        pitcher_dim=8,
        h_dim=64,
        latent_dim=LATENT_DIM,
        dropout=dropout,
        n_layers=2,
        n_pitchers=N_PITCHERS,
        n_pitches=N_PITCHES,
    ).to(DEVICE)


def toy_batch(B=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(B, INPUT_DIM, generator=g).to(DEVICE),
        "pitch": torch.randint(0, N_PITCHES, (B,), generator=g).to(DEVICE),
        "pitcher": torch.randint(0, N_PITCHERS, (B,), generator=g).to(DEVICE),
    }


def _forward(model, b):
    return model(b["x"], b["pitcher"], b["pitch"])


def test_forward_shapes():
    model = tiny_model().eval()
    B = 16
    b = toy_batch(B=B)
    with torch.no_grad():
        recon, mu, logvar, prior_mu, prior_logvar = _forward(model, b)
    assert recon.shape == (B, INPUT_DIM), recon.shape
    assert mu.shape == (B, LATENT_DIM), mu.shape
    assert logvar.shape == (B, LATENT_DIM), logvar.shape
    assert prior_mu.shape == (B, LATENT_DIM), prior_mu.shape
    assert prior_logvar.shape == (B, LATENT_DIM), prior_logvar.shape
    for t in (recon, mu, logvar, prior_mu, prior_logvar):
        assert torch.isfinite(t).all()


def test_loss_is_scalar():
    model = tiny_model().eval()
    b = toy_batch()
    recon, mu, logvar, prior_mu, prior_logvar = _forward(model, b)
    loss = cvae_loss(
        recon,
        b["x"],
        mu,
        logvar,
        F.mse_loss,
        beta=0.4,
        mu_p=prior_mu,
        logvar_p=prior_logvar,
    )
    assert loss.ndim == 0 and torch.isfinite(loss), loss


def test_prior_depends_on_condition():
    model = tiny_model().eval()
    cond_a = torch.zeros(1, model.h_dim)
    cond_b = torch.ones(1, model.h_dim)
    with torch.no_grad():
        model.prior_mu.weight.fill_(1.0)
        model.prior_sigma.weight.fill_(1.0)
        prior_mu_a, prior_logvar_a = model.get_prior(cond_a)
        prior_mu_b, prior_logvar_b = model.get_prior(cond_b)
    assert not torch.equal(prior_mu_a, prior_mu_b)
    assert not torch.equal(prior_logvar_a, prior_logvar_b)


def test_overfit_single_batch():
    torch.manual_seed(0)
    model = tiny_model(dropout=0.0).train()
    b = toy_batch()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    first_mse = last_mse = None
    for step in range(500):
        opt.zero_grad(set_to_none=True)
        recon, mu, logvar, prior_mu, prior_logvar = _forward(model, b)
        loss = cvae_loss(
            recon,
            b["x"],
            mu,
            logvar,
            F.mse_loss,
            beta=1e-4,
            mu_p=prior_mu,
            logvar_p=prior_logvar,
        )
        loss.backward()
        opt.step()
        with torch.no_grad():
            mse = F.mse_loss(recon, b["x"]).item()
        first_mse = mse if step == 0 else first_mse
        last_mse = mse

    assert last_mse < 0.05, (
        f"did not overfit: start MSE={first_mse:.3f} end={last_mse:.3f}"
    )
    assert last_mse < first_mse / 5, (
        f"recon barely moved: {first_mse:.3f} -> {last_mse:.3f}"
    )
