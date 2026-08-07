import torch
from torch import nn

from src.utils.modeling.opti import tag


# recon loss and kl loss
def cvae_loss(recon, x, mu_q, logvar_q, loss_fn, beta=1.0):
    rec = loss_fn(recon, x)
    kl = -0.5 * (1 + logvar_q - mu_q.pow(2) - logvar_q.exp()).sum(dim=-1)
    return (rec + beta * kl).mean()

# residual block
class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        # layer
        self.block = nn.Sequential(
            nn.RMSNorm(dim), 
            nn.Linear(dim, dim * 4), 
            nn.SiLU(), 

            nn.RMSNorm(dim * 4),
            nn.Linear(dim * 4, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return x + self.block(x)


# modulate input with respect to condtional
class FiLM(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.RMSNorm(dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.act = nn.SiLU()

        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        h = self.norm(self.fc(x))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        return x + self.act(h)


class CVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        pitcher_dim,
        pitch_dim,
        h_dim,
        latent_dim,
        dropout,
        n_layers,
        n_pitchers,
        n_pitches,
    ):
        self.latent_dim = latent_dim
        self.h_dim = h_dim

        super().__init__()
        # shared conditioning
        self.pitcherEmbed = nn.Embedding(num_embeddings=n_pitchers, embedding_dim=pitcher_dim)
        self.pitchEmbed = nn.Embedding(num_embeddings=n_pitches, embedding_dim=pitch_dim)

        torch.nn.init.constant_(self.pitcherEmbed.weight, 0.0)
        torch.nn.init.constant_(self.pitchEmbed.weight, 0.0)
        
        # condtioning modulation
        self.cond_proj = nn.Linear(pitcher_dim + pitch_dim, h_dim)
        
        # encoder  q(z | x)
        self.in_norm = nn.RMSNorm(input_dim)
        self.input = nn.Linear(input_dim, h_dim)

        self.encoder = nn.ModuleList(
            [FeedForward(h_dim, dropout) for _ in range(n_layers)]
        )
        self.mu = nn.Linear(h_dim, latent_dim)
        self.sigma = nn.Linear(h_dim, latent_dim)

        tag(self.mu, no_muon=True)
        tag(self.sigma, no_muon=True)

        # decoder  p(x | z, c)
        self.dinput = nn.Linear(latent_dim, h_dim)
        self.decoder = nn.ModuleList([FiLM(h_dim, h_dim) for _ in range(n_layers)])
        self.out = nn.Linear(h_dim, input_dim)

        tag(self.out, no_muon=True)

    def encode(self, x):
        # input normalization
        x = self.input(self.in_norm(x))
        for layer in self.encoder:
            x = layer(x)

        logvar = 10.0 * torch.tanh(self.sigma(x))
        return self.mu(x), logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, cond):
        z = self.dinput(z)
        for layer in self.decoder:
            z = layer(z, cond)
        return self.out(z)

    def forward(self, traits, pitcher, pitch):
        # generate all embeddings
        pitcher = self.pitcherEmbed(pitcher)
        pitch = self.pitchEmbed(pitch)
        cond = self.cond_proj(torch.concat([pitcher, pitch], dim=-1))

        mu_q, logvar_q = self.encode(traits)
        z = self.reparameterize(mu_q, logvar_q)
        recon = self.decode(z, cond)
        return recon, mu_q, logvar_q

    @torch.no_grad()
    def generate(self, c, n=None, temperature=1.0):
        cond = self.cond_proj(c)
        n = n if n is not None else cond.shape[0]
        z = temperature * torch.randn(n, self.latent_dim, device=cond.device)
        return self.decode(z, cond.expand(n, -1) if cond.shape[0] == 1 else cond)
