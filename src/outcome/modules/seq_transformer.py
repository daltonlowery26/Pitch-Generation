# %%
import torch
from torch import nn

from src.utils.modeling.rope import RoPE


# helper functions for transformer
class FiLM(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.film = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        h = self.norm(x)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        layers,
        dropout,
        nhead,
        output_dim,
        countdim,
        abdim,
        embed_dim,
        exp_dim,
        exp_proj_dim,
        pitcher_dim,
        batter_dim,
        n_pitcher_seasons,
        n_batter_seasons,
        max_seq_len=512,
    ):
        super().__init__()

        # model dim
        self.input = embed_dim
        self.dim = dim
        self.nhead = nhead
        self.batter_dim = batter_dim
        self.abdim = abdim
        self.countdim = countdim
        self.expdim = exp_dim

        # projecting pitcher exp
        self.expectation_proj = nn.Linear(exp_dim, exp_proj_dim)

        # batter and pitcher
        self.pitcher = nn.Embedding(n_pitcher_seasons, embed_dim)
        self.batter = nn.Embedding(n_batter_seasons, batter_dim)

        # context embeddings
        self.count_emb = nn.Embedding(16, self.countdim)
        self.ab_emb = nn.Embedding(8, self.abdim)

        # FILM modulation
        cont = self.countdim + self.abdim
        self.contextFiLM = FiLM(self.input, cont)

        # input projection
        self.input_proj = nn.Sequential(
            nn.RMSNorm(self.input),
            nn.Linear(self.input, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

        # stack of RoPE, Flash Attention, FFN
        self.transformer = nn.ModuleList(
            [
                RoPE(
                    dim=dim,
                    nhead=nhead,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(layers)
            ]
        )

        o_layer = dim + batter_dim + exp_proj_dim
        self.out = nn.Sequential(
            nn.RMSNorm(o_layer),
            nn.Linear(o_layer, dim * 4),
            nn.SiLU(),
            nn.RMSNorm(dim * 4, elementwise_affine=True),
            nn.Linear(dim * 4, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, output_dim),
        )

    def forward(
        self,
        pitchEmb,
        expEmb,
        pitcher_id,
        batter_id,
        atbat,
        count,
        lengths,
    ):
        B, T, _ = pitchEmb.shape
        Tp = T + 1

        # expectation of pitch to be thrown
        expEmb = self.expectation_proj(expEmb)

        # pitcher and batter embeddings
        pitcherEmb = self.pitcher(pitcher_id)
        batterEmb = self.batter(batter_id)

        # shift pitch based on context
        count_e = self.count_emb(count)
        atbat_e = self.ab_emb(atbat)
        context = torch.concat([count_e, atbat_e], dim=-1)
        pitchEmb = self.contextFiLM(pitchEmb, context)

        # embedding, project
        pitcher_tok = pitcherEmb[:, :1]
        embed = torch.cat([pitcher_tok, pitchEmb], dim=1)
        embed = self.input_proj(embed)

        # per-batch real positions 0...T-1
        positions = torch.arange(Tp, device=embed.device).unsqueeze(0).expand(B, -1)

        # causal and padding masks
        causal = ~torch.triu(
            torch.ones(Tp, Tp, device=embed.device, dtype=torch.bool), diagonal=1
        )
        key_valid = (
            torch.arange(Tp, device=embed.device).unsqueeze(0)
            < lengths.unsqueeze(1) + 1
        )
        attn_mask = causal[None, None] & key_valid[:, None, None, :]

        # pass through transformer blocks
        for block in self.transformer:
            embed = embed + block(embed, attn_mask, positions)

        # cut out pitcher token
        embed = embed[:, 1:]

        # add batter and batter exp to pitch input
        embed = torch.cat([embed, batterEmb, expEmb], dim=-1)
        preds = self.out(embed).squeeze(-1)

        return preds
