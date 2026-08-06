# %% batter expectation of pitch in different counts
import torch
from torch import nn

from src.utils.modeling.opti import tag
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


class BatterExp(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        nheads,
        layers,
        dropout,
        max_len,
        horizons,
        pitcher_dim,
        batter_dim,
        embed_dim,
        batter_seasons,
        pitcher_seasons,
    ):
        super().__init__()
        # constants
        self.embedDim = embed_dim
        self.pitcherDim = pitcher_dim
        self.batterDim = batter_dim
        self.horizonDim = 4
        self.abDim = 4
        self.countDim = 4
        
        self.dim = dim
        self.nhead = nheads
        self.out_dim = out_dim
        self.max_len = max_len
        self.horizons = tuple(horizons)

        # embeddings
        self.pitcher_embed = nn.Embedding(num_embeddings=pitcher_seasons, embedding_dim=self.pitcherDim)
        self.batter_embed = nn.Embedding(num_embeddings=batter_seasons, embedding_dim=self.batterDim)
        self.horizonEmb = nn.Embedding(num_embeddings=len(self.horizons), embedding_dim=self.horizonDim)
        
        # pitcher project
        self.pitcher_proj = nn.Sequential(
            nn.RMSNorm(self.pitcherDim),
            nn.Linear(self.pitcherDim, self.embedDim),
        )

        # ab number
        self.ab_emb = nn.Embedding(num_embeddings=8, embedding_dim=self.abDim)
        self.count_emb = nn.Embedding(num_embeddings=16, embedding_dim=self.countDim)

        # learned pitcher projection for first pitch
        prior = self.pitcherDim + self.batterDim
        self.pitcher_prior = nn.Sequential(
            nn.RMSNorm(prior),
            nn.Linear(prior, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, out_dim),
        )

        # input projection
        self.input_proj = nn.Sequential(
            nn.RMSNorm(self.embedDim),
            nn.Linear(self.embedDim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

        # FiLM modulation for ab
        self.context_film = FiLM(embed_dim, self.countDim + self.abDim)

        # transformer with RoPE + flash attention
        self.transformer = nn.ModuleList(
            [
                RoPE(
                    dim=dim,
                    nhead=nheads,
                    dropout=dropout,
                    max_seq_len=max_len + 1,
                )
                for _ in range(layers)
            ]
        )
        self.w = nn.ModuleList([nn.Linear(dim, 1) for _ in range(layers)])
        for lin in self.w:
            nn.init.zeros_(lin.weight)
            nn.init.ones_(lin.bias)

        # combine pitcher rep, to out
        out = self.horizonDim + self.batterDim + dim
        self.out = nn.Sequential(
            nn.RMSNorm(out),
            nn.Linear(out, dim * 4),
            nn.SiLU(),
            nn.RMSNorm(dim * 4),
            nn.Linear(dim * 4, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, out_dim),
        )
        tag(self)

    # first pitch of atbat
    def get_prior(self, pitcherIds, batterIds, repMask):
        # batter & pitcher
        pitcher = self.pitcher_embed(pitcherIds)
        batter = self.batter_embed(batterIds)

        # predicted pitch outputs
        logits = self.pitcher_prior(torch.concat([pitcher, batter], dim=1))

        # imputed logics
        pitcher_mask = repMask[pitcherIds]
        logits = logits.masked_fill(~pitcher_mask, -1e9)
        return logits

    def forward(
        self,
        embed,
        abNum,
        count,
        pitcherID,
        batterID,
        repMask,
        lengths,
    ):
        B, T, _ = embed.shape
        Tp = T + 1

        # pitcher embedding
        pitcher_embed = self.pitcher_embed(pitcherID)
        pitcher_embed = self.pitcher_proj(pitcher_embed)

        # batter embeddings
        batter_embed = self.batter_embed(batterID)

        # create context rep, shift and scale
        count_e = self.count_emb(count)
        ab_e = self.ab_emb(abNum)
        context = torch.cat([count_e, ab_e], dim=-1)
        embed = self.context_film(embed, context)

        # input projection given pitch and pitcher
        pitcher_tok = pitcher_embed[:, :1]
        embed = torch.cat([pitcher_tok, embed], dim=1)
        embed = self.input_proj(embed)

        # per-batch real positions
        positions = torch.arange(Tp, device=embed.device).unsqueeze(0).expand(B, -1)

        # causal and padding masks, true means attend
        causal = ~torch.triu(torch.ones(Tp, Tp, device=embed.device, dtype=torch.bool), diagonal=1)
        key_valid = torch.arange(Tp, device=embed.device).unsqueeze(0) < (lengths + 1).unsqueeze(1)
        attn_mask = causal[None, None] & key_valid[:, None, None, :]

        # pass through transformer blocks
        for block, w in zip(self.transformer, self.w):
            new = block(embed, attn_mask, positions)
            embed = embed + w(new) * new

        # cut out prended pitcher tokem
        embed = embed[:, 1:]

        # predict various horizons, richer heads
        hidx = torch.arange(len(self.horizons), device=embed.device)
        hemb = self.horizonEmb(hidx)

        # loop through outputs, add logits
        pitch_mask = repMask[pitcherID]  
        logits = []
        
        for i in range(len(self.horizons)):
            h = hemb[i].view(1, 1, -1).expand(B, T, -1)
            feat = torch.cat([embed, batter_embed, h], dim=-1)
            logits.append(self.out(feat))

        logits = torch.stack(logits, dim=1) 
        # mask out impossible pitches for a given pitcher       
        logits = logits.masked_fill(~pitch_mask.unsqueeze(1), -1e9) 
        return logits, embed
