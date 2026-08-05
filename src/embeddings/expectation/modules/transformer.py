# %% batter expectation of pitch in different counts
import torch
import torch.nn as nn
import torch.nn.functional as F


# RoPE block
class RoPEAttentionBlock(nn.Module):
    def __init__(self, dim, nhead, dropout, max_seq_len, rope_base=5000.0):
        super().__init__()
        assert dim % nhead == 0, f"dim ({dim}) must be divisible by nhead ({nhead})"
        head_dim = dim // nhead
        assert head_dim % 2 == 0, (
            f"head_dim ({head_dim}) must be even for LLaMA-style RoPE"
        )

        self.dim = dim
        self.nhead = nhead
        self.head_dim = head_dim
        self.dropout = dropout

        # qkv projection
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)

        # norms
        self.in_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.q_norm = nn.RMSNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(head_dim, elementwise_affine=False)

        # precompute cos and sin cache
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

        # attn proj
        self.attn_proj = nn.Linear(dim, dim)

        # ffn
        self.ffn = nn.Sequential(
            nn.RMSNorm(self.dim, elementwise_affine=False),
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

        # residual dropout
        self.resid_drop = nn.Dropout(dropout)

    def _apply_rope(self, x, positions):
        cos = self.rope_cos[positions].unsqueeze(1).to(x.dtype)  # ty:ignore
        sin = self.rope_sin[positions].unsqueeze(1).to(x.dtype)  # ty:ignore
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, attn_mask, positions):
        B, T, _ = x.shape
        h = self.in_norm(x)

        qkv = (
            self.qkv(h)
            .reshape(B, T, 3, self.nhead, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = out.transpose(1, 2).reshape(B, T, self.dim)

        x = x + self.resid_drop(self.attn_proj(attn))
        x = x + self.resid_drop(self.ffn(attn))
        return x


class PlayerEmbeddings(nn.Module):
    def __init__(self, n_players, embed_dim):
        super().__init__()
        self.player_embed = nn.Embedding(n_players, embed_dim)

    def forward(self, player_id):
        return self.player_embed(player_id)


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
        self.horizonDim = 8
        self.abDim = 2
        self.countDim = 4
        self.horizons = tuple(horizons)
        self.horizonEmb = nn.Embedding(len(self.horizons), self.horizonDim)
        self.dim = dim
        self.max_len = max_len
        self.nhead = nheads
        self.out_dim = out_dim

        # pretrained
        self.pitcher_embed = PlayerEmbeddings(
            n_players=pitcher_seasons, embed_dim=self.pitcherDim
        )
        self.batter_embed = PlayerEmbeddings(
            n_players=batter_seasons, embed_dim=self.batterDim
        )

        # pitcher project
        self.pitcher_proj = nn.Sequential(
            nn.RMSNorm(self.pitcherDim),
            nn.Linear(self.pitcherDim, self.embedDim),
        )

        # ab number
        self.abEmb = nn.Embedding(num_embeddings=6, embedding_dim=self.abDim)
        self.count_emb = nn.Embedding(20, self.countDim)

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
                RoPEAttentionBlock(
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
        ab_e = self.abEmb(abNum)
        context = torch.cat([count_e, ab_e], dim=-1)
        embed = self.context_film(embed, context)

        # input projection given pitch and pitcher
        pitcher_tok = pitcher_embed[:, :1]
        embed = torch.cat([pitcher_tok, embed], dim=1)
        embed = self.input_proj(embed)

        # per-batch real positions 0..T-1
        positions = torch.arange(Tp, device=embed.device).unsqueeze(0).expand(B, -1)

        # causal and padding masks, True == attend
        causal = ~torch.triu(
            torch.ones(Tp, Tp, device=embed.device, dtype=torch.bool), diagonal=1
        )
        # L + 1 real tokens per sequence once the pitcher token is prepended
        key_valid = torch.arange(Tp, device=embed.device).unsqueeze(0) < (
            lengths + 1
        ).unsqueeze(1)
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
        logits = []
        for i in range(len(self.horizons)):
            h = hemb[i].view(1, 1, -1).expand(B, T, -1)
            feat = torch.cat([embed, batter_embed, h], dim=-1)
            logits.append(self.out(feat))

        logits = torch.stack(logits, dim=1)
        return logits, embed
