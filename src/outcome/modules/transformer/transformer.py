# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# RoPE block
class RoPEAttentionBlock(nn.Module):
    def __init__(self, dim, nhead, dropout, max_seq_len, rope_base=5000.0):
        super().__init__()
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
            nn.RMSNorm(self.dim),
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.RMSNorm(self.dim * 4),
            nn.Linear(self.dim * 4, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

    def _apply_rope(self, x, positions):
        cos = self.rope_cos[positions].unsqueeze(1).to(x.dtype)  # ty:ignore
        sin = self.rope_sin[positions].unsqueeze(1).to(x.dtype)  # ty:ignore
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, attn_mask, positions):
        B, T, _ = x.shape
        x = self.in_norm(x)

        qkv = (
            self.qkv(x)
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
            is_causal=False,
        )
        attn = out.transpose(1, 2).reshape(B, T, self.dim)

        x = x + self.attn_proj(attn)
        x = x + self.ffn(x)
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
        self.pitcher = PlayerEmbeddings(n_pitcher_seasons, pitcher_dim)
        self.batter = PlayerEmbeddings(n_batter_seasons, batter_dim)

        # context embeddings
        self.count_emb = nn.Embedding(20, self.countdim)
        self.ab_emb = nn.Embedding(10, self.abdim)

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
                RoPEAttentionBlock(
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
        expMask = torch.isfinite(expEmb) & (expEmb > -1e-8)
        safe_logits = expEmb.masked_fill(~expMask, 0.0)
        expEmb = self.expectation_proj(safe_logits)

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
