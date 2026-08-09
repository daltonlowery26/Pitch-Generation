import torch
import torch.nn.functional as F
from torch import nn

from .opti import tag


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


# RoPE block
class RoPE(nn.Module):
    def __init__(
        self, dim, nhead, block_layers, dropout, max_seq_len, rope_base=5000.0
    ):
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
        tag(self.qkv, no_muon=True)

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
        self.attn_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        # ffn
        ffn_layers = [
            nn.Sequential(
                nn.RMSNorm(self.dim, elementwise_affine=False),
                nn.Linear(self.dim, self.dim * 4),
                nn.SiLU(),
            ),
        ]

        for _ in range(block_layers):
            ffn_layers.append(
                Residual(
                    nn.Sequential(
                        nn.RMSNorm(self.dim * 4),
                        nn.Linear(self.dim * 4, self.dim * 4),
                        nn.SiLU(),
                    )
                )
            )

        out = nn.Linear(self.dim * 4, self.dim)
        tag(out, no_muon=True)
        ffn_layers.append(out)
        self.ffn = nn.Sequential(*ffn_layers)

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

        attn = self.attn_proj(attn)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = attn + self.ffn(attn)
        return out
