"""Per-residue flexibility predictor for EnsFlow Level 3.

Maps ``(single, pair)`` embeddings to a positive ``nu_i`` on absolute scale
that scales the Level 2 bridge widths jointly: ``sigma_i = sigma_bar * sqrt(nu_i)``.

Strict L2 ⊂ L3 nesting at init: zero-init the final Linear weight and set its
bias to ``log(e-1)``, so ``softplus(bias) == 1`` and ``nu_i ≡ 1`` regardless
of the upstream attention block.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# softplus(log(e - 1)) == 1.
_SOFTPLUS_UNITY_BIAS = math.log(math.e - 1.0)


class FlexibilityNet(nn.Module):
    def __init__(
        self,
        c_single_in: int,
        c_pair_in: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})."
            )
        self._n_heads = int(n_heads)

        self.single_ln = nn.LayerNorm(c_single_in)
        self.single_proj = nn.Linear(c_single_in, hidden_dim)

        self.pair_ln = nn.LayerNorm(c_pair_in)
        self.pair_proj = nn.Linear(c_pair_in, n_heads, bias=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            bias=True,
        )
        self.attn_ln = nn.LayerNorm(hidden_dim)

        # Zero-weight + log(e-1) bias gives nu_i ≡ 1 at init (strict L2 ⊂ L3).
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, _SOFTPLUS_UNITY_BIAS)

    def forward(
        self,
        single_embedding: torch.Tensor,
        pair_embedding: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = single_embedding.shape
        mask = res_mask.to(single_embedding.dtype)            # [B, N]
        mask_col = mask[..., None]                            # [B, N, 1]

        # Zero padding rows after LN since dataset padding may not be exact zeros.
        s = self.single_proj(self.single_ln(single_embedding))
        s = s * mask_col

        # MHA expects a per-batch, per-head attn_mask flattened to (B*h, N, N).
        bias = self.pair_proj(self.pair_ln(pair_embedding))     # [B, N, N, h]
        bias = bias.permute(0, 3, 1, 2).contiguous()            # [B, h, N, N]
        attn_bias = bias.view(B * self._n_heads, N, N)

        # Float key_padding_mask (PyTorch deprecates mixing bool + float masks).
        key_padding_mask = torch.zeros_like(mask)
        key_padding_mask = key_padding_mask.masked_fill(mask < 0.5, float("-inf"))

        attn_out, _ = self.attn(
            query=s,
            key=s,
            value=s,
            attn_mask=attn_bias,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        # MHA can return NaN on fully-masked rows; sanitize before residual add.
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        s = self.attn_ln(s + attn_out)
        s = s * mask_col

        raw = self.head(s).squeeze(-1)                           # [B, N]
        nu = F.softplus(raw)

        # 1.0 on padding so a detached nu used as a width multiplier is a no-op.
        nu = nu * mask + (1.0 - mask)
        return nu
