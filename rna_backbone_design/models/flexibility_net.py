"""Per-residue flexibility predictor for EnsFlow Level 3.

Pair-only head: flexibility is relational (who residue i is packed / paired
with), so we pool pair features over j and predict nu_i from that context.

Strict L2 subset L3 nesting at init: zero-init the final Linear weight and
set its bias to log(e-1), so softplus(bias) == 1 and nu_i identical 1
regardless of the upstream MLP.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_SOFTPLUS_UNITY_BIAS = math.log(math.e - 1.0)


class FlexibilityNet(nn.Module):
    def __init__(
        self,
        c_pair_in: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pair_ln = nn.LayerNorm(c_pair_in)
        # Pooled features: mean + max over j, plus diagonal (self).
        in_dim = 3 * c_pair_in
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, _SOFTPLUS_UNITY_BIAS)

    def forward(
        self,
        pair_embedding: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _, _ = pair_embedding.shape
        mask = res_mask.to(pair_embedding.dtype)                   # [B, N]

        p = self.pair_ln(pair_embedding)                           # [B, N, N, C]
        mask_j = mask[:, None, :, None]                            # [B, 1, N, 1]
        p_masked = p * mask_j

        counts = mask.sum(dim=-1).clamp(min=1.0)[:, None, None]    # [B, 1, 1]
        p_mean = p_masked.sum(dim=2) / counts                      # [B, N, C]

        neg_inf = torch.finfo(p.dtype).min
        p_for_max = p.masked_fill(mask_j < 0.5, neg_inf)
        p_max = p_for_max.max(dim=2).values                        # [B, N, C]

        idx = torch.arange(N, device=p.device)
        p_self = p[:, idx, idx, :]                                 # [B, N, C]

        feats = torch.cat([p_mean, p_max, p_self], dim=-1)         # [B, N, 3C]
        raw = self.head(self.mlp(feats)).squeeze(-1)               # [B, N]
        # Clamp raw to keep softplus in [~0.05, ~e^3]. Paired with clamped
        # log_b_hat target in _compute_flex_losses; prevents pathological
        # gradient explosions from B-factor outliers.
        raw = raw.clamp(min=-3.0, max=3.0)
        nu = F.softplus(raw)

        nu = nu * mask + (1.0 - mask)
        return nu
