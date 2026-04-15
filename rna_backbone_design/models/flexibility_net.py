"""Per-residue flexibility predictor for EnsFlow Level 3.

Given the dataset's single embedding ``s_i`` and pair embedding ``p_ij`` (the
same tensors the FlowModel consumes as ``single_embedding`` / ``pair_embedding``),
produce a positive, mean-1 normalized scalar ``nu_i`` that scales the Level 2
bridge widths jointly:

    sigma_x_i    = sigma_bar_x    * sqrt(nu_i)
    sigma_ome_i  = sigma_bar_omega * sqrt(nu_i)

Architecture: ``LayerNorm + Linear`` projects ``s_i`` into a hidden width, then
**one block of pair-biased multi-head self-attention** mixes information across
residues with ``p_ij`` providing a per-head additive attention bias (analogous
to how the FlowModel's IPA trunk consumes pair features). A final scalar
Linear head reads off ``raw_i``, which is mapped through ``softplus`` and
per-sample mean-1 normalized to give ``nu_i``.

Design constraints (see docs/architecture_report.md §3.2):

1. **Strict nesting Level 2 ⊂ Level 3.** At initialization ``nu_i ≡ 1`` so the
   Level 3 corruption/inference reduces to Level 2 byte-for-byte. We achieve
   this by zero-initializing the final Linear weight and setting its bias to
   ``log(e-1)``: regardless of what the attention block outputs, the head's
   pre-softplus activation is the same constant for every residue, softplus
   maps it to 1, and the mean-1 step then leaves ``nu_i ≡ 1`` exactly.

2. **Identifiability: one scalar per residue.** A single scalar B-factor cannot
   uniquely pin down two independent width heads. We only learn ``nu_i``; both
   translation and rotation widths scale through the *same* ``nu_i``.

3. **Per-sample mean-1 normalization.** ``nu_i = a_i / mean(a_k)`` removes the
   global scale (that lives in ``sigma_bar_x``, ``sigma_bar_omega`` of the
   config) and ensures downstream identities like ``hat(b)_i = nu_i`` in the
   simplified ``L_B`` derivation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# softplus(log(e - 1)) == 1, so a zero-weight final Linear with this bias makes
# the pre-normalization activation a_i identically equal to 1 for every residue
# at initialization, which then survives the mean-1 step as nu_i ≡ 1.
_SOFTPLUS_UNITY_BIAS = math.log(math.e - 1.0)


class FlexibilityNet(nn.Module):
    """Pair-biased self-attention head that maps ``(single, pair)`` to ``nu_i``.

    Args:
        c_single_in: input feature dimension of ``single_embedding``
            (same as ``model.node_features.c_single_in`` in the config).
        c_pair_in: input feature dimension of ``pair_embedding``
            (same as ``model.edge_features.c_pair_in`` in the config).
        hidden_dim: hidden width of the attention block. Must be divisible
            by ``n_heads``.
        n_heads: number of attention heads in the pair-biased self-attention.
        eps: small positive constant added after softplus for numerical
            stability (avoids ``a_i == 0`` and ``log(nu_i) == -inf``).
    """

    def __init__(
        self,
        c_single_in: int,
        c_pair_in: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        eps: float = 1e-4,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})."
            )
        self._eps = float(eps)
        self._n_heads = int(n_heads)

        # Single → hidden projection.
        self.single_ln = nn.LayerNorm(c_single_in)
        self.single_proj = nn.Linear(c_single_in, hidden_dim)

        # Pair → per-head additive attention bias.
        self.pair_ln = nn.LayerNorm(c_pair_in)
        self.pair_proj = nn.Linear(c_pair_in, n_heads, bias=False)

        # One block of pair-biased multi-head self-attention + residual LN.
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            bias=True,
        )
        self.attn_ln = nn.LayerNorm(hidden_dim)

        # Final scalar head. Zero-init weight + log(e-1) bias is what makes
        # nu_i ≡ 1 at step 0 regardless of (single, pair) inputs and regardless
        # of the random init of the upstream attention block — i.e. strict
        # Level 2 ⊂ Level 3 nesting holds for *any* state of the trunk above.
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, _SOFTPLUS_UNITY_BIAS)

    def forward(
        self,
        single_embedding: torch.Tensor,
        pair_embedding: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``nu_i`` for every residue.

        Args:
            single_embedding: ``[B, N, c_single_in]`` per-residue conditioning
                features (the same tensor passed into ``FlowModel.forward``).
            pair_embedding: ``[B, N, N, c_pair_in]`` pair conditioning features
                (the same tensor passed into ``FlowModel.forward``).
            res_mask: ``[B, N]`` 0/1 residue mask; padding positions must be 0.

        Returns:
            ``nu``: ``[B, N]`` positive tensor with masked mean 1 along the
            residue axis. Padding positions are returned as 1.0 so that they
            contribute a no-op width multiplier if accidentally used.
        """
        B, N, _ = single_embedding.shape
        mask = res_mask.to(single_embedding.dtype)            # [B, N]
        mask_col = mask[..., None]                            # [B, N, 1]

        # Project single → hidden. Zero out padding rows after LN so they
        # cannot leak garbage into the attention output (LN of an all-zero
        # padding row is well-defined since LN normalizes per token over its
        # own feature axis, but the dataset's padding may not be exactly zero).
        s = self.single_proj(self.single_ln(single_embedding))  # [B, N, h]
        s = s * mask_col

        # Pair → per-head additive bias of shape [B, n_heads, N, N], then
        # reshape to the (B*n_heads, N, N) layout that nn.MultiheadAttention
        # expects for a per-batch, per-head attn_mask.
        bias = self.pair_proj(self.pair_ln(pair_embedding))     # [B, N, N, h]
        bias = bias.permute(0, 3, 1, 2).contiguous()            # [B, h, N, N]
        attn_bias = bias.view(B * self._n_heads, N, N)

        # key_padding_mask as a float additive mask: 0 for valid keys, -inf
        # for padding keys. PyTorch deprecates mixing bool and float masks
        # when attn_mask is float, so we make both floats. The -inf entries
        # broadcast over heads and dominate attn_bias on padding columns.
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
        # If a query row is on padding (or — degenerate — every key is masked)
        # MHA can return NaN for that row. Those rows are zeroed out by the
        # mask multiply below, but we sanitize first so the residual add does
        # not poison the valid rows on the same sample.
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        s = self.attn_ln(s + attn_out)
        s = s * mask_col

        # Scalar head → raw activation per residue.
        raw = self.head(s).squeeze(-1)                           # [B, N]

        # Positive activation ``a_i``. Adding eps prevents a_i==0 which would
        # otherwise make log(nu_i) undefined for L_B / L_nu.
        a = F.softplus(raw) + self._eps

        # Masked mean along residues: sum(a * mask) / max(sum(mask), 1).
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        a_masked = a * mask
        a_mean = a_masked.sum(dim=-1, keepdim=True) / mask_sum  # [B, 1]

        # nu_i = a_i / mean_k(a_k). Clamp the denominator to avoid division by
        # zero on fully-masked samples (should not happen in practice, but is
        # cheap insurance).
        nu = a / a_mean.clamp(min=1e-6)

        # On padding positions write 1.0 so a detached ``nu`` used as a width
        # multiplier is an identity no-op there; L_B / L_nu will apply its own
        # masking based on res_mask and valid b_factors before reducing.
        nu = nu * mask + (1.0 - mask)
        return nu
