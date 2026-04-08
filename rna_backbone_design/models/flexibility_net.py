"""Sequence-only per-residue flexibility predictor for EnsFlow Level 3.

Given the dataset's single-residue embedding ``s_i`` (same tensor the FlowModel
consumes as ``single_embedding``), produce a positive, mean-1 normalized scalar
``nu_i`` that scales the Level 2 bridge widths jointly:

    sigma_x_i    = sigma_bar_x    * sqrt(nu_i)
    sigma_ome_i  = sigma_bar_omega * sqrt(nu_i)

Design constraints (see docs/architecture_report.md §3.2):

1. **Strict nesting Level 2 ⊂ Level 3.** At initialization ``nu_i ≡ 1`` so the
   Level 3 corruption/inference reduces to Level 2 byte-for-byte. We achieve
   this with a zero-initialized final Linear weight and a bias ≈ log(e-1), so
   softplus(bias) = 1 and every ``a_i`` is the same constant, which then gets
   normalized to 1 by the mean-1 step.

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
    """2-layer MLP that maps ``single_embedding`` to per-residue ``nu_i``.

    Args:
        c_single_in: input feature dimension of ``single_embedding``
            (same as ``model.node_features.c_single_in`` in the config).
        hidden_dim: hidden width of the 2-layer MLP.
        eps: small positive constant added after softplus for numerical
            stability (avoids ``a_i == 0`` and ``log(nu_i) == -inf``).
    """

    def __init__(self, c_single_in: int, hidden_dim: int = 128, eps: float = 1e-4):
        super().__init__()
        self._eps = float(eps)

        self.net = nn.Sequential(
            nn.LayerNorm(c_single_in),
            nn.Linear(c_single_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Zero-init the final Linear so a_i is constant across residues at
        # init; set the bias so softplus(bias) = 1. Combined with the mean-1
        # normalization this gives nu_i ≡ 1 exactly at step 0 — i.e. Level 3
        # starts identical to Level 2 (strict nesting).
        final_linear = self.net[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, _SOFTPLUS_UNITY_BIAS)

    def forward(
        self, single_embedding: torch.Tensor, res_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute ``nu_i`` for every residue.

        Args:
            single_embedding: ``[B, N, c_single_in]`` per-residue conditioning
                features (the same tensor passed into ``FlowModel.forward``).
            res_mask: ``[B, N]`` 0/1 residue mask; padding positions must be 0.

        Returns:
            ``nu``: ``[B, N]`` positive tensor with masked mean 1 along the
            residue axis. Padding positions are returned as 1.0 so that they
            contribute a no-op width multiplier if accidentally used.
        """
        # Raw MLP output before the positive map: [B, N, 1] → [B, N]
        raw = self.net(single_embedding).squeeze(-1)

        # Positive activation ``a_i``. Adding eps prevents a_i==0 which would
        # otherwise make log(nu_i) undefined for L_B / L_nu.
        a = F.softplus(raw) + self._eps

        # Masked mean along residues: sum(a * mask) / max(sum(mask), 1).
        mask = res_mask.to(a.dtype)
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
