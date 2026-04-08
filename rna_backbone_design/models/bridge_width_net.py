"""Conditional heteroscedastic bridge width network for SE(3) stochastic bridge.

Predicts per-residue bounded modulation factors (alpha_x, alpha_omega) that scale
a base bridge schedule sigma_0(t) = lambda * sqrt(t(1-t)).

Effective bridge width: beta = sigma_0(t) * alpha, where alpha is bounded
in [alpha_min, alpha_max] via sigmoid mapping.

Key design: inputs include the pair representation from the FlowModel, which encodes
pairwise nucleotide relationships and distinguishes different structural members of the
same sequence family — enabling the network to assign different flexibility profiles to
structurally distinct conformations with identical sequences.
"""

import torch
from torch import nn

from rna_backbone_design.models.amplitude_net import FiLMLayer, ResConvBlock
from rna_backbone_design.models.utils import get_time_embedding
from rna_backbone_design.data import so3_utils


def base_schedule(t, lambda_val):
    """Endpoint-safe bridge schedule: sigma_0(t) = lambda * sqrt(t * (1 - t)).

    Vanishes at t=0 and t=1, peaks at t=0.5 with value lambda/2.

    Args:
        t: [B, 1] or scalar timestep in [0, 1].
        lambda_val: scale factor.

    Returns:
        sigma_0: same shape as t.
    """
    return lambda_val * torch.sqrt(t * (1.0 - t) + 1e-8)


def compute_geometric_residuals(trans_t, rotmats_t, pred_trans_0, pred_rotmats_0):
    """Compute geometric residuals from the FlowModel's reference prediction.

    Args:
        trans_t:        [B, N, 3]    noisy translations at time t.
        rotmats_t:      [B, N, 3, 3] noisy rotations at time t.
        pred_trans_0:   [B, N, 3]    predicted clean translations.
        pred_rotmats_0: [B, N, 3, 3] predicted clean rotations.

    Returns:
        delta_x:     [B, N, 3] translation residual (x_t - x_hat_0).
        delta_omega: [B, N, 3] rotation residual log(R_hat_0^T * R_t).
    """
    delta_x = trans_t - pred_trans_0
    delta_omega = so3_utils.calc_rot_vf(pred_rotmats_0, rotmats_t)
    return delta_x, delta_omega


class BridgeWidthNet(nn.Module):
    """Predicts per-residue bounded modulation factors (alpha_x, alpha_omega).

    alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(u)

    Conditioned on:
      - geometric residuals (delta_x, delta_omega): deviation of current trajectory
        from the drift model's reference prediction.
      - single_emb: per-residue sequence context.
      - pair_emb: pairwise nucleotide relationships from the FlowModel's edge
        representation. This is the key signal that distinguishes different structural
        members of the same sequence family — same sequence can have different pair
        representations if the underlying structural context differs.
      - timestep t.
    """

    def __init__(self, c_single_in=384, c_pair_in=128, timestep_embed_dim=64,
                 hidden_dim=128, num_layers=4, kernel_size=5,
                 alpha_min=0.1, alpha_max=3.0):
        super().__init__()
        self.timestep_embed_dim = timestep_embed_dim
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Single path: c_i (c_single_in) + delta_x (3) + delta_omega (3)
        self.single_proj = nn.Sequential(
            nn.Linear(c_single_in + 3 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pair path: mean-pool pair_emb over columns → per-residue pair context
        # pair_emb [B, N, N, c_pair_in] → mean over dim=2 → [B, N, c_pair_in]
        self.pair_proj = nn.Linear(c_pair_in, hidden_dim)

        self.blocks = nn.ModuleList([
            ResConvBlock(hidden_dim, hidden_dim, kernel_size)
            for _ in range(num_layers)
        ])

        self.head_trans = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.head_rots = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, single_emb, delta_x, delta_omega, pair_emb, t):
        """
        Args:
            single_emb:  [B, N, c_single_in] per-residue sequence context.
            delta_x:     [B, N, 3] translation residual.
            delta_omega: [B, N, 3] rotation residual (rotvec).
            pair_emb:    [B, N, N, c_pair_in] pairwise nucleotide representation.
            t:           [B, 1] timestep.

        Returns:
            alpha_x:     [B, N] bounded translation modulation factor.
            alpha_omega: [B, N] bounded rotation modulation factor.
        """
        # Single + geometric residual path
        node_feat = torch.cat([single_emb, delta_x, delta_omega], dim=-1)
        x = self.single_proj(node_feat)  # [B, N, H]

        # Pair path: aggregate pairwise context per residue via mean-pooling
        # pair_emb [B, N, N, c_p] → [B, N, c_p] → [B, N, H]
        pair_ctx = pair_emb.mean(dim=2)   # [B, N, c_pair_in]
        x = x + self.pair_proj(pair_ctx)  # fuse: additive

        # Time conditioning
        t_emb = get_time_embedding(t[:, 0], self.timestep_embed_dim)
        cond = self.time_mlp(t_emb)  # [B, H]

        for block in self.blocks:
            x = block(x, cond)

        raw_x = self.head_trans(x).squeeze(-1)
        raw_w = self.head_rots(x).squeeze(-1)

        alpha_x = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(raw_x)
        alpha_omega = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(raw_w)
        return alpha_x, alpha_omega
