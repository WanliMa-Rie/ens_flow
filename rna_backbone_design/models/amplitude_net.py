"""Per-residue diffusion amplitude network for SE(3) stochastic transport.

Predicts per-residue isotropic SE(3) diffusion amplitudes (σ_x, σ_ω)
conditioned on the noisy backbone state Z_t, structural context c_i,
and timestep t.

Inputs:
  - trans_t    [B, N, 3]   : noisy translation at time t
  - rotmats_t  [B, N, 3, 3]: noisy rotation at time t
  - single_emb [B, N, D_s] : per-residue structural context (e.g. 384-dim)
  - t          [B, 1]      : diffusion timestep

Outputs:
  - sigma_x    [B, N]      : per-residue translation amplitude (> 0)
  - sigma_omega[B, N]      : per-residue rotation amplitude (> 0)

B-factor profile is NOT an input — it is only used as supervision target.
"""

import torch
from torch import nn
import torch.nn.functional as F

from rna_backbone_design.models.utils import get_time_embedding


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: x → scale * x + shift."""

    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return x * (1 + scale) + shift


class ResConvBlock(nn.Module):
    """Residual block: LayerNorm → Conv1D → FiLM(t) → SiLU → Conv1D → residual."""

    def __init__(self, hidden_dim, cond_dim, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.film = FiLMLayer(cond_dim, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad)

    def forward(self, x, cond):
        h = self.norm(x)
        h = self.conv1(h.transpose(1, 2)).transpose(1, 2)
        h = self.film(h, cond)
        h = F.silu(h)
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        return x + h


class AmplitudeNet(nn.Module):
    """Predicts per-residue isotropic SE(3) diffusion amplitudes (σ_x, σ_ω)."""

    def __init__(self, c_single_in=384, timestep_embed_dim=64, hidden_dim=128,
                 num_layers=4, kernel_size=5):
        super().__init__()
        self.timestep_embed_dim = timestep_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        input_dim = 3 + 6 + c_single_in
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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

    def forward(self, trans_t, rotmats_t, single_emb, t):
        """
        Returns:
            sigma_x:     [B, N] translation amplitude (> 0).
            sigma_omega: [B, N] rotation amplitude (> 0).
        """
        rot_feat = torch.stack([
            rotmats_t[..., 0, 0], rotmats_t[..., 0, 1], rotmats_t[..., 0, 2],
            rotmats_t[..., 1, 1], rotmats_t[..., 1, 2],
            rotmats_t[..., 2, 2],
        ], dim=-1)

        node_feat = torch.cat([trans_t, rot_feat, single_emb], dim=-1)
        t_emb = get_time_embedding(t[:, 0], self.timestep_embed_dim)
        cond = self.time_mlp(t_emb)

        x = self.input_proj(node_feat)
        for block in self.blocks:
            x = block(x, cond)

        sigma_x = F.softplus(self.head_trans(x).squeeze(-1))
        sigma_omega = F.softplus(self.head_rots(x).squeeze(-1))
        return sigma_x, sigma_omega


class UncertaintyNet(nn.Module):
    """Predicts per-residue remaining cumulative uncertainty U_i(t).

    Feynman-Kac value function: U_i(t) = ∫_t^1 q_i(x_s, s) ds.
    One scalar per residue (not per atom).
    """

    def __init__(self, c_single_in=384, timestep_embed_dim=64, hidden_dim=128,
                 num_layers=4, kernel_size=5):
        super().__init__()
        self.timestep_embed_dim = timestep_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        input_dim = 3 + 6 + c_single_in
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            ResConvBlock(hidden_dim, hidden_dim, kernel_size)
            for _ in range(num_layers)
        ])

        # 1 scalar per residue
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trans_t, rotmats_t, single_emb, t):
        """
        Returns:
            U: [B, N] remaining uncertainty per residue.
        """
        rot_feat = torch.stack([
            rotmats_t[..., 0, 0], rotmats_t[..., 0, 1], rotmats_t[..., 0, 2],
            rotmats_t[..., 1, 1], rotmats_t[..., 1, 2],
            rotmats_t[..., 2, 2],
        ], dim=-1)

        node_feat = torch.cat([trans_t, rot_feat, single_emb], dim=-1)
        t_emb = get_time_embedding(t[:, 0], self.timestep_embed_dim)
        cond = self.time_mlp(t_emb)

        x = self.input_proj(node_feat)
        for block in self.blocks:
            x = block(x, cond)

        return self.head(x).squeeze(-1)  # [B, N]
