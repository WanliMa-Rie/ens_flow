"""Per-residue diffusion amplitude network for SE(3) stochastic transport.

Predicts per-residue SE(3) diffusion parameters conditioned on the noisy
backbone state Z_t, structural context c_i, and timestep t.

Inputs:
  - trans_t    [B, N, 3]   : noisy translation at time t
  - rotmats_t  [B, N, 3, 3]: noisy rotation at time t
  - single_emb [B, N, D_s] : per-residue structural context (e.g. 384-dim)
  - t          [B, 1]      : diffusion timestep

Outputs (anisotropic mode, default):
  - sigma_x    [B, N]      : isotropic translation amplitude (> 0)
  - s1, s2, s3 [B, N] each : anisotropic rotation variances along local x/y/z axes (> 0)

B-factor profile is NOT an input — it is only used as supervision target.

Architecture:
  - Input projection: concat(trans_t, upper-tri of rotmats_t, single_emb) → hidden
  - Timestep conditioning via FiLM (Feature-wise Linear Modulation)
  - Residual blocks with 1D convolutions for local sequence context
  - Output heads with softplus for positivity
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
        """
        Args:
            x: [B, N, D] features to modulate.
            cond: [B, cond_dim] conditioning vector (broadcast over N).
        """
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
        """
        Args:
            x: [B, N, D] residue features.
            cond: [B, cond_dim] timestep conditioning.
        """
        h = self.norm(x)
        h = self.conv1(h.transpose(1, 2)).transpose(1, 2)  # [B, N, D]
        h = self.film(h, cond)
        h = F.silu(h)
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        return x + h


class AmplitudeNet(nn.Module):
    """Predicts per-residue SE(3) diffusion amplitudes from noisy state and context.

    Conditioned on (Z_t, c, t) — the noisy backbone geometry, structural
    conditioning features, and timestep. B-factor profile is NOT an input.
    """

    def __init__(self, c_single_in=384, timestep_embed_dim=64, hidden_dim=128,
                 num_layers=4, kernel_size=5):
        super().__init__()
        self.timestep_embed_dim = timestep_embed_dim

        # Timestep embedding → conditioning vector
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-residue input: trans_t (3) + rotmat upper-tri (6) + single_emb (D_s)
        input_dim = 3 + 6 + c_single_in
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Residual conv blocks with FiLM time conditioning
        self.blocks = nn.ModuleList([
            ResConvBlock(hidden_dim, hidden_dim, kernel_size)
            for _ in range(num_layers)
        ])

        # Output heads: 1 for isotropic translation + 3 for anisotropic rotation
        self.head_trans = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.head_rots_aniso = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),  # s1, s2, s3 per residue
        )

    def _extract_features(self, trans_t, rotmats_t, single_emb, t):
        """Shared feature extraction through backbone."""
        # Extract compact rotation features: upper triangular of rotation matrix
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
        return x

    def forward(self, trans_t, rotmats_t, single_emb, t):
        """
        Args:
            trans_t:    [B, N, 3]    noisy translations at time t.
            rotmats_t:  [B, N, 3, 3] noisy rotations at time t.
            single_emb: [B, N, D_s]  per-residue structural context.
            t:          [B, 1]       timestep in [0, 1].

        Returns:
            sigma_x: [B, N]    isotropic translation amplitude (> 0).
            s1:      [B, N]    rotation variance around local x-axis (> 0).
            s2:      [B, N]    rotation variance around local y-axis (> 0).
            s3:      [B, N]    rotation variance around local z-axis (> 0).
        """
        x = self._extract_features(trans_t, rotmats_t, single_emb, t)

        sigma_x = F.softplus(self.head_trans(x).squeeze(-1))
        s_aniso = F.softplus(self.head_rots_aniso(x))  # [B, N, 3]
        s1, s2, s3 = s_aniso[..., 0], s_aniso[..., 1], s_aniso[..., 2]
        return sigma_x, s1, s2, s3


class UncertaintyNet(nn.Module):
    """Predicts per-frame-atom remaining cumulative uncertainty U_j(t).

    Feynman-Kac value function: U_j(t) = ∫_t^1 q_j(x_s, s) ds.
    Outputs 3 scalars per residue (one per frame atom: C4', O4', C3').
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

        # 3 outputs per residue: U for C4', O4', C3'
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, trans_t, rotmats_t, single_emb, t):
        """
        Args:
            trans_t:    [B, N, 3]    noisy translations at time t.
            rotmats_t:  [B, N, 3, 3] noisy rotations at time t.
            single_emb: [B, N, D_s]  per-residue structural context.
            t:          [B, 1]       timestep in [0, 1].

        Returns:
            U_j: [B, N, 3] remaining uncertainty per frame atom (C4', O4', C3').
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

        return self.head(x)  # [B, N, 3]
