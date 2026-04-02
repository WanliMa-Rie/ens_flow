"""Per-residue diffusion amplitude network for U-STEER.

Predicts separate positive translation and rotation amplitudes a_{x,ψ}^{(i)}
and a_{R,ψ}^{(i)} for each residue, conditioned on normalized B-factor profile
and timestep.

Architecture:
  - B-factor input embedding (Linear)
  - Timestep conditioning via FiLM (Feature-wise Linear Modulation):
    sinusoidal t → MLP → per-layer (scale, shift) pairs
  - Residual blocks with 1D convolutions for local sequence context
  - Separate output heads for translation and rotation channels
  - Softplus output to ensure positivity
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
    """Predicts per-residue diffusion amplitudes for translation and rotation.

    Following the steer.md decomposition Σ = U_θ · A_ψ, this network
    learns A_ψ(B̃, t) = diag(a^{(1)}_ψ, ..., a^{(L)}_ψ) with separate
    amplitudes for translation (a_x) and rotation (a_R) channels.
    """

    def __init__(self, timestep_embed_dim=64, hidden_dim=128, num_layers=4,
                 kernel_size=5):
        super().__init__()
        self.timestep_embed_dim = timestep_embed_dim

        # Timestep embedding → conditioning vector
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # B-factor input projection
        self.input_proj = nn.Linear(1, hidden_dim)

        # Residual conv blocks with FiLM time conditioning
        self.blocks = nn.ModuleList([
            ResConvBlock(hidden_dim, hidden_dim, kernel_size)
            for _ in range(num_layers)
        ])

        # Separate output heads for translation and rotation amplitudes
        self.head_trans = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.head_rots = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, b_norm, t):
        """
        Args:
            b_norm: [B, N] normalized B-factor profile.
            t: [B, 1] timestep in [0, 1].

        Returns:
            (a_trans, a_rots): each [B, N], positive amplitudes per residue.
        """
        # Timestep conditioning: [B, hidden_dim]
        t_emb = get_time_embedding(t[:, 0], self.timestep_embed_dim)
        cond = self.time_mlp(t_emb)

        # Per-residue input: [B, N, hidden_dim]
        x = self.input_proj(b_norm.unsqueeze(-1))

        # Residual blocks with local context and time modulation
        for block in self.blocks:
            x = block(x, cond)

        # Separate positive amplitudes
        a_trans = F.softplus(self.head_trans(x).squeeze(-1))
        a_rots = F.softplus(self.head_rots(x).squeeze(-1))
        return a_trans, a_rots
