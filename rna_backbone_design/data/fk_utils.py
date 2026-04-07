"""Utilities for TD-Flow: B-factor-guided SE(3) stochastic bridge.

Per-residue instantaneous flexibility rate:
  q_i = σ_{x,i}² + σ_{ω,i}²

The model decides how to split the flexibility budget between
translation and rotation. B-factor supervision is per-residue
(average B-factor of the nucleotide).
"""

import torch


def compute_flexibility_rate(sigma_x, sigma_omega):
    """Per-residue instantaneous flexibility rate.

    q_i = σ_{x,i}² + σ_{ω,i}²

    Args:
        sigma_x:     [B, N] translation amplitude.
        sigma_omega: [B, N] rotation amplitude.

    Returns:
        q: [B, N] per-residue flexibility rate.
    """
    return sigma_x ** 2 + sigma_omega ** 2


def normalize_b_factors_positive(b_factors, mask, eps=1e-6):
    """Per-molecule positive normalization of B-factors.

    log-shift-scale: log → shift min to 0 → scale mean to 1.
    Preserves positivity (required by U ≥ 0 constraint).

    Args:
        b_factors: [B, N] per-residue average B-factors.
        mask:      [B, N] which residues are resolved.
        eps: numerical stability.

    Returns:
        [B, N] positive normalized values, zero at masked positions.
    """
    log_b = torch.log(b_factors.clamp(min=eps))
    # Per-molecule min (only over valid residues)
    large_val = 1e6
    log_b_masked = log_b + (1 - mask) * large_val
    log_min = log_b_masked.min(dim=-1, keepdim=True).values  # [B, 1]
    shifted = (log_b - log_min) * mask  # ≥ 0
    count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    shifted_mean = (shifted * mask).sum(dim=-1, keepdim=True) / count
    return (shifted / (shifted_mean + eps)) * mask
