"""Utilities for SE(3) stochastic transport with atom-wise B-factor supervision.

Supports both isotropic and anisotropic rotation parameterizations.

Anisotropic (Feynman-Kac framework):
  q_j = 3σ_x² + (a_{j2}² + a_{j3}²)s_1 + (a_{j1}² + a_{j3}²)s_2 + (a_{j1}² + a_{j2}²)s_3

where a_j is the fixed local coordinate of frame atom j, and s_1, s_2, s_3
are anisotropic rotation variances along local x, y, z axes.

Frame atom local coordinates (constant across all nucleotide types):
  C4' = (0, 0, 0),  O4' = (1.45, 0, 0),  C3' = (-0.38, 1.48, 0)
"""

import torch

from rna_backbone_design.data import all_atom as rna_all_atom

# Fixed local coordinates for the 3 frame atoms [3, 3]
# Order: C4', O4', C3' — same as frame definition
FRAME_ATOM_LOCAL_COORDS = torch.tensor([
    [0.0, 0.0, 0.0],       # C4' (origin)
    [1.45, 0.0, 0.0],      # O4' (x-axis)
    [-0.38, 1.48, 0.0],    # C3' (xy-plane)
], dtype=torch.float32)


def get_frame_atom_local_coords(device):
    """Return fixed frame atom local coordinates [3, 3] on given device."""
    return FRAME_ATOM_LOCAL_COORDS.to(device)


def compute_local_atom_coords(trans, rotmats, atom23_pos):
    """Compute atom positions in the local backbone frame.

    Given global frame (x_i, R_i) and global atom positions r_ij,
    computes a_ij = R_i^T (r_ij - x_i).

    Args:
        trans:      [B, N, 3]      backbone frame translations.
        rotmats:    [B, N, 3, 3]   backbone frame rotations.
        atom23_pos: [B, N, 23, 3]  global atom positions (compact 23-atom format).

    Returns:
        a_local: [B, N, 23, 3] atom positions in local frame.
    """
    centered = atom23_pos - trans.unsqueeze(-2)
    a_local = torch.einsum("bnij,bnaj->bnai", rotmats.transpose(-1, -2), centered)
    return a_local


def compute_atom_trace(sigma_x, sigma_omega, a_local, atom_mask):
    """Compute atom-wise instantaneous diffusion trace (isotropic rotation).

    Tr(D_ij^atom) = 3 * σ_{x,i}² + 2 * ||a_ij||² * σ_{ω,i}²

    Args:
        sigma_x:     [B, N]       translation diffusion amplitude.
        sigma_omega:  [B, N]       rotation diffusion amplitude.
        a_local:     [B, N, A, 3] local atom coordinates.
        atom_mask:   [B, N, A]    which atoms are resolved.

    Returns:
        trace: [B, N, A] per-atom diffusion trace, zero for masked atoms.
    """
    arm_sq = (a_local ** 2).sum(dim=-1)  # [B, N, A] = ||a_ij||²
    trace = 3.0 * sigma_x.unsqueeze(-1) ** 2 + 2.0 * arm_sq * sigma_omega.unsqueeze(-1) ** 2
    return trace * atom_mask


def compute_frame_atom_trace_aniso(sigma_x, s1, s2, s3, device):
    """Compute per-frame-atom trace with anisotropic rotation (§3.5 of framework doc).

    q_j = 3σ_x² + (a_{j2}² + a_{j3}²)s_1 + (a_{j1}² + a_{j3}²)s_2 + (a_{j1}² + a_{j2}²)s_3

    Uses fixed local coordinates from nucleotide_constants.

    Args:
        sigma_x: [B, N]  isotropic translation amplitude.
        s1:      [B, N]  rotation variance around local x-axis.
        s2:      [B, N]  rotation variance around local y-axis.
        s3:      [B, N]  rotation variance around local z-axis.
        device:  torch device.

    Returns:
        q: [B, N, 3] per-frame-atom trace (C4', O4', C3').
    """
    a = get_frame_atom_local_coords(device)  # [3, 3]
    a1_sq = a[:, 0] ** 2  # [3]
    a2_sq = a[:, 1] ** 2  # [3]
    a3_sq = a[:, 2] ** 2  # [3]

    # Sensitivity coefficients for each rotation axis
    coeff_s1 = a2_sq + a3_sq  # [3], rotation around x affects y,z components
    coeff_s2 = a1_sq + a3_sq  # [3], rotation around y affects x,z components
    coeff_s3 = a1_sq + a2_sq  # [3], rotation around z affects x,y components

    # q_j = 3σ_x² + coeff_s1[j]*s1 + coeff_s2[j]*s2 + coeff_s3[j]*s3
    trans_part = 3.0 * sigma_x.unsqueeze(-1) ** 2  # [B, N, 1]
    rot_part = (
        s1.unsqueeze(-1) * coeff_s1  # [B, N, 3]
        + s2.unsqueeze(-1) * coeff_s2
        + s3.unsqueeze(-1) * coeff_s3
    )
    return trans_part + rot_part  # [B, N, 3]


def normalize_atom_b_factors(b_factors, atom_mask, eps=1e-6):
    """Per-molecule z-score normalization of log atom-wise B-factors.

    Normalizes across all resolved atoms in the molecule (flattened over
    residues and atom types).

    Args:
        b_factors: [B, N, A] raw atom-wise B-factors.
        atom_mask: [B, N, A] which atoms are resolved (1/0).
        eps: numerical stability constant.

    Returns:
        [B, N, A] normalized values, zero at masked positions.
    """
    log_b = torch.log(b_factors.clamp(min=0) + eps)
    count = atom_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1)  # [B, 1, 1]
    mean = (log_b * atom_mask).sum(dim=(-2, -1), keepdim=True) / count
    var = ((log_b - mean) ** 2 * atom_mask).sum(dim=(-2, -1), keepdim=True) / count
    return (log_b - mean) / (var.sqrt() + eps) * atom_mask


def normalize_b_factors_positive(b_factors, atom_mask, eps=1e-6):
    """Per-molecule normalization preserving positivity for Feynman-Kac framework.

    Uses log-scale divided by per-molecule mean, ensuring all values > 0.
    This is necessary because U(t) must be non-negative (integral of non-negative q).

    Args:
        b_factors: [B, N, A] raw atom-wise B-factors.
        atom_mask: [B, N, A] which atoms are resolved (1/0).
        eps: numerical stability constant.

    Returns:
        [B, N, A] positive normalized values, zero at masked positions.
    """
    log_b = torch.log(b_factors.clamp(min=eps))
    count = atom_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1)
    mean = (log_b * atom_mask).sum(dim=(-2, -1), keepdim=True) / count
    # Shift so minimum is near 0, then scale to mean=1
    shifted = (log_b - log_b.min()) * atom_mask  # ≥ 0
    shifted_mean = (shifted * atom_mask).sum(dim=(-2, -1), keepdim=True) / count
    return (shifted / (shifted_mean + eps)) * atom_mask


def get_gt_atom23_pos(trans_1, rotmats_1, is_na_residue_mask, torsions):
    """Get ground truth atom23 positions from SE(3) frames and torsion angles.

    Calls compute_backbone and returns the compact 23-atom positions.

    Args:
        trans_1:             [B, N, 3]    GT translations.
        rotmats_1:           [B, N, 3, 3] GT rotations.
        is_na_residue_mask:  [B, N]       nucleotide mask.
        torsions:            [B, N, 16]   torsion sin/cos (8 angles × 2).

    Returns:
        atom23_pos: [B, N, 23, 3] atom positions in global frame.
    """
    _, _, _, _, atom23_pos = rna_all_atom.compute_backbone(
        bb_rigids=rna_all_atom.create_rna_rigid(rotmats_1, trans_1),
        torsions=torsions,
        is_na_residue_mask=is_na_residue_mask,
    )
    return atom23_pos
