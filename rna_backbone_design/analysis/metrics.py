from collections import namedtuple
from typing import Optional

import numpy as np
import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Float
from loguru import logger
from torch import Tensor

from rna_backbone_design.data import nucleotide_constants
from rna_backbone_design.data import all_atom as rna_all_atom

KabschResult = namedtuple("KabschResult", ["rot_mats", "trans_vecs"])

def exists(x: object) -> bool:
    return x is not None


def safe_divide(a: Tensor, b: Tensor, eps: float = 1e-10) -> Tensor:
    return a / (b + eps) 


def safe_sqrt(x: Tensor, eps: float = 1e-10) -> Tensor:
    return torch.sqrt(x + eps)


def masked_mean(x: Tensor, mask: Tensor, dim: int = None, keepdim: bool = False) -> Tensor:
    assert x.shape == mask.shape
    x_masked = x * mask
    num_valid = mask.sum(dim=dim, keepdim=keepdim)
    return safe_divide(x_masked.sum(dim=dim, keepdim=keepdim), num_valid)


def rmsd(x: Tensor, y: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-10) -> Tensor:
    """Compute RMSD (root mean square distance) between two sets of coordinates, with optional masking"""
    if not exists(mask):
        mask = torch.ones_like(x[..., 0])
    sq_dist = (x - y).pow(2).sum(dim=-1) * mask  # [B, N]
    mean_sq_dist = sq_dist.sum(dim=-1) / (mask.sum(dim=-1) + eps)  # [B]
    return mean_sq_dist.sqrt()  # [B]


def angle_from_points(a: Float[Tensor, "B 3"], b: Float[Tensor, "B 3"], c: Float[Tensor, "B 3"]) -> Float[Tensor, "B"]:
    return angle_from_vectors(b - a, b - c)  # [B]


def angle_from_vectors(v1: Float[Tensor, "B 3"], v2: Float[Tensor, "B 3"]) -> Float[Tensor, "B"]:
    cos_angle = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)  # [B]
    return torch.acos(cos_angle)  # [B]


def dihedral_from_points(
    a: Float[Tensor, "B 3"], b: Float[Tensor, "B 3"], c: Float[Tensor, "B 3"], d: Float[Tensor, "B 3"]
) -> Float[Tensor, "B"]:
    return dihedral_from_vectors(b - a, c - b, d - c)  # [B]


def dihedral_from_vectors(
    v1: "Float[Tensor, 'B 3']", v2: "Float[Tensor, 'B 3']", v3: "Float[Tensor, 'B 3']"
) -> "Float[Tensor, 'B']":
    normalize = lambda v: torch.nn.functional.normalize(v, dim=-1)
    vector_dot = lambda v1, v2: torch.sum(v1 * v2, dim=-1)

    v1 = normalize(v1)  # [B, 3]
    v2 = normalize(v2)  # [B, 3]
    v3 = normalize(v3)  # [B, 3]

    n1 = torch.cross(v1, v2, dim=-1)  # [B, 3]
    n2 = torch.cross(v2, v3, dim=-1)  # [B, 3]

    x = vector_dot(n1, n2)  # [B]
    y = vector_dot(torch.cross(n1, n2, dim=-1), v2)  # [B]
    return torch.atan2(y, x)  # [B]


def center_of_mass(x: Tensor, mask: Optional[Tensor] = None, keepdim: bool = True) -> Float[Tensor, "B 1 3"]:
    if not exists(mask):
        mask = torch.ones_like(x[..., 0], dtype=torch.bool)
    # ... compute center of mass over unmasked atoms in each sample
    x_masked = x * rearrange(mask, "B N -> B N 1")
    x_mean = reduce(x_masked, "B N C -> B 1 C", "sum") / reduce(mask, "B N -> B 1 1", "sum")
    return x_mean if keepdim else x_mean.squeeze(1)


def to_zero_mean(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Subtract center of mass from coordinates, optionally using a mask to ignore certain atoms."""
    return x - center_of_mass(x, mask)


def masked_cov(x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> Float[Tensor, "B 3 3"]:
    if not exists(mask):
        mask = torch.ones_like(x[..., 0], dtype=torch.bool)
    # Mask the input (invalid values are set to 0)
    x_masked = x * rearrange(mask, "B N -> B N 1")
    y_masked = y * rearrange(mask, "B N -> B N 1")
    cov = einsum(x_masked, y_masked, "B N i, B N j -> B i j")
    return cov


def kabsch(
    fixed: Tensor, mobile: Tensor, mask: Optional[Tensor] = None, max_iter: int = 10
) -> tuple[Float[Tensor, "B 3 3"], Float[Tensor, "B 3"]]:
    """
    Applies the differentiable Kabsch algorithm to align two batches (possibly with masks) of 3D point clouds.

    Args:
        fixed (Tensor): A tensor of shape (B, N, 3) representing the first set of points.
        mobile (Tensor): A tensor of shape (B, N, 3) representing the second set of points.
        mask (Tensor): A tensor of shape (B, N) representing the mask of valid points.
        max_iter (int): Maximum number of iterations to perform if SVD is numerically unstable.

    Returns:
        A tuple containing two tensors:
        - rot_mats (Float[Tensor, "B 3 3"]): A tensor of shape (B, 3, 3) representing the rotation matrices that
            align `mobile` with `fixed`.
        - trans_vecs (Float[Tensor, "B 3"]): A tensor of shape (B, 3) representing the translation vectors that
            align `mobile` with `fixed`.
    """
    B, N, _ = fixed.shape  # [B, N, 3]
    device = fixed.device
    assert fixed.shape == mobile.shape
    if not exists(mask):
        mask = torch.ones_like(fixed[..., 0], dtype=torch.bool)
    assert torch.all(mask.sum(dim=1) >= 3), "At least 3 points are required to compute the Kabsch algorithm."

    # Compute centers of mass
    fixed_com = center_of_mass(fixed, mask)  # [B, 1, 3]
    mobile_com = center_of_mass(mobile, mask)  # [B, 1, 3]

    # Compute (spatial) covariance matrix
    cov = masked_cov(fixed - fixed_com, mobile - mobile_com, mask)  # [B, 3, 3]
    assert not torch.isnan(cov).any(), "Covariance matrix contains NaN values."

    # Compute SVD
    U, S, Vt = torch.linalg.svd(cov, full_matrices=True)  # [B, 3, 3], [B, 3], [B, 3, 3]

    # Ensure numerical stability:
    #  Sometimes SVD is poorly conditioned and has zero or degenerate singular values for a square covariance matrix.
    #  This leads to unstable gradients in the backward pass. To avoid this we add a small amount of noise to the
    #  covariance matrix and recompute the SVD until all singular values are non-zero and non-degenerate.
    _has_zero_singular_value = lambda S: S.min(dim=1).values < 1e-3
    _has_degenerate_singular_values = (
        lambda S: reduce(
            (S.pow(2)[:, :, None] - S.pow(2)[:, None, :] + torch.eye(3, device=device)[None, ...]).abs(),
            "B 3 3 -> B",
            "min",
        )
        < 1e-2
    )
    _is_problematic = lambda S: _has_zero_singular_value(S) | _has_degenerate_singular_values(S)
    num_it = 0
    S_is_problematic = _is_problematic(S)  # [B]
    while torch.any(S_is_problematic):
        # Add a small amount of noise to the covariance matrix and recompute SVD
        noise_scale = (
            cov[S_is_problematic].abs().max(dim=1).values * 5e-2  # [B, 3]
        )  # set noise scale to be 5% of the largest value in the covariance matrix
        cov[S_is_problematic] = cov[S_is_problematic] + (torch.rand(3, device=device) * noise_scale).diag()
        U, S, Vt = torch.linalg.svd(cov, full_matrices=True)  # [B, 3, 3], [B, 3], [B, 3, 3]
        num_it += 1
        S_is_problematic = _is_problematic(S)  # [B]

        if num_it > max_iter:
            raise RuntimeError(f"SVD is consistently numerically unstable: {torch.where(S_is_problematic)}")
    if num_it > 0:
        logger.warning(f"Kabsch: SVD was numerically unstable for {num_it} iterations.")

    # Compute rotation matrices and translation vectors
    flip_mat = torch.eye(3, device=device).repeat(B, 1, 1)  # [B, 3, 3]
    flip_mat[:, -1, -1] = cov.det().sign()  # Set all last diagonal elements to sign of determinant
    rot_mats = U @ flip_mat @ Vt  # [B, 3, 3]
    trans_vecs = fixed_com.squeeze(1) - einsum(rot_mats, mobile_com.squeeze(1), "B i j, B j -> B i")  # [B, 3]

    return KabschResult(rot_mats, trans_vecs)


def rototranslate(
    p: Tensor,
    rot_mat: Float[Tensor, "*B 3 3"],
    trans_vec: Float[Tensor, "*B 3"],
    inverse: bool = False,
    validate: bool = False,
) -> torch.Tensor:
    """
    Apply batched rototranslation to a batch of 3D point clouds.
        p' = R @ p + t
    (First rotate, then translate.)

    Args:
        p (Tensor): A tensor of shape (B, N, 3) representing the batch of point clouds.
        rot_mat (Tensor): A tensor of shape (B, 3, 3) representing the rotation matrices.
        trans_vec (Tensor): A tensor of shape (B, 3) representing the translation vectors.
        validate (bool): Whether to validate the input.

    Returns:
        A tensor of shape (B, N, 3) representing the set of points after rototranslation.
    """
    if validate:
        p.device
        B, N, C = p.shape
        assert rot_mat.shape == (B, C, C)
        assert trans_vec.shape == (B, C)
    trans_vec = rearrange(trans_vec, "B i -> B 1 i")  # [B, 1, 3]
    if inverse:
        return einsum((p - trans_vec), rot_mat, "B N i, B i j -> B N j")
    return einsum(p, rot_mat, "B N i, B j i -> B N j") + trans_vec  # [B, N, 3]


def superimpose(fixed: Tensor, mobile: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Superimpose two sets of coordinates using the Kabsch algorithm.

    Args:
        fixed (Tensor): A tensor of shape (B, N, 3) representing the fixed set of points.
        mobile (Tensor): A tensor of shape (B, N, 3) representing the mobile set of points.
        mask (Tensor): A tensor of shape (B, N) representing the mask of valid points.

    Returns:
        A tensor of shape (B, N, 3) representing the mobile set of points after superimposition.
    """
    rots, trans = kabsch(fixed, mobile, mask)
    return rototranslate(mobile, rots, trans)


def gyration_radius(x: Tensor, mask: Optional[Tensor] = None, masses: Optional[Tensor] = None):
    # x: [B, N, 3]
    # mask: [B, N]
    # masses: [B, N]
    if not exists(mask):
        mask = torch.ones_like(x[..., 0], dtype=torch.bool)
    if not exists(masses):
        masses = torch.ones_like(mask, dtype=torch.float32)
    x_com = center_of_mass(x, mask)  # [B, 1, 3]
    dists_sq = ((x - x_com) * rearrange(mask, "B N -> B N 1")).pow(2).sum(dim=-1)  # [B, N]
    inertia_moments = (masses * dists_sq).sum(dim=-1)  # [B]
    protein_masses = (masses * mask).sum(dim=-1)  # [B]
    return torch.sqrt(inertia_moments / protein_masses)  # [B]


def tm_score(
    x: Tensor, y: Tensor, mask: Optional[Tensor] = None, apply_superimpose: bool = True
) -> Tensor:
    """
    Calculate TM-score between batched x and y

    Reference:
     - https://en.wikipedia.org/wiki/Template_modeling_score
    """
    # x: [B, N, 3]
    # y: [B, N, 3]
    # mask: [B, N]
    if not exists(mask):
        mask = torch.ones_like(x[..., 0], dtype=torch.bool)

    if apply_superimpose:
        y = superimpose(x, y, mask)

    d0 = 1.24 * torch.pow(mask.sum(dim=-1, keepdim=True) - 15.0, 1.0 / 3) - 1.8  # [B, N]
    d = (x - y).pow(2).sum(dim=-1).sqrt()  # [B, N]
    score = masked_mean(1 / (1 + (d / d0).pow(2)), mask, dim=-1)  # [B]
    return score


def gdt_score(
    x: Tensor,
    y: Tensor,
    mask: Optional[Tensor] = None,
    thresholds: list[float] = [1.0, 2.0, 4.0, 8.0],
    apply_superimpose: bool = True,
) -> Tensor:
    """
    Calculate GDT score between batched x and y coordinates.

    Reference:
     - https://en.wikipedia.org/wiki/Global_distance_test
    """
    # Ensure the mask exists; if not, create a mask of ones (include all residues)
    if mask is None:
        mask = torch.ones_like(x[..., 0], dtype=torch.bool)

    if apply_superimpose:
        y = superimpose(x, y, mask)

    # Calculate squared distances
    squared_distances = (x - y).pow(2).sum(dim=-1)  # Shape: [B, N]

    # Calculate scores for each threshold
    scores = []
    for threshold in thresholds:
        within_threshold = (squared_distances <= threshold**2).float()  # Shape: [B, N]
        score = masked_mean(within_threshold, mask, dim=-1)  # Average over residues
        scores.append(score)

    # Stack scores across thresholds and average them
    scores = torch.stack(scores, dim=-1)
    return scores.mean(dim=-1)  # Average over thresholds



def compute_single_metrics(pred_c4, gt_c4, mask):
    """
    pred_c4: [G, L, 3]
    gt_c4: [1, L, 3]
    mask: [1, L]
    """

    num_generated = pred_c4.shape[0]
    rmsd_list = torch.full((num_generated, ), float("nan"), dtype=pred_c4.dtype, device=pred_c4.device)
    tm_list = torch.full_like(rmsd_list, float("nan"))
    for j in range(num_generated):
        pred_c4_j = pred_c4[j : j + 1]
        try:
            pred_c4_aligned = superimpose(gt_c4, pred_c4_j, mask)
        except RuntimeError:
            pred_c4_aligned = pred_c4_j

        rmsd_j = rmsd(gt_c4, pred_c4_aligned, mask=mask)
        tm_j = tm_score(gt_c4, pred_c4_aligned, mask=mask, apply_superimpose=False)
        rmsd_list[j] = rmsd_j.squeeze(0)
        tm_list[j] = tm_j.squeeze(0)

    valid = ~torch.isnan(rmsd_list)
    best_rmsd = rmsd_list[valid].min() if valid.any() else rmsd_list[0]
    best_tm_score = tm_list[valid].max() if valid.any() else tm_list[0]
    return {"rmsd": best_rmsd, "tm_score": best_tm_score}


def pairwise_rmsd_matrix(coords, mask):
    """
    coords: [N, L, 3]
    mask: [L] or [1, L]
    Returns: [N, N] symmetric RMSD matrix (with superimposition)
    """
    N = coords.shape[0]
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)  # [1, L]
    matrix = torch.zeros(N, N, device=coords.device)
    for i in range(N):
        for j in range(i + 1, N):
            aligned_j = superimpose(coords[i:i+1], coords[j:j+1], mask)
            val = rmsd(coords[i:i+1], aligned_j, mask=mask).item()
            matrix[i, j] = val
            matrix[j, i] = val
    return matrix


def compute_ensemble_metrics(pred_c4, gt_c4, mask, deltas=(5.0, 6.0, 7.0)):
    """
    pred_c4: [G, L, 3] - generated conformers
    gt_c4: [K, L, 3] - ground truth conformers
    mask: [L] - residue mask
    deltas: RMSD thresholds for coverage metrics
    """
    G = pred_c4.shape[0]
    K = gt_c4.shape[0]

    if mask.dim() == 1:
        mask = mask.unsqueeze(0)  # [1, L]

    # RMSD matrix [G, K]
    rmsd_matrix = torch.zeros(G, K, device=pred_c4.device)
    for g in range(G):
        for k in range(K):
            aligned = superimpose(gt_c4[k:k+1], pred_c4[g:g+1], mask)
            rmsd_matrix[g, k] = rmsd(gt_c4[k:k+1], aligned, mask=mask)

    min_rmsd_per_gt = rmsd_matrix.min(dim=0).values      # [K]
    min_rmsd_per_pred = rmsd_matrix.min(dim=1).values     # [G]

    result = {
        "amr_recall": min_rmsd_per_gt.mean().item(),
        "amr_precision": min_rmsd_per_pred.mean().item(),
    }

    for d in deltas:
        result[f"cov_recall_{d}"] = (min_rmsd_per_gt <= d).float().mean().item()
        result[f"cov_precision_{d}"] = (min_rmsd_per_pred <= d).float().mean().item()

    # Pairwise RMSD among generated samples
    if G >= 2:
        pw_matrix = pairwise_rmsd_matrix(pred_c4, mask)
        triu_idx = torch.triu_indices(G, G, offset=1)
        result["pairwise_rmsd"] = pw_matrix[triu_idx[0], triu_idx[1]].mean().item()
    else:
        result["pairwise_rmsd"] = 0.0

    return result


def per_residue_variance(pred_c4: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Per-residue positional variance across an aligned ensemble.

    All G conformers are Kabsch-aligned to conformer 0, then
        σ²_i = (1/G) Σ_g ‖x_{g,i} − x̄_i‖²    ∈ ℝ,  x̄_i = (1/G) Σ_g x_{g,i}.
    The result has shape [L]. Residues outside `mask` still receive a value
    (alignment is mask-aware, variance is computed for all residues).
    """
    G, L, _ = pred_c4.shape
    if G < 2:
        return torch.zeros(L, device=pred_c4.device, dtype=pred_c4.dtype)
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)
    ref = pred_c4[0:1].expand(G, -1, -1)                       # [G, L, 3]
    align_mask = mask.expand(G, -1) if mask is not None else None
    try:
        aligned = superimpose(ref, pred_c4, align_mask)        # [G, L, 3]
    except RuntimeError:
        aligned = pred_c4
    mean_pos = aligned.mean(dim=0, keepdim=True)                # [1, L, 3]
    return (aligned - mean_pos).pow(2).sum(dim=-1).mean(dim=0)  # [L]


def _rankdata_avg(x: Tensor) -> Tensor:
    """Average-rank (ties get the mean of their block), like scipy.rankdata."""
    order = torch.argsort(x, stable=True)
    n = x.numel()
    ranks = torch.empty(n, dtype=torch.float64, device=x.device)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float64, device=x.device)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def spearman_rho(x: Tensor, y: Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    rx = _rankdata_avg(x.flatten().double())
    ry = _rankdata_avg(y.flatten().double())
    rx -= rx.mean()
    ry -= ry.mean()
    den = torch.sqrt(rx.pow(2).sum() * ry.pow(2).sum())
    if den.item() == 0.0:
        return float("nan")
    return float((rx * ry).sum() / den)


def local_rmsf(
    coords: Tensor,
    window: int = 6,
    n_ref: int = 5,
    seed: int = 0,
) -> Tensor:
    """Per-residue *local* RMSF in the Viliuga et al. (2025) sense.

    For each residue i we locally Kabsch-align a window W_i = {i-S/2..i+S/2}
    to a reference conformer and measure how far residue i has moved. This
    removes the "a distal loop wiggle makes the whole helix look flexible"
    artefact of global RMSF.

    Args:
        coords: [N, L, 3] — N conformers (aligned or unaligned).
        window: S — number of neighbours used for the local alignment.
        n_ref:  number of reference conformers; per-residue RMSF is the
                median across references (reduces reference-choice bias).
        seed:   RNG seed for picking reference conformers.

    Returns:
        xi: [L] per-residue local RMSF in Å. Boundary residues with <3
            valid neighbours (can't do Kabsch) get 0.
    """
    N, L, _ = coords.shape
    if N < 2:
        return torch.zeros(L, device=coords.device, dtype=coords.dtype)
    half = window // 2
    n_ref = min(n_ref, N)

    g = torch.Generator(device="cpu").manual_seed(seed)
    ref_idx = torch.randperm(N, generator=g)[:n_ref].tolist()

    # For every residue i, build the window indices once.
    windows = [
        list(range(max(0, i - half), min(L, i + half + 1)))
        for i in range(L)
    ]

    xi_per_ref = torch.zeros(n_ref, L, device=coords.device, dtype=coords.dtype)
    for r, ref in enumerate(ref_idx):
        ref_coords = coords[ref]                           # [L, 3]
        for i, idxs in enumerate(windows):
            if len(idxs) < 3:                              # not enough points for Kabsch
                continue
            idxs_t = torch.tensor(idxs, device=coords.device)
            # Stack windows across all N conformers in one Kabsch call:
            fixed = ref_coords[idxs_t].unsqueeze(0).expand(N, -1, -1)   # [N, |W|, 3]
            mobile = coords[:, idxs_t]                                    # [N, |W|, 3]
            try:
                aligned = superimpose(fixed, mobile)                      # [N, |W|, 3]
            except RuntimeError:
                continue
            # Position of residue i within the window:
            local_i = idxs.index(i)
            disp = (aligned[:, local_i] - ref_coords[i]).pow(2).sum(-1)   # [N]
            xi_per_ref[r, i] = disp.mean().sqrt()
    # median over references (Viliuga §2.2): robust to reference choice.
    return xi_per_ref.median(dim=0).values                                # [L]


def compute_flex_profile_metrics(
    xi_pred: Tensor,
    xi_ref: Tensor,
    mask: Optional[Tensor] = None,
    prefix: str = "flex",
) -> dict:
    """Pearson r / Spearman ρ / MAE / composite score between two per-residue
    flexibility profiles, aggregated over the masked residues.

    Viliuga et al. (2025) §4.1 report r, MAE, and a combined score
        s = w_r · r − w_mae · MAE,
    with w_r = 1, w_mae = 2. r captures profile *shape*; MAE captures
    *amplitude* in physical units (both profiles in Å).

    Returns keys `{prefix}_pearson`, `{prefix}_spearman`, `{prefix}_mae`,
    `{prefix}_score`, `{prefix}_amp_ratio`.
    """
    xi_pred = xi_pred.flatten().float()
    xi_ref = xi_ref.flatten().float()
    if mask is None:
        sel = torch.ones_like(xi_pred, dtype=torch.bool)
    else:
        sel = mask.flatten().bool()
    sel = sel & torch.isfinite(xi_pred) & torch.isfinite(xi_ref)
    # Drop residues with zero reference (unsolved in anchor B-factor convention).
    sel = sel & (xi_ref > 0)
    if int(sel.sum().item()) < 3:
        nan = float("nan")
        return {
            f"{prefix}_pearson": nan, f"{prefix}_spearman": nan,
            f"{prefix}_mae": nan, f"{prefix}_score": nan,
            f"{prefix}_amp_ratio": nan,
        }

    xp = xi_pred[sel]
    xr = xi_ref[sel]

    # Pearson
    xp_c = xp - xp.mean()
    xr_c = xr - xr.mean()
    den = torch.sqrt(xp_c.pow(2).sum() * xr_c.pow(2).sum())
    pearson = float((xp_c * xr_c).sum() / den) if den.item() > 0 else float("nan")

    spearman = spearman_rho(xp, xr)

    mae = float((xp - xr).abs().mean())
    amp_ratio = float(xp.mean() / xr.mean()) if xr.mean().item() > 0 else float("nan")

    score = (pearson - 2.0 * mae) if (pearson == pearson) else float("nan")  # NaN guard

    return {
        f"{prefix}_pearson": pearson,
        f"{prefix}_spearman": spearman,
        f"{prefix}_mae": mae,
        f"{prefix}_score": score,
        f"{prefix}_amp_ratio": amp_ratio,
    }


def compute_flexibility_b_correlation(
    pred_c4: Tensor, b_factors: Tensor, mask: Optional[Tensor] = None
) -> dict:
    """ρ(σ̂²_i, B_i): Spearman correlation between per-residue ensemble
    variance from the generated conformers and GT crystallographic B-factors.

    Tests whether the generated samples are consistent with physical
    flexibility (Table 2 of the paper).

    pred_c4: [G, L, 3] generated conformers
    b_factors: [L] per-residue B-factor (0 marks unsolved — dropped here)
    mask: [L] residue mask
    """
    sigma_sq = per_residue_variance(pred_c4, mask)      # [L]
    b = b_factors.flatten().float().to(sigma_sq.device)  # [L]

    sel = mask.flatten().bool() if mask is not None else torch.ones_like(b, dtype=torch.bool)
    sel = sel & (b > 0) & torch.isfinite(sigma_sq) & torch.isfinite(b)
    if int(sel.sum().item()) < 3:
        return {"rho_sigma_b": float("nan"), "sigma_sq_mean": float("nan")}

    return {
        "rho_sigma_b": spearman_rho(sigma_sq[sel], b[sel]),
        "sigma_sq_mean": float(sigma_sq[sel].mean().item()),
    }


def calc_rna_c4_c4_metrics(c4_pos, bond_tol=0.1, clash_tol=1.0):
    '''
    Compute the following metrics:
        - bond length deviation
        - valid bonds
        - clashes
        - radgyr
    '''

    c4_bond_dists = np.linalg.norm(c4_pos - np.roll(c4_pos, 1, axis=0), axis=-1)[1:]
    avg_c4_bond_dist = np.linalg.norm(c4_pos - np.roll(c4_pos, 1, axis=0), axis=-1)[1:].mean()
    c4_c4_dev = np.mean(np.abs(c4_bond_dists - nucleotide_constants.c4_c4))
    c4_c4_valid = np.mean(c4_bond_dists < (nucleotide_constants.c4_c4 + bond_tol))

    c4_c4_dists2d = np.linalg.norm(c4_pos[:, None, :] - c4_pos[None, :, :], axis=-1)
    inter_dists = c4_c4_dists2d[np.where(np.triu(c4_c4_dists2d, k=0) > 0)]
    clashes = inter_dists < clash_tol

    c4_pos = torch.from_numpy(c4_pos).unsqueeze(0)
    rad_gyr = gyration_radius(x=c4_pos, masses=12.011 * torch.ones_like(c4_pos[..., 0], dtype=torch.float32)).numpy()[0]

    return {
        'avg_c4_bond_dists': avg_c4_bond_dist,
        'c4_c4_deviation': c4_c4_dev,
        'c4_c4_valid_percent': c4_c4_valid,
        'num_c4_c4_clashes': np.sum(clashes),
        'radius_of_gyration': rad_gyr
    }
