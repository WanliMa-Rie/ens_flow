from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rna_backbone_design.analysis import metrics


def _pairwise_rmsd_matrix(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    k = int(coords.shape[0])

    mask_bool = mask.to(coords.device).bool()
    mask_b = mask_bool[None, :].expand(k, -1)
    mask_b_f = mask_b.to(coords.dtype)

    out = torch.empty((k, k), device=coords.device, dtype=coords.dtype)
    for i in range(k):
        fixed = coords[i : i + 1].expand(k, -1, -1)
        try:
            aligned = metrics.superimpose(fixed, coords, mask=mask_b)
            row = metrics.rmsd(fixed, aligned, mask=mask_b_f)
        except RuntimeError:
            row = metrics.rmsd(fixed, coords, mask=mask_b_f)
        out[i] = row

    out = 0.5 * (out + out.transpose(0, 1))
    out[torch.arange(k, device=coords.device), torch.arange(k, device=coords.device)] = 0.0
    return out


def pairwise_rmsd_mean(coords: torch.Tensor, mask: torch.Tensor) -> float:
    k = coords.shape[0]
    if k < 2:
        return float("nan")
    mat = _pairwise_rmsd_matrix(coords, mask)
    triu = torch.triu_indices(k, k, offset=1, device=coords.device)
    return float(mat[triu[0], triu[1]].mean().detach().cpu().numpy())


def wasserstein_distance_equal_weight(
    distmat: np.ndarray, p: int = 2
) -> float:
    if distmat.ndim != 2:
        raise ValueError(f"distmat must be 2D, got shape {distmat.shape}")
    if distmat.shape[0] != distmat.shape[1]:
        raise ValueError(f"distmat must be square, got shape {distmat.shape}")
    cost = distmat.astype(np.float64) ** p
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        return float((cost[row_ind, col_ind].mean()) ** (1.0 / p))
    except Exception:
        pass

    try:
        import networkx as nx

        k = int(cost.shape[0])
        left = [f"l{i}" for i in range(k)]
        right = [f"r{j}" for j in range(k)]
        G = nx.Graph()
        G.add_nodes_from(left, bipartite=0)
        G.add_nodes_from(right, bipartite=1)
        for i in range(k):
            for j in range(k):
                G.add_edge(left[i], right[j], weight=float(cost[i, j]))
        matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(
            G, top_nodes=left
        )
        picked = [float(cost[i, int(matching[left[i]][1:])]) for i in range(k)]
        return float((np.mean(picked)) ** (1.0 / p))
    except Exception as e:
        raise RuntimeError(
            "Unable to compute equal-weight Wp matching; install scipy or ensure networkx bipartite matching is available."
        ) from e


def pairwise_rmsd_pearson_r(gt_coords: torch.Tensor, pred_coords: torch.Tensor, mask: torch.Tensor) -> float:
    if gt_coords.shape != pred_coords.shape:
        raise ValueError(
            f"gt_coords and pred_coords must have same shape, got {tuple(gt_coords.shape)} vs {tuple(pred_coords.shape)}"
        )
    k = gt_coords.shape[0]
    if k < 2:
        return float("nan")
    gt_mat = _pairwise_rmsd_matrix(gt_coords, mask)
    pred_mat = _pairwise_rmsd_matrix(pred_coords, mask)
    triu = torch.triu_indices(k, k, offset=1, device=gt_coords.device)
    x = gt_mat[triu[0], triu[1]]
    y = pred_mat[triu[0], triu[1]]
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if float(denom.detach().cpu().numpy()) == 0.0:
        return float("nan")
    r = (x * y).mean() / denom
    return float(r.detach().cpu().numpy())


def _cross_rmsd_matrix(gt_coords: torch.Tensor, pred_coords: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    if gt_coords.ndim != 3 or pred_coords.ndim != 3:
        raise ValueError("gt_coords and pred_coords must be [K, L, 3]")
    if gt_coords.shape != pred_coords.shape:
        raise ValueError(
            f"gt_coords and pred_coords must have same shape, got {tuple(gt_coords.shape)} vs {tuple(pred_coords.shape)}"
        )
    if mask.ndim != 1:
        raise ValueError(f"mask must be [L], got {tuple(mask.shape)}")
    k = int(gt_coords.shape[0])
    l = int(gt_coords.shape[1])
    if l != int(mask.shape[0]):
        raise ValueError(f"mask length mismatch: L={l} vs mask={int(mask.shape[0])}")

    mask_b = mask.to(gt_coords.device).bool()[None, :].expand(k, -1)
    mask_b_f = mask_b.to(gt_coords.dtype)
    out = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        fixed = gt_coords[i : i + 1].expand(k, -1, -1)
        try:
            aligned = metrics.superimpose(fixed, pred_coords, mask=mask_b)
            row = metrics.rmsd(fixed, aligned, mask=mask_b_f)
        except RuntimeError:
            row = metrics.rmsd(fixed, pred_coords, mask=mask_b_f)
        out[i] = row.detach().cpu().numpy()
    return out


@dataclass(frozen=True)
class EnsembleMetrics:
    pairwise_rmsd: float
    w2_distance: float
    pairwise_rmsd_r: float


def compute_ensemble_metrics(
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    mask: torch.Tensor,
) -> EnsembleMetrics:
    if gt_coords.ndim != 3 or pred_coords.ndim != 3:
        raise ValueError("gt_coords and pred_coords must be [K, L, 3]")
    if mask.ndim != 1:
        raise ValueError("mask must be [L]")
    if mask.sum().item() < 3:
        return EnsembleMetrics(float("nan"), float("nan"), float("nan"))

    k = int(gt_coords.shape[0])
    if k < 2:
        return EnsembleMetrics(float("nan"), float("nan"), float("nan"))

    pairwise_rmsd = pairwise_rmsd_mean(pred_coords, mask)
    cross = _cross_rmsd_matrix(gt_coords, pred_coords, mask)
    w2 = wasserstein_distance_equal_weight(cross, p=2)
    pairwise_r = pairwise_rmsd_pearson_r(gt_coords, pred_coords, mask)
    return EnsembleMetrics(pairwise_rmsd=pairwise_rmsd, w2_distance=w2, pairwise_rmsd_r=pairwise_r)
