#!/usr/bin/env python3
"""
Evaluate ensemble generation against multi-state ground truth clusters.

Metrics:
  - Recall (Coverage): fraction of GT states with min RMSD < {4, 8, 10} Å
  - Fidelity: mean min RMSD across GT states
  - Balance: min assigned fraction across GT states (anti-collapse)

Outputs per-bin CSVs under compare_ensemble/.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN

from rna_backbone_design.analysis import metrics
from rna_backbone_design.data import nucleotide_constants
from rna_backbone_design.simple_pdb_parser import get_pdb_features


def extract_c4_coords(pdb_path: str) -> Optional[np.ndarray]:
    """
    Extract C4' atom coordinates from a PDB file using existing utilities.

    Args:
        pdb_path: Path to PDB file

    Returns:
        numpy array of shape (N, 3) with C4' coordinates, or None if extraction fails
    """
    feats = get_pdb_features(pdb_path, verbose=False)
    if feats is None:
        return None

    atom_positions = feats.get("atom_positions")
    if atom_positions is None:
        return None

    if atom_positions.shape[1] == 23:
        c4_idx = 1
    else:
        c4_idx = nucleotide_constants.atom_order["C4'"]

    return atom_positions[:, c4_idx, :]


def compute_rmsd_with_superimpose(gt_coords: np.ndarray, pred_coords: np.ndarray) -> float:
    """
    Compute RMSD between two sets of coordinates using Kabsch superimposition.
    """
    gt_tensor = torch.from_numpy(gt_coords).unsqueeze(0).float()
    pred_tensor = torch.from_numpy(pred_coords).unsqueeze(0).float()
    aligned_pred = metrics.superimpose(gt_tensor, pred_tensor)
    rmsd_value = metrics.rmsd(gt_tensor, aligned_pred)
    return float(rmsd_value.item())


def compute_rmsd_any_length(gt_coords: np.ndarray, pred_coords: np.ndarray) -> float:
    if len(gt_coords) != len(pred_coords):
        min_len = min(len(gt_coords), len(pred_coords))
        return compute_rmsd_with_superimpose(gt_coords[:min_len], pred_coords[:min_len])
    return compute_rmsd_with_superimpose(gt_coords, pred_coords)


def get_ground_truth_pdbs(cluster_full_name: str, ensemble_data_dir: str) -> list[str]:
    """Return all GT PDB paths for a cluster (multi-state)."""
    structure_dir = Path(ensemble_data_dir) / cluster_full_name / "structure"
    if not structure_dir.exists():
        return []
    pdb_files = sorted(structure_dir.glob("*.pdb"))
    return [str(p) for p in pdb_files]


def get_generated_pdbs(cluster_id: str, cluster_full_name: str, model_dir: str) -> list[str]:
    """
    Get list of generated PDB files for a cluster.
    """
    cluster_dir = Path(model_dir) / cluster_id
    if not cluster_dir.exists():
        cluster_dir = Path(model_dir) / cluster_full_name

    if not cluster_dir.exists():
        return []

    pdb_files = []
    pdb_files.extend(cluster_dir.glob(f"{cluster_id}_conf*.pdb"))
    if not pdb_files:
        pdb_files.extend(cluster_dir.glob(f"{cluster_full_name}_conf*.pdb"))
    if not pdb_files:
        pdb_files.extend(cluster_dir.glob("conf*.pdb"))
        pdb_files.extend(cluster_dir.glob("sample_*.pdb"))
    if not pdb_files:
        pdb_files.extend(cluster_dir.glob("*.pdb"))

    def get_conf_num(path: Path) -> int:
        import re

        name = path.stem
        match = re.search(r"(?:conf|sample_?)?(\d+)$", name)
        if match:
            return int(match.group(1))
        return 0

    pdb_files = list(set(pdb_files))
    pdb_files.sort(key=get_conf_num)
    return [str(p) for p in pdb_files]


def compute_cross_rmsd_matrix(
    gt_coords_list: list[np.ndarray],
    pred_coords_list: list[np.ndarray],
) -> np.ndarray:
    k = len(gt_coords_list)
    n = len(pred_coords_list)
    out = np.full((k, n), np.inf, dtype=np.float64)
    for i, gt_coords in enumerate(gt_coords_list):
        for j, pred_coords in enumerate(pred_coords_list):
            out[i, j] = compute_rmsd_any_length(gt_coords, pred_coords)
    return out


def compute_pairwise_rmsd_matrix(coords_list: list[np.ndarray]) -> np.ndarray:
    n = len(coords_list)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            val = compute_rmsd_any_length(coords_list[i], coords_list[j])
            out[i, j] = val
            out[j, i] = val
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ensemble generation vs GT states")
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="/projects/u6bk/wanli/inference",
        help="Path to inference outputs (contains test_clusters.json)",
    )
    parser.add_argument(
        "--ensemble_data_dir",
        type=str,
        default="/projects/u6bk/wanli/rna_ensemble_data",
        help="Path to rna_ensemble_data",
    )
    parser.add_argument(
        "--balance_delta",
        type=float,
        default=10.0,
        help="RMSD threshold for balance assignment (Angstrom)",
    )
    parser.add_argument(
        "--phase3_only",
        action="store_true",
        help="If set, only compute and save Phase 3 (DBSCAN) metrics table.",
    )
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    ensemble_data_dir = Path(args.ensemble_data_dir)
    test_clusters_path = inference_dir / "test_clusters.json"
    output_dir = inference_dir / "compare_ensemble"

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(test_clusters_path, "r") as f:
        clusters_data = json.load(f)
    test_ensemble_clusters = clusters_data["test_ensemble"]
    print(f"Found {len(test_ensemble_clusters)} test_ensemble clusters")

    allowed_models = {"dynarna_ensemble", "rnafm_ensemble", "af3_ensemble", "rhofold_ensemble"}
    model_dirs = []
    for item in inference_dir.iterdir():
        if (
            item.is_dir()
            and item.name in allowed_models
            and item.name not in ["compare_single", "compare_ensemble"]
        ):
            has_clusters = False
            for c in test_ensemble_clusters[:5]:
                parts = c.split("_")
                short_name = f"{parts[0]}_{parts[1]}"
                if (item / short_name).exists() or (item / c).exists():
                    has_clusters = True
                    break
            if has_clusters:
                model_dirs.append(item)

    print(f"Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}")

    state_bins = [
        (0, 20, "0-20"),
        (21, 40, "21-40"),
        (41, 80, "41-80"),
        (81, float("inf"), "81+"),
    ]

    def get_state_bin(num_states: int) -> str:
        for min_s, max_s, label in state_bins:
            if min_s <= num_states <= max_s:
                return label
        return "other"

    cluster_gt_coords: dict[str, list[np.ndarray]] = {}
    cluster_state_counts: dict[str, int] = {}
    if not args.phase3_only:
        for cluster_full_name in test_ensemble_clusters:
            gt_pdbs = get_ground_truth_pdbs(cluster_full_name, str(ensemble_data_dir))
            coords_list = []
            for pdb_path in gt_pdbs:
                coords = extract_c4_coords(pdb_path)
                if coords is None:
                    print(f"Warning: Could not extract C4' coords from {pdb_path}")
                    continue
                coords_list.append(coords)
            cluster_gt_coords[cluster_full_name] = coords_list
            cluster_state_counts[cluster_full_name] = len(coords_list)

        print("\nCluster GT-state count distribution:")
        for min_s, max_s, label in state_bins:
            count = sum(1 for s in cluster_state_counts.values() if min_s <= s <= max_s)
            print(f"  {label}: {count} clusters")

    recall_thresholds = [4.0, 8.0, 10.0]
    all_results: dict[str, dict[str, dict[str, object]]] = {}
    phase3_epsilons = [2.0, 2.5, 3.0, 3.5, 4.0]
    phase3_results: dict[str, dict[float, dict[str, list[float]]]] = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\nProcessing model: {model_name}")

        if not args.phase3_only:
            all_results[model_name] = {}
            for _, _, label in state_bins:
                all_results[model_name][label] = {
                    "recall": {t: [] for t in recall_thresholds},
                    "fidelity": [],
                    "balance": [],
                }
            all_results[model_name]["all"] = {
                "recall": {t: [] for t in recall_thresholds},
                "fidelity": [],
                "balance": [],
            }
        phase3_results[model_name] = {
            eps: {"K": [], "intra_avg": [], "inter_avg": [], "sep_ratio": []} for eps in phase3_epsilons
        }

        for cluster_full_name in test_ensemble_clusters:
            parts = cluster_full_name.split("_")
            cluster_id = f"{parts[0]}_{parts[1]}"

            if not args.phase3_only:
                gt_coords_list = cluster_gt_coords.get(cluster_full_name, [])
                if not gt_coords_list:
                    print(f"  Warning: No GT states for {cluster_full_name}")
                    continue

            generated_pdbs = get_generated_pdbs(cluster_id, cluster_full_name, str(model_dir))
            if not generated_pdbs:
                print(f"  Warning: No generated structures found for {cluster_id}")
                continue

            pred_coords_all: list[Optional[np.ndarray]] = []
            valid_pred_indices: list[int] = []
            for idx, gen_pdb in enumerate(generated_pdbs):
                coords = extract_c4_coords(gen_pdb)
                if coords is None:
                    print(f"  Warning: Could not extract C4' coords from {gen_pdb}")
                    pred_coords_all.append(None)
                    continue
                pred_coords_all.append(coords)
                valid_pred_indices.append(idx)

            if not valid_pred_indices:
                print(f"  Warning: No valid generated structures for {cluster_id}")
                continue

            if not args.phase3_only:
                pred_coords_valid = [pred_coords_all[i] for i in valid_pred_indices]
                rmsd_mat = compute_cross_rmsd_matrix(gt_coords_list, pred_coords_valid)

                best_rmsd_gt = rmsd_mat.min(axis=1)
                best_rmsd = float(rmsd_mat.min())

                num_states = cluster_state_counts.get(cluster_full_name, 0)
                bin_label = get_state_bin(num_states)

                recall_by_t = {}
                for t in recall_thresholds:
                    recall_by_t[t] = (
                        float(np.mean(best_rmsd_gt < t)) if len(best_rmsd_gt) > 0 else float("nan")
                    )
                recall_for_balance = (
                    float(np.mean(best_rmsd_gt < args.balance_delta))
                    if len(best_rmsd_gt) > 0
                    else float("nan")
                )
                fidelity = float(np.mean(best_rmsd_gt)) if len(best_rmsd_gt) > 0 else float("nan")

                # Balance: for each GT, count samples within balance_delta; sum as N_assign,
                # then take the minimum fraction across GTs. If recall < 1, set to 0.
                within_delta = rmsd_mat < args.balance_delta
                counts = within_delta.sum(axis=1)
                n_assign = int(counts.sum())
                if recall_for_balance < 1:
                    balance = 0.0
                else:
                    balance = float(counts.min() / n_assign) if n_assign > 0 else float("nan")

                for t in recall_thresholds:
                    all_results[model_name][bin_label]["recall"][t].append(recall_by_t[t])
                all_results[model_name][bin_label]["fidelity"].append(fidelity)
                all_results[model_name][bin_label]["balance"].append(balance)
                for t in recall_thresholds:
                    all_results[model_name]["all"]["recall"][t].append(recall_by_t[t])
                all_results[model_name]["all"]["fidelity"].append(fidelity)
                all_results[model_name]["all"]["balance"].append(balance)

            pred_coords_valid = [pred_coords_all[i] for i in valid_pred_indices]
            if len(pred_coords_valid) >= 2:
                pred_pairwise = compute_pairwise_rmsd_matrix(pred_coords_valid)
                degenerate_pairwise = pred_pairwise.size <= 1 or np.allclose(pred_pairwise, 0.0)
            else:
                pred_pairwise = None
                degenerate_pairwise = True

            for eps in phase3_epsilons:
                if degenerate_pairwise:
                    # Skip clusters with only one valid sample (or degenerate distances)
                    continue

                labels = DBSCAN(eps=eps, min_samples=5, metric="precomputed").fit_predict(pred_pairwise)
                cluster_labels = [lbl for lbl in np.unique(labels) if lbl != -1]
                if not cluster_labels:
                    phase3_results[model_name][eps]["K"].append(0.0)
                    phase3_results[model_name][eps]["intra_avg"].append(0.0)
                    phase3_results[model_name][eps]["inter_avg"].append(0.0)
                    phase3_results[model_name][eps]["sep_ratio"].append(0.0)
                    continue

                clusters = [np.where(labels == lbl)[0] for lbl in cluster_labels]
                sizes = np.array([len(c) for c in clusters], dtype=np.float64)
                intra_vals = []
                for idxs in clusters:
                    if len(idxs) < 2:
                        intra_vals.append(0.0)
                        continue
                    sub = pred_pairwise[np.ix_(idxs, idxs)]
                    tri = sub[np.triu_indices_from(sub, k=1)]
                    intra_vals.append(float(tri.mean()) if tri.size > 0 else 0.0)
                intra_avg = float(np.mean(intra_vals)) if intra_vals else 0.0

                if len(clusters) < 2:
                    inter_avg = 0.0
                else:
                    inter_vals = []
                    for i in range(len(clusters)):
                        for j in range(i + 1, len(clusters)):
                            sub = pred_pairwise[np.ix_(clusters[i], clusters[j])]
                            inter_vals.append(float(sub.min()))
                    inter_avg = float(np.mean(inter_vals)) if inter_vals else 0.0

                sep_ratio = (
                    float(inter_avg / intra_avg) if inter_avg > 0.0 and intra_avg > 0.0 else 0.0
                )
                phase3_results[model_name][eps]["K"].append(float(len(clusters)))
                phase3_results[model_name][eps]["intra_avg"].append(intra_avg)
                phase3_results[model_name][eps]["inter_avg"].append(inter_avg)
                phase3_results[model_name][eps]["sep_ratio"].append(sep_ratio)

            if not args.phase3_only:
                print(
                    f"  {cluster_id} (states={num_states}, bin={bin_label}): "
                    f"Best RMSD = {best_rmsd:.2f}A, "
                    f"Recall@4/8/10 = {recall_by_t[4.0]:.2f}/{recall_by_t[8.0]:.2f}/{recall_by_t[10.0]:.2f}, "
                    f"Fidelity = {fidelity:.2f}A, "
                    f"Balance = {balance:.3f}"
                )

    if not args.phase3_only:
        all_bins = ["all"] + [label for _, _, label in state_bins]
        for bin_label in all_bins:
            results = {"Model": [], "NumClusters": []}
            for t in recall_thresholds:
                results[f"Recall@{t}"] = []
            results["Fidelity"] = []
            results["Balance"] = []

            has_data = False
            for model_name in all_results:
                if bin_label not in all_results[model_name]:
                    continue
                bin_data = all_results[model_name][bin_label]
                if not bin_data["fidelity"]:
                    continue
                has_data = True
                results["Model"].append(model_name)
                num_clusters = len(bin_data["fidelity"])
                results["NumClusters"].append(num_clusters)
                for t in recall_thresholds:
                    vals = bin_data["recall"][t]
                    results[f"Recall@{t}"].append(float(np.mean(vals)) if vals else float("nan"))
                results["Fidelity"].append(
                    float(np.mean(bin_data["fidelity"])) if bin_data["fidelity"] else float("nan")
                )
                results["Balance"].append(
                    float(np.mean(bin_data["balance"])) if bin_data["balance"] else float("nan")
                )

            if has_data:
                df = pd.DataFrame(results)
                if bin_label == "all":
                    output_path = output_dir / "compare_ensemble.csv"
                else:
                    output_path = output_dir / f"compare_ensemble_states_{bin_label}.csv"
                df.to_csv(output_path, index=False, float_format="%.3f")
                print(f"\nResults for state-count bin '{bin_label}' saved to: {output_path}")
                print(df.to_string(index=False))

    phase3_table = {"Model": [], "NumClusters": []}
    for eps in phase3_epsilons:
        phase3_table[f"K@{eps:.1f}"] = []
        phase3_table[f"Intra_avg@{eps:.1f}"] = []
        phase3_table[f"Inter_avg@{eps:.1f}"] = []
        phase3_table[f"SepRatio@{eps:.1f}"] = []

    for model_name in phase3_results:
        phase3_table["Model"].append(model_name)
        if not args.phase3_only:
            phase3_table["NumClusters"].append(len(all_results[model_name]["all"]["fidelity"]))
        else:
            phase3_table["NumClusters"].append(len(phase3_results[model_name][phase3_epsilons[0]]["K"]))
        for eps in phase3_epsilons:
            metrics_list = phase3_results[model_name][eps]
            mean_k = float(np.mean(metrics_list["K"])) if metrics_list["K"] else float("nan")
            mean_intra = (
                float(np.mean(metrics_list["intra_avg"])) if metrics_list["intra_avg"] else float("nan")
            )
            mean_inter = (
                float(np.mean(metrics_list["inter_avg"])) if metrics_list["inter_avg"] else float("nan")
            )
            if metrics_list["sep_ratio"]:
                mean_sep_ratio = float(np.mean(metrics_list["sep_ratio"]))
            else:
                mean_sep_ratio = 0.0

            phase3_table[f"K@{eps:.1f}"].append(mean_k)
            phase3_table[f"Intra_avg@{eps:.1f}"].append(mean_intra)
            phase3_table[f"Inter_avg@{eps:.1f}"].append(mean_inter)
            phase3_table[f"SepRatio@{eps:.1f}"].append(mean_sep_ratio)

    phase3_df = pd.DataFrame(phase3_table)
    phase3_output_path = output_dir / "compare_ensemble_phase3.csv"
    phase3_df.to_csv(phase3_output_path, index=False, float_format="%.3f")
    print(f"\nPhase 3 results saved to: {phase3_output_path}")
    print(phase3_df.to_string(index=False))

    print("\n" + "=" * 60)
    if args.phase3_only:
        print(f"Phase 3 results saved to: {phase3_output_path}")
    else:
        print(f"All results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
