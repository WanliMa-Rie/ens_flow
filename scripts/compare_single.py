#!/usr/bin/env python3
"""
Compare generated RNA structures against ground truth using RMSD metrics.

This script:
1. Loads test_single clusters from test_clusters.json
2. For each cluster, finds the ground truth PDB and compares against 100 generated conformations
3. Computes Top-1, Top-3, Top-5 RMSD metrics (best K samples averaged)
4. Supports multiple model directories (RhoFold, rnafm, etc.)
5. Outputs results to compare_single/compare.csv

Uses existing utilities from rna_backbone_design for PDB parsing and RMSD calculation.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Use existing utilities from the codebase
from rna_backbone_design.simple_pdb_parser import get_pdb_features
from rna_backbone_design.data import nucleotide_constants
from rna_backbone_design.analysis import metrics


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
    
    # Get C4' atom index from nucleotide constants
    # Note: simple_pdb_parser returns compact features (N, 23, 3) where C4' is at index 1.
    # However, nucleotide_constants.atom_order['C4\''] gives index 3 (standard 27-atom schema).
    # We must check the shape to decide which index to use.
    
    if atom_positions.shape[1] == 23:
        # Compact features from simple_pdb_parser
        c4_idx = 1
    else:
        # Standard features or other
        c4_idx = nucleotide_constants.atom_order["C4'"]
    
    # Extract C4' coordinates: atom_positions has shape (N_res, N_atoms, 3)
    c4_coords = atom_positions[:, c4_idx, :]
    
    return c4_coords


def compute_rmsd_with_superimpose(gt_coords: np.ndarray, pred_coords: np.ndarray) -> float:
    """
    Compute RMSD between two sets of coordinates using Kabsch superimposition.
    
    Uses existing metrics.superimpose() and metrics.rmsd() functions.
    
    Args:
        gt_coords: Ground truth coordinates, shape (N, 3)
        pred_coords: Predicted coordinates, shape (N, 3)
        
    Returns:
        RMSD value in Angstroms
    """
    # Convert to torch tensors with batch dimension
    gt_tensor = torch.from_numpy(gt_coords).unsqueeze(0).float()  # (1, N, 3)
    pred_tensor = torch.from_numpy(pred_coords).unsqueeze(0).float()  # (1, N, 3)
    
    # Superimpose predicted onto ground truth
    aligned_pred = metrics.superimpose(gt_tensor, pred_tensor)
    
    # Compute RMSD
    rmsd_value = metrics.rmsd(gt_tensor, aligned_pred)
    
    return float(rmsd_value.item())


def get_ground_truth_pdb(cluster_full_name: str, ensemble_data_dir: str) -> Optional[str]:
    """
    Find the ground truth PDB file for a cluster.
    
    For test_single, we use the first structure in the cluster's structure directory.
    
    Args:
        cluster_full_name: Full cluster name (e.g., "cluster_0287_2az0_C")
        ensemble_data_dir: Path to rna_ensemble_data directory
        
    Returns:
        Path to ground truth PDB file, or None if not found
    """
    structure_dir = Path(ensemble_data_dir) / cluster_full_name / "structure"
    if not structure_dir.exists():
        return None
    
    pdb_files = list(structure_dir.glob("*.pdb"))
    if not pdb_files:
        return None
    
    # Return the first PDB file as ground truth
    return str(pdb_files[0])


def get_generated_pdbs(cluster_id: str, cluster_full_name: str, model_dir: str) -> list[str]:
    """
    Get list of generated PDB files for a cluster.
    
    Supports both short format (cluster_0287) and full format (cluster_0287_2az0_C).
    
    Args:
        cluster_id: Short cluster ID (e.g., "cluster_0287")
        cluster_full_name: Full cluster name (e.g., "cluster_0287_2az0_C")
        model_dir: Path to model output directory
        
    Returns:
        List of paths to generated PDB files, sorted by conformation number
    """
    # Try short format first (e.g., rhofold uses cluster_0287/)
    cluster_dir = Path(model_dir) / cluster_id
    if not cluster_dir.exists():
        # Try full format (e.g., rnafm uses cluster_0287_2az0_C/)
        cluster_dir = Path(model_dir) / cluster_full_name
    
    if not cluster_dir.exists():
        return []
    
    # Search for PDB files with various naming patterns
    pdb_files = []
    
    # Pattern 1: cluster_XXXX_confN.pdb (short format)
    pdb_files.extend(cluster_dir.glob(f"{cluster_id}_conf*.pdb"))
    
    # Pattern 2: cluster_XXXX_YYYY_Z_confN.pdb (full format)
    if not pdb_files:
        pdb_files.extend(cluster_dir.glob(f"{cluster_full_name}_conf*.pdb"))
    
    # Pattern 3: confN.pdb or sample_N.pdb
    if not pdb_files:
        pdb_files.extend(cluster_dir.glob("conf*.pdb"))
        pdb_files.extend(cluster_dir.glob("sample_*.pdb"))
    
    # Pattern 4: Any .pdb file
    if not pdb_files:
        pdb_files.extend(cluster_dir.glob("*.pdb"))
    
    # Sort by conformation number
    def get_conf_num(path: Path) -> int:
        name = path.stem
        # Try to extract number from various patterns
        import re
        match = re.search(r'(?:conf|sample_?)?(\d+)$', name)
        if match:
            return int(match.group(1))
        return 0
    
    pdb_files = list(set(pdb_files))  # Remove duplicates
    pdb_files.sort(key=get_conf_num)
    return [str(p) for p in pdb_files]


def compute_cluster_rmsd(gt_pdb: str, generated_pdbs: list[str]) -> list[float]:
    """
    Compute RMSD for all generated structures against ground truth.
    
    Args:
        gt_pdb: Path to ground truth PDB
        generated_pdbs: List of paths to generated PDB files
        
    Returns:
        List of RMSD values
    """
    gt_coords = extract_c4_coords(gt_pdb)
    if gt_coords is None:
        print(f"Error: Could not extract C4' coords from {gt_pdb}")
        return []
    
    rmsds = []
    for gen_pdb in generated_pdbs:
        try:
            gen_coords = extract_c4_coords(gen_pdb)
            
            if gen_coords is None:
                print(f"Warning: Could not extract C4' coords from {gen_pdb}")
                rmsds.append(float('inf'))
                continue
            
            # Ensure same number of atoms - use minimum length if mismatched
            if len(gt_coords) != len(gen_coords):
                min_len = min(len(gt_coords), len(gen_coords))
                rmsd = compute_rmsd_with_superimpose(gt_coords[:min_len], gen_coords[:min_len])
            else:
                rmsd = compute_rmsd_with_superimpose(gt_coords, gen_coords)
            
            rmsds.append(rmsd)
        except Exception as e:
            print(f"Error processing {gen_pdb}: {e}")
            rmsds.append(float('inf'))
    
    return rmsds


def compute_top_k_rmsd(rmsds: list[float], k: int) -> float:
    """
    Compute mean RMSD of top-K (best K) samples.
    
    Args:
        rmsds: List of RMSD values
        k: Number of best samples to average
        
    Returns:
        Mean of best K RMSDs
    """
    if not rmsds:
        return float('inf')
    
    sorted_rmsds = sorted(rmsds)
    return np.mean(sorted_rmsds[:k])


def main():
    # Configuration
    inference_dir = Path("/projects/u6bk/wanli/inference")
    ensemble_data_dir = Path("/projects/u6bk/wanli/rna_ensemble_data")
    test_clusters_path = inference_dir / "test_clusters.json"
    output_dir = inference_dir / "compare_single"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test clusters
    with open(test_clusters_path, 'r') as f:
        clusters_data = json.load(f)
    
    test_single_clusters = clusters_data["test_single"]
    print(f"Found {len(test_single_clusters)} test_single clusters")
    
    # Discover model directories (exclude compare_single and files)
    model_dirs = []
    for item in inference_dir.iterdir():
        if item.is_dir() and item.name not in ["compare_single"]:
            # Check if it contains cluster folders (short or full format)
            has_clusters = False
            for c in test_single_clusters[:5]:
                parts = c.split("_")
                short_name = f"{parts[0]}_{parts[1]}"  # cluster_0287
                if (item / short_name).exists() or (item / c).exists():
                    has_clusters = True
                    break
            if has_clusters:
                model_dirs.append(item)
    
    print(f"Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}")
    
    # Top-K values to compute
    top_k_values = [1, 3, 5, 10, 20, 50, 100]
    
    # Length bins: (min, max, label)
    length_bins = [
        (0, 40, "0-40"),
        (41, 80, "41-80"),
        (81, 120, "81-120"),
        (121, 160, "121-160"),
        (161, 200, "161-200"),
        (201, float('inf'), "201+"),
    ]
    
    def get_length_bin(length: int) -> str:
        """Get the bin label for a given length."""
        for min_len, max_len, label in length_bins:
            if min_len <= length <= max_len:
                return label
        return "other"
    
    # First pass: get cluster lengths
    cluster_lengths = {}
    for cluster_full_name in test_single_clusters:
        gt_pdb = get_ground_truth_pdb(cluster_full_name, str(ensemble_data_dir))
        if gt_pdb:
            coords = extract_c4_coords(gt_pdb)
            cluster_lengths[cluster_full_name] = len(coords) if coords is not None else 0
        else:
            cluster_lengths[cluster_full_name] = 0
    
    # Print length distribution
    print("\nCluster length distribution:")
    for min_len, max_len, label in length_bins:
        count = sum(1 for l in cluster_lengths.values() if min_len <= l <= max_len)
        print(f"  {label}: {count} clusters")
    
    # Results: model -> bin -> list of top-k values per cluster
    # Structure: {model_name: {bin_label: {k: [rmsd_values]}}}
    all_results = {}
    
    # Process each model
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\nProcessing model: {model_name}")
        
        # Initialize result structure for this model
        all_results[model_name] = {label: {k: [] for k in top_k_values} 
                                    for _, _, label in length_bins}
        all_results[model_name]["all"] = {k: [] for k in top_k_values}
        
        for cluster_full_name in test_single_clusters:
            # Extract cluster ID (e.g., "cluster_0287" from "cluster_0287_2az0_C")
            parts = cluster_full_name.split("_")
            cluster_id = f"{parts[0]}_{parts[1]}"
            
            # Get ground truth
            gt_pdb = get_ground_truth_pdb(cluster_full_name, str(ensemble_data_dir))
            if gt_pdb is None:
                print(f"  Warning: No ground truth found for {cluster_full_name}")
                continue
            
            # Get generated structures
            generated_pdbs = get_generated_pdbs(cluster_id, cluster_full_name, str(model_dir))
            if not generated_pdbs:
                print(f"  Warning: No generated structures found for {cluster_id}")
                continue
            
            # Compute RMSD for all conformations
            rmsds = compute_cluster_rmsd(gt_pdb, generated_pdbs)
            
            if not rmsds:
                continue
            
            # Get length bin for this cluster
            length = cluster_lengths.get(cluster_full_name, 0)
            bin_label = get_length_bin(length)
            
            # Compute Top-K metrics
            for k in top_k_values:
                if k <= len(rmsds):
                    top_k_rmsd = compute_top_k_rmsd(rmsds, k)
                    all_results[model_name][bin_label][k].append(top_k_rmsd)
                    all_results[model_name]["all"][k].append(top_k_rmsd)
            
            print(f"  {cluster_id} (len={length}, bin={bin_label}): "
                  f"Best RMSD = {min(rmsds):.2f}Å, Mean RMSD = {np.mean(rmsds):.2f}Å")
    
    # Generate CSV files for each length bin (+ overall)
    all_bins = ["all"] + [label for _, _, label in length_bins]
    
    for bin_label in all_bins:
        results = {
            "Model": [],
        }
        for k in top_k_values:
            results[f"Top-{k}"] = []
        
        has_data = False
        for model_name in all_results:
            if bin_label in all_results[model_name]:
                bin_data = all_results[model_name][bin_label]
                # Check if there's any data for this bin
                if any(bin_data[k] for k in top_k_values):
                    has_data = True
                    results["Model"].append(model_name)
                    for k in top_k_values:
                        if bin_data[k]:
                            results[f"Top-{k}"].append(np.mean(bin_data[k]))
                        else:
                            results[f"Top-{k}"].append(float('nan'))
        
        if has_data:
            df = pd.DataFrame(results)
            if bin_label == "all":
                output_path = output_dir / "compare.csv"
            else:
                output_path = output_dir / f"compare_len_{bin_label}.csv"
            df.to_csv(output_path, index=False, float_format="%.3f")
            print(f"\nResults for length bin '{bin_label}' saved to: {output_path}")
            print(df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
