"""
Standalone evaluation script for RNA conformer generation methods.

Reads predicted structures from /projects/u6bk/wanli/inference_ours/{method}/
and ground truth from the preprocessed dataset.  Computes single-conformer
metrics (RMSD, TM-score) and ensemble metrics (AMR, coverage, pairwise RMSD)
and writes per-cluster + summary CSVs.

Usage:
    # Evaluate all methods
    python evaluate_metrics.py

    # Evaluate specific methods
    python evaluate_metrics.py --methods ens_flow3 af3 rhofold

    # Only test_ensemble split
    python evaluate_metrics.py --split test_ensemble
"""

import argparse
import os
import glob
import logging
import re
import numpy as np
import pandas as pd
import torch

from Bio.PDB import PDBParser, MMCIFParser

from rna_backbone_design.analysis.metrics import (
    compute_single_metrics,
    compute_ensemble_metrics,
)

# Same convention as rna_conformer_dataset._DECOY_CONFORMER_RE — synthetic
# completion models must not enter GT, validation, or eval. See that module
# for the motivation.
_DECOY_CONFORMER_RE = re.compile(r"_c\d+_model$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────
DATA_DIR = "/projects/u6bk/wanli/ensemble_dataset"
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed_data")
INFERENCE_ROOT = "/projects/u6bk/wanli/inference_ours"
SPLIT_JSON = os.path.join(DATA_DIR, "split_cdhit80.json")


# ── structure parsing ────────────────────────────────────────────────────
_pdb_parser = PDBParser(QUIET=True)
_cif_parser = MMCIFParser(QUIET=True)


def extract_c4_coords(filepath: str) -> np.ndarray:
    """Parse a PDB/CIF file and return C4' coordinates as [N_res, 3]."""
    if filepath.endswith(".cif"):
        structure = _cif_parser.get_structure("s", filepath)
    else:
        structure = _pdb_parser.get_structure("s", filepath)

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "C4'" in residue:
                    coords.append(residue["C4'"].get_vector().get_array())
        break  # first model only

    if len(coords) == 0:
        raise ValueError(f"No C4' atoms found in {filepath}")
    return np.array(coords, dtype=np.float32)


def load_pred_c4_for_cluster(cluster_dir: str) -> np.ndarray | None:
    """Load all generated conformers for a cluster.

    Returns [G, L, 3] tensor or None if no valid files found.
    """
    files = sorted(
        glob.glob(os.path.join(cluster_dir, "*_conf*.cif"))
        + glob.glob(os.path.join(cluster_dir, "*_conf*.pdb"))
    )
    if not files:
        return None

    all_coords = []
    for f in files:
        try:
            c4 = extract_c4_coords(f)
            all_coords.append(c4)
        except Exception as e:
            log.warning(f"Skipping {f}: {e}")

    if not all_coords:
        return None

    # All conformers in a cluster must have the same length
    lengths = {c.shape[0] for c in all_coords}
    if len(lengths) > 1:
        # Keep only the most common length
        from collections import Counter
        most_common_len = Counter(c.shape[0] for c in all_coords).most_common(1)[0][0]
        all_coords = [c for c in all_coords if c.shape[0] == most_common_len]
        log.warning(
            f"Mixed lengths in {cluster_dir}: {lengths}. "
            f"Keeping {len(all_coords)} conformers with length {most_common_len}."
        )

    return np.stack(all_coords, axis=0)  # [G, L, 3]


# ── ground truth loading ─────────────────────────────────────────────────
def load_gt_data(split_names: list[str]):
    """Load real-experimental GT records from preprocessed .pt files, grouped
    by cluster.

    Synthetic '_cNNN_model' completion decoys are filtered out so that metric
    aggregation is per cluster over real structures only — matching the
    validation-time semantics in rna_conformer_dataset / flow_module.

    Returns:
        gt_single: {cluster_name: c4_coords [L, 3]}  (anchor real conformer)
        gt_ensemble: {cluster_name: c4_coords [K, L, 3]}  (all real conformers)
        gt_mask: {cluster_name: res_mask [L]}
        cluster_split: {cluster_name: "test_single" | "test_ensemble"}
    """
    import json

    with open(SPLIT_JSON) as f:
        split_data = json.load(f)

    # Build cluster -> split mapping
    cluster_split: dict[str, str] = {}
    for s in split_names:
        for cname in split_data.get(s, []):
            cluster_split[cname] = s

    gt_single = {}    # cluster -> anchor C4' [L, 3]
    gt_ensemble = {}  # cluster -> all real C4' [K, L, 3]
    gt_mask = {}      # cluster -> res_mask [L]

    n_decoy = 0
    # Load both test files
    for pt_name in ["test_single_conformers.pt", "test_ensemble_conformers.pt"]:
        pt_path = os.path.join(PREPROCESSED_DIR, pt_name)
        if not os.path.exists(pt_path):
            log.warning(f"GT file not found: {pt_path}")
            continue

        records = torch.load(pt_path, map_location="cpu")

        # Group records by cluster, dropping synthetic decoys and records
        # without enough resolved residues for Kabsch alignment. Mirror the
        # filter in rna_conformer_dataset so validation and evaluation see the
        # same set of clusters.
        cluster_records: dict[str, list] = {}
        for rec in records:
            cname = str(rec["cluster_name"])
            if cname not in cluster_split:
                continue
            if _DECOY_CONFORMER_RE.search(str(rec.get("conformer_name", ""))):
                n_decoy += 1
                continue
            if rec.get("res_mask") is None or int(rec["res_mask"].sum()) < 16:
                continue
            cluster_records.setdefault(cname, []).append(rec)

        for cname, recs in cluster_records.items():
            # Anchor pick: match conformer_name to the PDB-id-chain suffix of
            # the cluster (same rule as RNAClusterDataset). Fall back to the
            # first record if that lookup fails.
            suffix = cname.split("_", 1)[1] if "_" in cname else cname
            anchor = next(
                (r for r in recs if str(r["conformer_name"]) == suffix),
                recs[0],
            )
            gt_single[cname] = anchor["c4_coords"]  # [L, 3]
            gt_mask[cname] = anchor["res_mask"]      # [L]

            # Stack all real conformers for ensemble metrics
            gt_ensemble[cname] = torch.stack(
                [r["c4_coords"] for r in recs], dim=0
            )  # [K, L, 3]

    # Drop clusters that had no real-experimental records surviving the
    # filters (e.g. anchor with fewer than 16 resolved residues).
    cluster_split = {c: s for c, s in cluster_split.items() if c in gt_single}

    log.info(
        f"Loaded GT for {len(gt_single)} clusters "
        f"({sum(1 for v in cluster_split.values() if v == 'test_single')} single, "
        f"{sum(1 for v in cluster_split.values() if v == 'test_ensemble')} ensemble); "
        f"filtered {n_decoy} synthetic '_cNNN_model' records"
    )
    return gt_single, gt_ensemble, gt_mask, cluster_split


# ── evaluation ───────────────────────────────────────────────────────────
def evaluate_method(
    method_dir: str,
    gt_single: dict,
    gt_ensemble: dict,
    gt_mask: dict,
    cluster_split: dict[str, str],
) -> list[dict]:
    """Evaluate a single method against GT data.

    Both metric families are computed for every cluster:
        - anchor RMSD / TM-score  (best-of-G vs the real anchor structure)
        - ensemble AMR / coverage / pairwise  (vs all real conformers in the
          cluster; for test_single clusters K may be 1, which still yields a
          well-defined amr_recall = best-of-G anchor RMSD).
    """
    method_name = os.path.basename(method_dir)
    rows = []

    cluster_dirs = sorted(
        d for d in glob.glob(os.path.join(method_dir, "cluster_*"))
        if os.path.isdir(d)
    )
    if not cluster_dirs:
        log.warning(f"No cluster_* directories found in {method_dir}")
        return rows

    for cluster_dir in cluster_dirs:
        cluster_name = os.path.basename(cluster_dir)

        if cluster_name not in gt_single:
            log.debug(f"Skipping {cluster_name}: no GT data")
            continue

        pred_c4_np = load_pred_c4_for_cluster(cluster_dir)
        if pred_c4_np is None:
            log.warning(f"No valid predictions for {cluster_name} in {method_name}")
            continue

        gt_c4 = gt_single[cluster_name]     # [L, 3]
        mask = gt_mask[cluster_name]         # [L]

        # Check length compatibility
        pred_len = pred_c4_np.shape[1]
        gt_len = gt_c4.shape[0]
        if pred_len != gt_len:
            log.warning(
                f"{method_name}/{cluster_name}: pred length {pred_len} != GT length {gt_len}, skipping"
            )
            continue

        pred_c4 = torch.from_numpy(pred_c4_np).float()    # [G, L, 3]
        split = cluster_split.get(cluster_name, "")

        row = {
            "method": method_name,
            "cluster": cluster_name,
            "split": split,
            "length": gt_len,
            "num_generated": pred_c4.shape[0],
        }

        # Split-specific metric families:
        #   test_single  -> best-of-G RMSD + TM-score vs the real anchor
        #   test_ensemble -> AMR / coverage / pairwise vs the real ensemble
        if split == "test_single":
            gt_c4_t = gt_c4.unsqueeze(0).float()   # [1, L, 3]
            mask_t = mask.unsqueeze(0).float()      # [1, L]
            try:
                single_result = compute_single_metrics(pred_c4, gt_c4_t, mask_t)
                row["rmsd"] = float(single_result["rmsd"])
                row["tm_score"] = float(single_result["tm_score"])
            except Exception as e:
                log.warning(f"Single metrics failed for {method_name}/{cluster_name}: {e}")

        elif split == "test_ensemble" and pred_c4.shape[0] >= 2:
            gt_ens = gt_ensemble[cluster_name].float()  # [K, L, 3]
            ens_mask = mask.float()                      # [L]
            try:
                ens_result = compute_ensemble_metrics(pred_c4, gt_ens, ens_mask)
                row.update(ens_result)
            except Exception as e:
                log.warning(f"Ensemble metrics failed for {method_name}/{cluster_name}: {e}")

        rows.append(row)

    return rows


def print_summary(df: pd.DataFrame):
    """Print per-method summary statistics, separated by split."""
    metric_cols = [
        "rmsd", "tm_score",
        "amr_recall", "amr_precision",
        "cov_recall_3.0", "cov_precision_3.0",
        "cov_recall_4.0", "cov_precision_4.0",
        "pairwise_rmsd",
    ]

    for method, mdf in df.groupby("method"):
        log.info(f"\n{'=' * 60}")
        log.info(f"Method: {method}  ({len(mdf)} clusters)")
        log.info(f"{'=' * 60}")

        for split, sdf in mdf.groupby("split"):
            log.info(f"  --- {split} ({len(sdf)} clusters) ---")
            for col in metric_cols:
                if col in sdf.columns:
                    vals = sdf[col].dropna()
                    if len(vals) > 0:
                        log.info(f"  {col:20s}  mean={vals.mean():.4f}  median={vals.median():.4f}")


# ── main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate RNA conformer generation methods")
    parser.add_argument(
        "--methods", nargs="*", default=None,
        help="Method directory names to evaluate (default: all dirs under inference_ours)",
    )
    parser.add_argument(
        "--split", default="all",
        choices=["all", "test_single", "test_ensemble"],
        help="Which test split to evaluate on",
    )
    parser.add_argument(
        "--inference_root", default=INFERENCE_ROOT,
        help="Root directory containing method subdirectories",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: {inference_root}/metrics_{split}.csv)",
    )
    args = parser.parse_args()

    # Determine splits
    if args.split == "all":
        split_names = ["test_single", "test_ensemble"]
    else:
        split_names = [args.split]

    # Discover methods
    if args.methods:
        method_dirs = [os.path.join(args.inference_root, m) for m in args.methods]
    else:
        method_dirs = sorted(
            d for d in glob.glob(os.path.join(args.inference_root, "*"))
            if os.path.isdir(d)
        )

    log.info(f"Methods to evaluate: {[os.path.basename(d) for d in method_dirs]}")
    log.info(f"Splits: {split_names}")

    # Load GT
    gt_single, gt_ensemble, gt_mask, cluster_split = load_gt_data(split_names)

    # Evaluate each method
    all_rows = []
    for method_dir in method_dirs:
        if not os.path.isdir(method_dir):
            log.warning(f"Skipping {method_dir}: not a directory")
            continue
        log.info(f"Evaluating {os.path.basename(method_dir)} ...")
        rows = evaluate_method(method_dir, gt_single, gt_ensemble, gt_mask, cluster_split)
        all_rows.extend(rows)
        log.info(f"  {len(rows)} clusters evaluated")

    if not all_rows:
        log.warning("No results produced.")
        return

    df = pd.DataFrame(all_rows)

    # Save
    output_path = args.output or os.path.join(
        args.inference_root, f"metrics_{args.split}.csv"
    )
    df.to_csv(output_path, index=False)
    log.info(f"Saved {len(df)} rows to {output_path}")

    # Summary
    print_summary(df)


if __name__ == "__main__":
    main()
