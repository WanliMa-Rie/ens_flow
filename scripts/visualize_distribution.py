"""
uv run python scripts/visualize_distribution.py \
  --infer_cluster_id cluster_0347_4v7l_AY \
  --infer_ckpt_path /projects/u6bk/wanli/checkpoint/rnafm_ckpt/ensemble_flow_matching/last.ckpt \
  --infer_use_existing \
  --infer_output_dir generated_rna_bb_samples/generated_samples/generated_samples \
  --infer_name generated_samples \
  --infer_override data_cfg.data_dir=data_ensemble/rna_ensemble_data \
  --infer_override inference.num_gpus=1 \
  --infer_override inference.ensemble.num_generated=200 \
  --output viz.png

uv run python scripts/visualize_distribution.py \
--compare_cluster_id cluster_0121_3eph_E \
--output cluster_0121_3eph_E.png
"""


import os
import argparse
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tqdm import tqdm
import warnings

# Suppress PDB warnings
warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=PDBConstructionWarning)

# Import local modules
import sys
if "." not in sys.path:
    sys.path.append(".")

from rna_backbone_design.analysis.ensemble_metrics import _pairwise_rmsd_matrix
from rna_backbone_design.analysis.metrics import superimpose


def _list_pdb_files(dir_path: str) -> list[str]:
    if dir_path is None or not os.path.exists(dir_path):
        return []
    return sorted(
        [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(".pdb") and os.path.isfile(os.path.join(dir_path, f))
        ]
    )


def _infer_sampled_dir(input_dir: str) -> str:
    sampled_subdir = os.path.join(input_dir, "sampled")
    if os.path.isdir(sampled_subdir) and _list_pdb_files(sampled_subdir):
        return sampled_subdir
    return input_dir


def _infer_gt_dir(input_dir: str) -> str | None:
    gt_subdir = os.path.join(input_dir, "gt")
    if os.path.isdir(gt_subdir) and _list_pdb_files(gt_subdir):
        return gt_subdir
    return None


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs_or_repo_relative(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(_repo_root(), path)


def _run_val_single_inference(
    cluster_id: str,
    ckpt_path: str | None,
    output_dir: str,
    name: str,
    num_gpus: int | None,
    num_generated: int | None,
    write_ensemble_pdbs: bool | None,
    extra_overrides: list[str] | None,
) -> None:
    repo_root = _repo_root()
    script_path = os.path.join(repo_root, "inference_se3_flows.py")

    cmd = [
        sys.executable,
        script_path,
        "inference.dataset_split=val_single",
        f"inference.cluster_id={cluster_id}",
        f"inference.output_dir={output_dir}",
        f"inference.name={name}",
    ]
    if ckpt_path:
        cmd.append(f"inference.ckpt_path={ckpt_path}")
    if num_gpus is not None:
        cmd.append(f"inference.num_gpus={int(num_gpus)}")
    if num_generated is not None:
        cmd.append(f"inference.ensemble.num_generated={int(num_generated)}")
    if write_ensemble_pdbs is not None:
        cmd.append(f"inference.ensemble.write_ensemble_pdbs={str(bool(write_ensemble_pdbs))}")
    if extra_overrides:
        cmd.extend(extra_overrides)

    subprocess.run(cmd, cwd=repo_root, check=True)


def get_c4_coords(pdb_file):
    """
    Parses a PDB file and returns the coordinates of C4' atoms (or P if C4' missing).
    Returns: numpy array of shape (L, 3)
    """
    parser = PDBParser()
    try:
        structure = parser.get_structure('struct', pdb_file)
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return None

    coords = []
    # Assume single model
    try:
        model = next(iter(structure))
        for chain in model:
            for residue in chain:
                # Skip heteroatoms
                if residue.id[0] != ' ':
                    continue
                
                atom = None
                # Prioritize C4' as it's a key backbone atom for RNA
                if "C4'" in residue:
                    atom = residue["C4'"]
                elif "P" in residue:
                    atom = residue["P"]
                
                if atom:
                    coords.append(atom.get_coord())
    except StopIteration:
        pass
        
    return np.array(coords)

def main():
    parser = argparse.ArgumentParser(description="Visualize RNA structure distribution via MDS on RMSD matrix.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=False,
        help="Directory containing generated PDB files, or an experiment root with sampled/ and gt/ subfolders.",
    )
    parser.add_argument(
        "--compare_dir",
        action="append",
        default=None,
        help="Repeatable. Compare multiple algorithm outputs on the same RNA by providing multiple cluster dirs.",
    )
    parser.add_argument(
        "--compare_label",
        action="append",
        default=None,
        help="Optional labels matching --compare_dir order. If omitted, uses parent folder name.",
    )
    parser.add_argument(
        "--compare_cluster_id",
        type=str,
        default=None,
        help="Cluster full name to compare across model folders (e.g., cluster_0121_3eph_E).",
    )
    parser.add_argument(
        "--compare_root",
        type=str,
        default="/projects/u6bk/wanli/inference",
        help="Root directory containing model subfolders for compare mode.",
    )
    parser.add_argument(
        "--compare_models",
        type=str,
        default="dynarna_ensemble,rnafm_ensemble,rhofold_ensemble,af3_ensemble",
        help="Comma-separated model folder names for compare mode.",
    )
    parser.add_argument("--infer_cluster_id", type=str, default=None, help="If set, runs val_single inference for this cluster and then visualizes.")
    parser.add_argument("--infer_ckpt_path", type=str, default=None, help="Optional override for inference.ckpt_path.")
    parser.add_argument(
        "--infer_output_dir",
        type=str,
        default="generated_rna_bb_samples/rna-frameflow_cond_ensemble2",
        help="Output root for inference (same as inference.output_dir).",
    )
    parser.add_argument("--infer_name", type=str, default="generated_samples", help="Subdir name for inference outputs (same as inference.name).")
    parser.add_argument("--infer_num_gpus", type=int, default=None, help="Optional override for inference.num_gpus.")
    parser.add_argument("--infer_num_generated", type=int, default=None, help="Optional override for inference.ensemble.num_generated.")
    parser.add_argument(
        "--infer_write_ensemble_pdbs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override for inference.ensemble.write_ensemble_pdbs.",
    )
    parser.add_argument(
        "--infer_override",
        action="append",
        default=None,
        help="Additional Hydra override(s) passed to inference_se3_flows.py (repeatable). Example: --infer_override data_cfg.data_dir=... ",
    )
    parser.add_argument(
        "--infer_use_existing",
        action="store_true",
        help="If set, skips running inference and looks for existing files in the inferred output directory.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Directory containing ground truth PDB files (optional). If omitted, tries {input_dir}/gt.",
    )
    parser.add_argument(
        "--ensemble_data_dir",
        type=str,
        default="/projects/u6bk/wanli/rna_ensemble_data",
        help="RNA ensemble data root. If set and gt_dir/gt_pdb not provided, "
             "tries {ensemble_data_dir}/{cluster_full_name}/structure based on input_dir name.",
    )
    parser.add_argument("--gt_pdb", type=str, default=None, help="Path to a single ground truth PDB file (optional).")
    parser.add_argument("--output", type=str, default="distribution_viz.png", help="Output plot filename.")
    parser.add_argument("--title", type=str, default="Ensemble Distribution", help="Plot title.")
    parser.add_argument(
        "--mode",
        type=str,
        default="mds",
        choices=["mds", "pca_gt"],
        help="Visualization mode: 'mds' (default, RMSD-based layout) or 'pca_gt' (GT-centered PCA, requires GT)."
    )
    parser.add_argument(
        "--cluster_dbscan",
        action="store_true",
        help="If set, run DBSCAN on sampled structures and plot MDS with cluster colors.",
    )
    parser.add_argument(
        "--cluster_eps",
        type=float,
        default=3.0,
        help="DBSCAN eps (Angstrom) for clustering sampled structures.",
    )
    parser.add_argument(
        "--cluster_min_samples",
        type=int,
        default=5,
        help="DBSCAN min_samples for clustering sampled structures.",
    )
    args = parser.parse_args()

    if args.compare_cluster_id and args.compare_dir:
        raise ValueError("Use either --compare_cluster_id or --compare_dir, not both.")

    if args.compare_cluster_id:
        models = [m.strip() for m in str(args.compare_models).split(",") if m.strip()]
        compare_root = _abs_or_repo_relative(args.compare_root)
        args.compare_dir = [
            os.path.join(compare_root, model, str(args.compare_cluster_id)) for model in models
        ]
        if args.compare_label is None:
            args.compare_label = models
        if args.output == "distribution_viz.png":
            args.output = f"{args.compare_cluster_id}_compare.png"

    compare_mode = bool(args.compare_dir)
    if compare_mode and args.infer_cluster_id is not None:
        raise ValueError("--compare_dir cannot be used with --infer_cluster_id.")
    if compare_mode and args.input_dir is not None:
        print("Warning: --compare_dir is set; ignoring --input_dir.")

    if args.infer_cluster_id is not None:
        infer_output_dir = _abs_or_repo_relative(args.infer_output_dir)
        infer_name = str(args.infer_name)
        infer_ckpt_path = _abs_or_repo_relative(args.infer_ckpt_path)
        
        if not args.infer_use_existing:
            _run_val_single_inference(
                cluster_id=str(args.infer_cluster_id),
                ckpt_path=infer_ckpt_path,
                output_dir=infer_output_dir,
                name=infer_name,
                num_gpus=args.infer_num_gpus,
                num_generated=args.infer_num_generated,
                write_ensemble_pdbs=args.infer_write_ensemble_pdbs,
                extra_overrides=args.infer_override,
            )
        else:
            print(f"Skipping inference. Using existing samples in: {os.path.join(infer_output_dir, infer_name, str(args.infer_cluster_id))}")
            
        args.input_dir = os.path.join(infer_output_dir, infer_name, str(args.infer_cluster_id))

    if not compare_mode:
        if args.input_dir is None:
            raise ValueError("Either --input_dir must be provided, or --infer_cluster_id must be set to run inference first.")
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist.")
            return

    all_coords = []
    model_ranges: list[tuple[str, int, int]] = []

    # Load GT (either dir or a single file)
    gt_coords_entries = []
    gt_pdb_files = []
    if args.gt_dir is not None:
        gt_pdb_files = _list_pdb_files(args.gt_dir)
        if not gt_pdb_files:
            print(f"Warning: No PDB files found in gt_dir={args.gt_dir}.")
    elif args.gt_pdb:
        if os.path.exists(args.gt_pdb) and os.path.isfile(args.gt_pdb):
            gt_pdb_files = [args.gt_pdb]
        else:
            print(f"Warning: GT PDB {args.gt_pdb} not found.")
    else:
        if compare_mode:
            cluster_names = {os.path.basename(os.path.normpath(d)) for d in args.compare_dir}
            if len(cluster_names) == 1:
                cluster_full_name = next(iter(cluster_names))
                if args.ensemble_data_dir:
                    ensemble_gt_dir = os.path.join(
                        _abs_or_repo_relative(args.ensemble_data_dir),
                        cluster_full_name,
                        "structure",
                    )
                    if os.path.isdir(ensemble_gt_dir):
                        gt_pdb_files = _list_pdb_files(ensemble_gt_dir)
                        if gt_pdb_files:
                            print(f"Found {len(gt_pdb_files)} GT structures in {ensemble_gt_dir}.")
            else:
                print(
                    "Warning: compare_dir paths point to different cluster names; "
                    "auto GT lookup is disabled. Use --gt_dir or --gt_pdb."
                )
        else:
            inferred_gt_dir = _infer_gt_dir(args.input_dir)
            if inferred_gt_dir is not None:
                gt_pdb_files = _list_pdb_files(inferred_gt_dir)
                if gt_pdb_files:
                    print(f"Found {len(gt_pdb_files)} GT structures in {inferred_gt_dir}.")
            if not gt_pdb_files and args.ensemble_data_dir:
                cluster_full_name = os.path.basename(os.path.normpath(args.input_dir))
                if cluster_full_name.startswith("cluster_"):
                    ensemble_gt_dir = os.path.join(
                        _abs_or_repo_relative(args.ensemble_data_dir),
                        cluster_full_name,
                        "structure",
                    )
                    if os.path.isdir(ensemble_gt_dir):
                        gt_pdb_files = _list_pdb_files(ensemble_gt_dir)
                        if gt_pdb_files:
                            print(f"Found {len(gt_pdb_files)} GT structures in {ensemble_gt_dir}.")

    if compare_mode:
        if args.cluster_dbscan:
            print("Warning: --cluster_dbscan is ignored in compare mode.")
            args.cluster_dbscan = False

        compare_dirs = [_abs_or_repo_relative(d) for d in args.compare_dir]
        compare_labels = args.compare_label or []
        if compare_labels and len(compare_labels) != len(compare_dirs):
            raise ValueError("--compare_label count must match --compare_dir count.")

        entries = []
        for idx, dir_path in enumerate(compare_dirs):
            if not os.path.exists(dir_path):
                print(f"Warning: compare_dir does not exist: {dir_path}")
                continue
            sampled_dir = _infer_sampled_dir(dir_path)
            sampled_pdb_files = _list_pdb_files(sampled_dir)
            if not sampled_pdb_files:
                print(f"Warning: No PDB files found in {sampled_dir}")
                continue
            label = (
                compare_labels[idx]
                if compare_labels
                else os.path.basename(os.path.dirname(os.path.normpath(dir_path)))
            )
            print(f"Found {len(sampled_pdb_files)} sampled structures in {sampled_dir} ({label}).")
            entries.append((label, sampled_pdb_files))

        if not entries:
            print("No valid compare_dir entries found.")
            return

        print("Loading sampled structures...")
        sampled_coords_by_model: list[list[np.ndarray]] = []
        for label, pdb_files in entries:
            coords = []
            for f in tqdm(pdb_files, desc=label):
                c = get_c4_coords(f)
                if c is not None and len(c) > 0:
                    coords.append(c)
            sampled_coords_by_model.append(coords)

        all_sampled_coords = [c for coords in sampled_coords_by_model for c in coords]
        if len(all_sampled_coords) < 2:
            print("Need at least 2 sampled structures to plot distribution.")
            return

        lengths = [len(c) for c in all_sampled_coords]
        from collections import Counter
        common_len = Counter(lengths).most_common(1)[0][0]

        filtered_sampled_coords = []
        for label, coords in zip([e[0] for e in entries], sampled_coords_by_model):
            start_idx = len(filtered_sampled_coords)
            kept = [c for c in coords if len(c) == common_len]
            filtered_sampled_coords.extend(kept)
            end_idx = len(filtered_sampled_coords)
            if end_idx == start_idx:
                print(f"Warning: No samples with common length for {label}.")
            model_ranges.append((label, start_idx, end_idx))

        if len(filtered_sampled_coords) < 2:
            print(f"Need at least 2 sampled structures with common length ({common_len}).")
            return

        all_coords.extend(filtered_sampled_coords)
    else:
        sampled_dir = _infer_sampled_dir(args.input_dir)
        sampled_pdb_files = _list_pdb_files(sampled_dir)
        if not sampled_pdb_files:
            print(f"No PDB files found in {sampled_dir}")
            return

        print(f"Found {len(sampled_pdb_files)} sampled structures in {sampled_dir}.")

        print("Loading sampled structures...")
        sampled_coords = []
        for f in tqdm(sampled_pdb_files):
            c = get_c4_coords(f)
            if c is not None and len(c) > 0:
                sampled_coords.append(c)

        if len(sampled_coords) < 2:
            print("Need at least 2 structures to plot distribution.")
            return

        lengths = [len(c) for c in sampled_coords]
        from collections import Counter
        common_len = Counter(lengths).most_common(1)[0][0]

        filtered_sampled_coords = [c for c in sampled_coords if len(c) == common_len]
        if len(filtered_sampled_coords) < 2:
            print(f"Need at least 2 sampled structures with common length ({common_len}).")
            return

        all_coords.extend(filtered_sampled_coords)
            
    # Add GT entries aligned to sampled common length
    if gt_pdb_files:
        print("Loading GT structures...")
        for f in tqdm(gt_pdb_files):
            c = get_c4_coords(f)
            if c is not None and len(c) > 0:
                gt_coords_entries.append(c)

        aligned_gt_coords = []
        skipped_gt = 0
        truncated_gt = 0
        for c in gt_coords_entries:
            if len(c) == common_len:
                aligned_gt_coords.append(c)
            elif len(c) > common_len:
                aligned_gt_coords.append(c[:common_len])
                truncated_gt += 1
            else:
                skipped_gt += 1

        if truncated_gt > 0:
            print(f"Truncated {truncated_gt} GT structures to length {common_len}.")
        if skipped_gt > 0:
            print(f"Skipped {skipped_gt} GT structures shorter than {common_len}.")

        gt_coords_entries = aligned_gt_coords
        if len(gt_coords_entries) == 0:
            print(f"Warning: No GT structures could be aligned to common length ({common_len}).")
        if gt_coords_entries:
            all_coords.extend(gt_coords_entries)
    
    print(f"Using {len(all_coords)} structures with length {common_len}.")

    # Convert to tensor for RMSD calculation
    coords_tensor = torch.tensor(np.array(all_coords), dtype=torch.float32) # (N, L, 3)
    L = coords_tensor.shape[1]
    mask = torch.ones(L, dtype=torch.float32)
    
    if torch.cuda.is_available():
        coords_tensor = coords_tensor.cuda()
        mask = mask.cuda()

    coords_2d = None
    cluster_labels = None
    cluster_metrics = None

    if args.cluster_dbscan:
        print("Running DBSCAN clustering on sampled structures...")
        sampled_tensor = torch.tensor(np.array(filtered_sampled_coords), dtype=torch.float32)
        if torch.cuda.is_available():
            sampled_tensor = sampled_tensor.cuda()

        rmsd_matrix = _pairwise_rmsd_matrix(sampled_tensor, mask)
        rmsd_matrix_np = rmsd_matrix.cpu().numpy()

        labels = DBSCAN(
            eps=float(args.cluster_eps),
            min_samples=int(args.cluster_min_samples),
            metric="precomputed",
        ).fit_predict(rmsd_matrix_np)
        cluster_labels = labels

        unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]
        if not unique_labels:
            cluster_metrics = {"K": 0, "intra_avg": 0.0, "inter_min": 0.0}
        else:
            clusters = [np.where(labels == lbl)[0] for lbl in unique_labels]
            sizes = np.array([len(c) for c in clusters], dtype=np.float64)
            intra_vals = []
            for idxs in clusters:
                if len(idxs) < 2:
                    intra_vals.append(0.0)
                    continue
                sub = rmsd_matrix_np[np.ix_(idxs, idxs)]
                tri = sub[np.triu_indices_from(sub, k=1)]
                intra_vals.append(float(tri.mean()) if tri.size > 0 else 0.0)
            intra_vals_arr = np.array(intra_vals, dtype=np.float64)
            intra_avg = float(np.sum(sizes * intra_vals_arr) / np.sum(sizes)) if sizes.sum() > 0 else 0.0

            if len(clusters) < 2:
                inter_min = 0.0
            else:
                inter_min = np.inf
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        sub = rmsd_matrix_np[np.ix_(clusters[i], clusters[j])]
                        inter_min = min(inter_min, float(sub.min()))
                if not np.isfinite(inter_min):
                    inter_min = 0.0

            cluster_metrics = {"K": len(clusters), "intra_avg": intra_avg, "inter_min": inter_min}

        print(
            f"DBSCAN eps={args.cluster_eps:.2f} min_samples={args.cluster_min_samples} -> "
            f"K={cluster_metrics['K']}, Intra_avg={cluster_metrics['intra_avg']:.2f}A, "
            f"Inter_min={cluster_metrics['inter_min']:.2f}A"
        )

        # MDS on sampled structures using RMSD distance matrix
        print("Running MDS on sampled structures...")
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_jobs=-1, max_iter=300)
        sampled_2d = mds.fit_transform(rmsd_matrix_np)

        if len(gt_coords_entries) > 0:
            gt_tensor = torch.tensor(np.array(gt_coords_entries), dtype=torch.float32)
            if torch.cuda.is_available():
                gt_tensor = gt_tensor.cuda()
            all_tensor = torch.cat([sampled_tensor, gt_tensor], dim=0)
            rmsd_all = _pairwise_rmsd_matrix(all_tensor, mask).cpu().numpy()
            coords_2d = mds.fit_transform(rmsd_all)
        else:
            coords_2d = sampled_2d

    elif args.mode == "pca_gt":
        print("Running PCA relative to GT...")
        if len(gt_coords_entries) == 0:
            raise ValueError("Mode 'pca_gt' requires at least one GT structure to serve as the anchor.")
        
        # 1. Use the first GT as the anchor
        # In the combined list `all_coords`, GTs are at the end
        # num_sampled = len(filtered_sampled_coords)
        # anchor_idx = num_sampled
        # Actually, let's just pick the first GT from the tensor to avoid index confusion
        
        # Slice out the GTs part
        num_sampled = len(filtered_sampled_coords)
        # GTs are from num_sampled to end
        anchor = coords_tensor[num_sampled] # First GT
        
        # 2. Superimpose ALL structures to the Anchor
        # Expand anchor to match batch size
        B = coords_tensor.shape[0]
        anchor_batch = anchor.unsqueeze(0).expand(B, -1, -1) # (N, L, 3)
        mask_batch = mask.unsqueeze(0).expand(B, -1) # (N, L)
        
        print(f"Aligning {B} structures to the first GT...")
        aligned_tensor = superimpose(anchor_batch, coords_tensor, mask_batch)
        
        # 3. Compute difference vectors (N, L, 3)
        diff_tensor = aligned_tensor - anchor_batch
        
        # 4. Flatten to (N, L*3) for PCA
        diff_flat = diff_tensor.reshape(B, -1).cpu().numpy()
        
        # 5. PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(diff_flat)
        
        # Ensure GT is exactly at (0,0) (it should be, but numerical errors exist)
        # The first GT (anchor) corresponds to index `num_sampled`
        # Let's shift everything so that the anchor is at (0,0)
        anchor_2d = coords_2d[num_sampled]
        coords_2d = coords_2d - anchor_2d
        
        print("PCA done. Anchor GT is at (0,0).")
        
    else:
        # Default MDS mode
        print("Calculating pairwise RMSD matrix...")
    
        rmsd_matrix = _pairwise_rmsd_matrix(coords_tensor, mask)
        rmsd_matrix_np = rmsd_matrix.cpu().numpy()

        print("Running MDS...")
        
        # Calculate and print diversity metrics to explain plot scale
        num_sampled = len(filtered_sampled_coords)
        
        # 1. Sampled Diversity (Intra-group RMSD)
        # Exclude diagonal (0) by using upper triangle
        sampled_rmsd_block = rmsd_matrix_np[:num_sampled, :num_sampled]
        if num_sampled > 1:
            # Get upper triangle indices
            triu_indices = np.triu_indices(num_sampled, k=1)
            sampled_pairwise_dists = sampled_rmsd_block[triu_indices]
            avg_sampled_div = np.mean(sampled_pairwise_dists)
            print(f"  > Sampled Diversity (Avg Pairwise RMSD): {avg_sampled_div:.2f} Å")
        
        # 2. GT Diversity
        if len(all_coords) > num_sampled:
            gt_rmsd_block = rmsd_matrix_np[num_sampled:, num_sampled:]
            num_gt = gt_rmsd_block.shape[0]
            if num_gt > 1:
                triu_indices_gt = np.triu_indices(num_gt, k=1)
                gt_pairwise_dists = gt_rmsd_block[triu_indices_gt]
                avg_gt_div = np.mean(gt_pairwise_dists)
                print(f"  > GT Diversity (Avg Pairwise RMSD):      {avg_gt_div:.2f} Å")
            else:
                print(f"  > GT Diversity: N/A (Only 1 GT structure)")
                
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_jobs=-1, max_iter=300)
        coords_2d = mds.fit_transform(rmsd_matrix_np)

        if compare_mode and gt_coords_entries:
            print("Per-model RMSD to GT:")
            gt_start = num_sampled
            gt_end = num_sampled + len(gt_coords_entries)
            gt_indices = np.arange(gt_start, gt_end)
            for label, start_idx, end_idx in model_ranges:
                if end_idx <= start_idx:
                    continue
                block = rmsd_matrix_np[start_idx:end_idx, gt_indices]
                if block.size == 0:
                    print(f"  {label}: N/A (no GT or samples)")
                    continue
                min_per_sample = block.min(axis=1)
                mean_min = float(np.mean(min_per_sample))
                best_min = float(np.min(min_per_sample))
                print(f"  {label}: min={best_min:.2f} Å, mean={mean_min:.2f} Å")

    # Separate sampled and GT 2D coords
    num_sampled = len(filtered_sampled_coords)
    gen_coords_2d = coords_2d[:num_sampled]
    gt_coords_2d = coords_2d[num_sampled:] if len(all_coords) > num_sampled else None

    # --- Plotting ---
    print("Plotting...")
    
    # Nature-style aesthetics
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0
    
    # Custom Colormap (Light to Dark for Density)
    # White/Light Blue -> Dark Blue
    colors = ["#FFFFFF", "#3040A0"] 
    cmap_name = "custom_rna_density"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    # Layout: 
    # Main Scatter (Center)
    # Top Marginal (Top)
    # Right Marginal (Right)
    
    fig = plt.figure(figsize=(8, 8))
    
    # Define grid
    # Width ratios: Main(4), RightMarginal(1)
    # Height ratios: TopMarginal(1), Main(4)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], 
                           wspace=0.05, hspace=0.05)

    # Axes
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    # KDE / scatter data
    x = gen_coords_2d[:, 0]
    y = gen_coords_2d[:, 1]
    
    # Calculate robust limits
    q1_x, q3_x = np.percentile(x, [5, 95])
    iqr_x = q3_x - q1_x
    view_min_x = x.min() - 0.1 * (x.max() - x.min())
    view_max_x = x.max() + 0.1 * (x.max() - x.min())
    
    # Focus closer if outliers exist
    if iqr_x > 0:
        view_min_x = max(view_min_x, q1_x - 1.5 * iqr_x)
        view_max_x = min(view_max_x, q3_x + 1.5 * iqr_x)
    
    q1_y, q3_y = np.percentile(y, [5, 95])
    iqr_y = q3_y - q1_y
    view_min_y = y.min() - 0.1 * (y.max() - y.min())
    view_max_y = y.max() + 0.1 * (y.max() - y.min())

    if iqr_y > 0:
        view_min_y = max(view_min_y, q1_y - 1.5 * iqr_y)
        view_max_y = min(view_max_y, q3_y + 1.5 * iqr_y)

    # Ensure GT points are within view if present
    if gt_coords_2d is not None and len(gt_coords_2d) > 0:
        gt_x = gt_coords_2d[:, 0]
        gt_y = gt_coords_2d[:, 1]
        view_min_x = min(view_min_x, gt_x.min())
        view_max_x = max(view_max_x, gt_x.max())
        view_min_y = min(view_min_y, gt_y.min())
        view_max_y = max(view_max_y, gt_y.max())

        pad_x = 0.05 * (view_max_x - view_min_x) if view_max_x > view_min_x else 1.0
        pad_y = 0.05 * (view_max_y - view_min_y) if view_max_y > view_min_y else 1.0
        view_min_x -= pad_x
        view_max_x += pad_x
        view_min_y -= pad_y
        view_max_y += pad_y

    if compare_mode:
        # Plot per-model points with distinct colors
        num_models = max(len(model_ranges), 1)
        hues = np.linspace(0.0, 1.0, num_models, endpoint=False)
        model_colors = [plt.cm.hsv(h) for h in hues]
        for i, (label, start_idx, end_idx) in enumerate(model_ranges):
            if end_idx <= start_idx:
                continue
            ax_main.scatter(
                x[start_idx:end_idx],
                y[start_idx:end_idx],
                c=[model_colors[i % len(model_colors)]],
                s=20,
                alpha=0.6,
                edgecolors="none",
                zorder=2,
                label=label,
            )

        ax_top.axis("off")
        ax_right.axis("off")

    elif not args.cluster_dbscan:
        # Grid for KDE
        xi, yi = np.mgrid[view_min_x:view_max_x:100j, view_min_y:view_max_y:100j]
        positions = np.vstack([xi.flatten(), yi.flatten()])
        values = np.vstack([x, y])

        try:
            k = gaussian_kde(values)
            zi = k(positions)
            zi = zi.reshape(xi.shape)

            # Smoother distribution requested
            k.set_bandwidth(bw_method=k.factor)
            zi = k(positions)
            zi = zi.reshape(xi.shape)

            ax_main.contourf(xi, yi, zi, levels=20, cmap=cm, alpha=0.9, extend='both')
            ax_main.contour(xi, yi, zi, levels=20, colors='white', linewidths=0.5, alpha=0.3)

        except Exception as e:
            print(f"KDE failed: {e}")
            ax_main.scatter(x, y, c="#3040A0", alpha=0.5)

        # Plot sampled points (gray)
        ax_main.scatter(
            x,
            y,
            c="#9E9E9E",
            alpha=0.35,
            s=20,
            edgecolors="none",
            zorder=2,
            label="Sampled",
        )
    else:
        # Plot DBSCAN clusters with distinct, high-contrast colors
        labels = cluster_labels if cluster_labels is not None else np.full(len(x), -1)
        unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]
        num_clusters = max(len(unique_labels), 1)
        hues = np.linspace(0.0, 1.0, num_clusters, endpoint=False)
        cluster_colors = [plt.cm.hsv(h) for h in hues]
        for i, lbl in enumerate(unique_labels):
            idxs = labels == lbl
            ax_main.scatter(
                x[idxs],
                y[idxs],
                s=24,
                c=[cluster_colors[i % len(cluster_colors)]],
                alpha=0.7,
                edgecolors="none",
                zorder=2,
                label=f"Cluster {lbl}",
            )
        noise = labels == -1
        if np.any(noise):
            ax_main.scatter(
                x[noise],
                y[noise],
                s=20,
                c="#B0B0B0",
                alpha=0.35,
                edgecolors="none",
                zorder=1,
                label="Noise",
            )

    # Plot GT
    if gt_coords_2d is not None and len(gt_coords_2d) > 0:
        gt_label = "GT" if len(gt_coords_2d) == 1 else f"GT (n={len(gt_coords_2d)})"
        ax_main.scatter(
            gt_coords_2d[:, 0],
            gt_coords_2d[:, 1],
            c="red",
            marker="*",
            s=200,
            edgecolors="white",
            linewidth=1.5,
            zorder=10,
            label=gt_label,
        )
    if (gt_coords_2d is not None and len(gt_coords_2d) > 0) or args.cluster_dbscan or compare_mode:
        ax_main.legend(loc="upper right", frameon=False, fontsize=9)

    # Clean Main Axes
    ax_main.set_xlim(view_min_x, view_max_x)
    ax_main.set_ylim(view_min_y, view_max_y)
    # ax_main.set_xticks([])
    # ax_main.set_yticks([])
    # ax_main.set_xlabel("MDS Dimension 1")
    # ax_main.set_ylabel("MDS Dimension 2")
    
    # Remove top and right spines for Nature style
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.tick_params(direction='out')

    # 2. Top Marginal
    # Use the primary color from the palette (#3040A0)
    marginal_color = "#3040A0"
    
    # Plot histogram (gray bars) AND density line (colored)
    ax_top.hist(x, bins=30, density=True, color='gray', alpha=0.3, edgecolor='white')
    
    try:
        x_grid = np.linspace(view_min_x, view_max_x, 200)
        kde_x = gaussian_kde(x)
        kde_x.set_bandwidth(bw_method=kde_x.factor * 1.5)
        y_vals = kde_x(x_grid)
        ax_top.plot(x_grid, y_vals, color=marginal_color, linewidth=1.5)
        # ax_top.fill_between(x_grid, y_vals, color=marginal_color, alpha=0.1) # Keep white background mostly
    except:
        pass
        
    ax_top.axis('off')

    # 3. Right Marginal
    # Plot histogram (gray bars) AND density line (colored)
    ax_right.hist(y, bins=30, density=True, orientation='horizontal', color='gray', alpha=0.3, edgecolor='white')
    
    try:
        y_grid = np.linspace(view_min_y, view_max_y, 200)
        kde_y = gaussian_kde(y)
        kde_y.set_bandwidth(bw_method=kde_y.factor * 1.5)
        x_vals = kde_y(y_grid)
        ax_right.plot(x_vals, y_grid, color=marginal_color, linewidth=1.5)
        # ax_right.fill_betweenx(y_grid, x_vals, x2=0, color=marginal_color, alpha=0.1)
    except:
        pass
        
    ax_right.axis('off')

    # Add Title
    # plt.suptitle(args.title, fontsize=14, y=0.95)
    
    # Save
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {args.output}")

if __name__ == "__main__":
    main()
