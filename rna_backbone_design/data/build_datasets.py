#!/usr/bin/env python
"""
Offline preprocessing: build per-split .pt files **directly from PDB files**.

**Conformer-centric output format**: each .pt file is a flat list of conformer
dicts.  Every conformer carries its own geometry tensors AND cluster-level
metadata (cluster name, sibling conformer names, optional RMSD matrix) so
that an ensemble can be reassembled at runtime by grouping on ``cluster_name``.

For each cluster directory under ``data_dir``:
  1. Parse every ``structure/*.pdb`` with BioPython → chain features
  2. Compute SE(3) geometry (frames, torsions, backbone coords)
  3. Determine unified seq_len across conformers + embeddings
  4. Optionally compute pairwise RMSD matrix (for train clusters)
  5. Emit one record per conformer into the per-split list

Output files (under ``output_dir``):
  - train_conformers.pt
  - val_ensemble_conformers.pt
  - val_single_conformers.pt
  - test_ensemble_conformers.pt
  - test_single_conformers.pt

Each .pt is ``List[Dict]`` where every dict is a single conformer with keys:
  - conformer_name, cluster_name, seq_len, embedding_dir
  - aatype, trans_1, rotmats_1, torsion_angles_sin_cos, torsion_angles_mask,
    res_mask, is_na_residue_mask, c4_coords, bb_coords
  - cluster_conformer_names (List[str]), cluster_size (int)
  - rmsd_matrix (optional Tensor), rmsd_names (optional List[str])

Example (single element in the saved ``List[Dict[str, Any]]``):
.. code-block:: python

    {
        "conformer_name": "7pla_B_structure",
        "cluster_name": "cluster_1946_7pla_B",
        "seq_len": 217,
        "embedding_dir": "cluster_1946_7pla_B/embedding",
        "aatype": Tensor[217],                   # int64
        "trans_1": Tensor[217, 3],               # float32
        "rotmats_1": Tensor[217, 3, 3],          # float32
        "torsion_angles_sin_cos": Tensor[217, 8, 2],
        "torsion_angles_mask": Tensor[217, 8],
        "res_mask": Tensor[217],                 # 1 for modeled residues
        "is_na_residue_mask": Tensor[217],       # bool
        "c4_coords": Tensor[217, 3],
        "bb_coords": Tensor[217, 3, 3],
        "cluster_conformer_names": ["7pla_B_structure", "..."],
        "cluster_size": 12,
        # train split only if --compute_rmsd:
        "rmsd_matrix": Tensor[12, 12],
        "rmsd_names": ["7pla_B_structure", "..."],
    }

Embeddings are NOT included (loaded at runtime from npy files).

  uv run -m rna_backbone_design.data.build_datasets \
    --data_dir /projects/u6bk/wanli/rna_ensemble_data \
    --output_dir /projects/u6bk/wanli/rna_ensemble_data/preprocessed_data \
    --cluster_split /projects/u6bk/wanli/rna_ensemble_data/split_cdhit80.json \
    --compute_rmsd
"""

import argparse
import collections
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from Bio import PDB
from tqdm import tqdm

from rna_backbone_design.data import data_transforms
from rna_backbone_design.data import nucleotide_constants as nc
from rna_backbone_design.data import parsers
from rna_backbone_design.data import utils as du
from rna_backbone_design.data.rigid_utils import Rigid
from rna_backbone_design.data.cdhit_split import (
    CDHitSplitConfig,
    collect_valid_cluster_dirs,
    load_or_build_split_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build preprocessed datasets directly from PDB files.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/projects/u6bk/wanli/rna_ensemble_data",
        help="Root directory containing cluster folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/projects/u6bk/wanli/rna_ensemble_data/preprocessed_data",
        help="Directory to save the processed .pt files.",
    )
    parser.add_argument(
        "--cluster_manifest",
        type=str,
        default=None,
        help=(
            "Path to cluster_manifest.json (unified manifest). "
            "If set, uses its train/val_*/test_* splits."
        ),
    )
    parser.add_argument(
        "--cluster_split",
        type=str,
        default=None,
        help="Path to split_cdhit80.json. Defaults to <data_dir>/split_cdhit80.json.",
    )
    parser.add_argument("--split_cache_filename", type=str, default="split_cdhit80.json")
    parser.add_argument("--cdhit_identity_threshold", type=float, default=0.8)
    parser.add_argument("--cdhit_word_length", type=int, default=5)
    parser.add_argument("--split_val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--cdhit_bin", type=str, default="cd-hit-est")
    parser.add_argument("--cdhit_threads", type=int, default=0)
    parser.add_argument(
        "--compute_rmsd",
        action="store_true",
        help="Compute pairwise RMSD matrices for clusters with >=2 conformers.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-cluster debug info.",
    )
    return parser.parse_args()


def _split_names_from_entry(entry: Any) -> List[str]:
    if entry is None:
        return []
    if isinstance(entry, list):
        return list(entry)
    if isinstance(entry, tuple):
        return list(entry)
    if isinstance(entry, dict):
        cluster_names = entry.get("cluster_names", None)
        if cluster_names is not None:
            return list(cluster_names)
    raise ValueError(f"Unsupported split entry format: {type(entry)!r}")


# ---------------------------------------------------------------------------
#  PDB → feature dict  (replaces the pkl intermediate)
# ---------------------------------------------------------------------------



def pdb_to_complex_feats(
    pdb_path: str,
) -> Optional[Dict[str, Any]]:
    """Parse a PDB file into a feature dict identical to what preprocess_ensemble.py
    would write to a pkl, but without touching disk.

    Returns None if the file cannot be parsed.
    """
    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
    bio_parser = PDB.PDBParser(QUIET=True)

    structure = bio_parser.get_structure(pdb_name, pdb_path)


    struct_chains = {chain.id.upper(): chain for chain in structure.get_chains()}

    all_seqs = set()
    struct_feats: List[Dict[str, Any]] = []
    na_natype = None
    chain_dict = None

    for chain_id, chain in struct_chains.items():
        chain_index = du.chain_str_to_int(chain_id)
        chain_mol = parsers.process_chain_pdb(
            chain, chain_index, chain_id,
        )
        if chain_mol is None:
            continue
        if chain_mol[-1]["molecule_type"] == "na":
            na_natype = (
                chain_mol[-2]
                if na_natype is None
                else torch.cat((na_natype, chain_mol[-2]), dim=0)
            )

        chain_mol_constants = chain_mol[-1]["molecule_constants"]
        chain_mol_backbone_atom_name = chain_mol[-1]["molecule_backbone_atom_name"]
        chain_dict = parsers.macromolecule_outputs_to_dict(chain_mol)
        chain_dict = du.parse_chain_feats_pdb(
            chain_feats=chain_dict,
            molecule_constants=chain_mol_constants,
            molecule_backbone_atom_name=chain_mol_backbone_atom_name,
        )
        all_seqs.add(tuple(chain_dict["aatype"]))
        struct_feats.append(chain_dict)


    # Assembly features (entity_id / asym_id / sym_id)
    seq_to_entity_id: Dict[tuple, int] = {}
    grouped_chains: Dict[int, list] = collections.defaultdict(list)
    for cd in struct_feats:
        seq = tuple(cd["aatype"])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(cd)

    cid = 1
    for entity_id, group in grouped_chains.items():
        for sym_id, cd in enumerate(group, start=1):
            seq_length = len(cd["aatype"])
            cd["asym_id"] = cid * np.ones(seq_length)
            cd["sym_id"] = sym_id * np.ones(seq_length)
            cd["entity_id"] = entity_id * np.ones(seq_length)
            cid += 1

    complex_feats = du.concat_np_features(struct_feats, add_batch_dim=False)
    if complex_feats["bb_mask"].sum() < 1.0:
        return None

    complex_aatype = complex_feats["aatype"]
    modeled_idx = np.where((complex_aatype != 20) & (complex_aatype != 26))[0]
    if len(modeled_idx) == 0:
        return None
    complex_feats["modeled_idx"] = modeled_idx
    complex_feats["na_modeled_idx"] = (
        None if na_natype is None else np.where(na_natype.numpy() != 26)[0]
    )

    # Add is_na_residue_mask explicitly (used by _processed_to_geometry)
    mol_enc = complex_feats.get("molecule_type_encoding", None)
    if mol_enc is not None:
        complex_feats["is_na_residue_mask"] = (
            (mol_enc[:, 1] == 1) | (mol_enc[:, 2] == 1)
        )

    return complex_feats


# ---------------------------------------------------------------------------
#  Feature dict → SE(3) geometry for offline conformer records
# ---------------------------------------------------------------------------

def _trim_to_modeled_idx(processed_feats: Dict[str, Any]) -> Dict[str, Any]:
    modeled_idx = processed_feats.get("modeled_idx", None)
    if modeled_idx is None or len(modeled_idx) == 0:
        return processed_feats
    min_idx = int(np.min(modeled_idx))
    max_idx = int(np.max(modeled_idx))
    full_len = int(np.asarray(processed_feats["aatype"]).shape[0])
    return {
        k: (
            v[min_idx : (max_idx + 1)]
            if (isinstance(v, np.ndarray) and v.ndim >= 1 and int(v.shape[0]) == full_len)
            else v
        )
        for k, v in processed_feats.items()
    }


def raw_to_processed(raw_feats: Dict[str, Any]) -> Dict[str, Any]:
    return _trim_to_modeled_idx(du.parse_complex_feats(raw_feats))


def processed_to_geometry(processed_feats: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    aatype = torch.as_tensor(processed_feats["aatype"]).long()
    atom_positions = torch.as_tensor(processed_feats["atom_positions"]).double()
    atom_mask = torch.as_tensor(processed_feats["atom_mask"]).double()
    atom_deoxy = torch.as_tensor(processed_feats["atom_deoxy"]).bool()

    num_res = int(aatype.shape[0])
    c4_idx = int(nc.atom_order["C4'"])
    bb_atom_idx = [
        int(nc.atom_order["C3'"]),
        int(nc.atom_order["C4'"]),
        int(nc.atom_order["O4'"]),
    ]

    bb_mask_np = processed_feats.get("bb_mask", None)
    if bb_mask_np is None:
        bb_mask_np = np.asarray(atom_mask.detach().cpu().numpy()[:, c4_idx] > 0.5)
    res_mask = torch.as_tensor(bb_mask_np).int()

    is_na_residue_mask_np = processed_feats.get("is_na_residue_mask", None)
    if is_na_residue_mask_np is None:
        is_na_residue_mask_np = np.ones((num_res,), dtype=np.bool_)
    is_na_residue_mask = torch.as_tensor(is_na_residue_mask_np).bool()

    num_na_atoms = 23
    na_feats = {
        "aatype": aatype,
        "all_atom_positions": atom_positions[:, :num_na_atoms],
        "all_atom_mask": atom_mask[:, :num_na_atoms],
        "atom_deoxy": atom_deoxy,
    }
    na_feats["atom23_gt_positions"] = na_feats["all_atom_positions"]
    na_feats = data_transforms.make_atom23_masks(na_feats)
    data_transforms.atom23_list_to_atom27_list(
        na_feats, ["all_atom_positions", "all_atom_mask"], inplace=True,
    )
    na_feats = data_transforms.atom27_to_frames(na_feats)
    na_feats = data_transforms.atom27_to_torsion_angles()(na_feats)

    rigids_1 = Rigid.from_tensor_4x4(na_feats["rigidgroups_gt_frames"])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats().float()
    trans_1 = rigids_1.get_trans().float()

    atom_pos_float = torch.as_tensor(processed_feats["atom_positions"]).float()
    if atom_pos_float.ndim == 3 and int(atom_pos_float.shape[1]) > c4_idx:
        c4_coords = atom_pos_float[:, c4_idx]
        bb_coords = atom_pos_float[:, bb_atom_idx]
    elif atom_pos_float.ndim == 2 and int(atom_pos_float.shape[1]) == 3:
        c4_coords = atom_pos_float
        bb_coords = atom_pos_float[:, None, :].repeat(1, 3, 1)
    else:
        c4_coords = trans_1
        bb_coords = torch.stack([trans_1, trans_1, trans_1], dim=1)

    return {
        "aatype": aatype,
        "trans_1": trans_1,
        "rotmats_1": rotmats_1,
        "torsion_angles_sin_cos": na_feats["torsion_angles_sin_cos"][:, :8].float(),
        "torsion_angles_mask": na_feats["torsion_angles_mask"][:, :8].float(),
        "res_mask": res_mask,
        "is_na_residue_mask": is_na_residue_mask,
        "c4_coords": c4_coords.float(),
        "bb_coords": bb_coords.float(),
    }


def crop_geometry(feats: Dict[str, torch.Tensor], crop_len: int) -> Dict[str, torch.Tensor]:
    return {
        k: (v[:crop_len] if isinstance(v, torch.Tensor) and v.ndim >= 1 else v)
        for k, v in feats.items()
    }


# ---------------------------------------------------------------------------
#  Cluster-level validation (PDB-based, no pkl dependency)
# ---------------------------------------------------------------------------

def collect_valid_cluster_dirs_pdb(data_dir: pathlib.Path) -> List[pathlib.Path]:
    """Like ``collect_valid_cluster_dirs`` but validates using structure/*.pdb
    instead of features/*.pkl.  Embedding files are still required.
    """
    potential = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("cluster_")
    )
    valid: List[pathlib.Path] = []
    for cluster_dir in potential:
        structure_dir = cluster_dir / "structure"
        embedding_dir = cluster_dir / "embedding"
        if not structure_dir.is_dir() or not embedding_dir.is_dir():
            continue
        pdb_files = list(structure_dir.glob("*.pdb"))
        single_files = list(embedding_dir.glob("*_single.npy"))
        pair_files = list(embedding_dir.glob("*_pair.npy"))
        if pdb_files and single_files and pair_files:
            valid.append(cluster_dir)
    return valid


# ---------------------------------------------------------------------------
#  Per-cluster processing  (returns a flat list of conformer records)
# ---------------------------------------------------------------------------

def process_one_cluster(
    cluster_dir: pathlib.Path,
    data_dir: pathlib.Path,
    compute_rmsd: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Process a single cluster directory into a list of conformer dicts.

    Each conformer dict contains its own geometry tensors plus shared
    cluster-level metadata so that ensembles can be reassembled at runtime
    by grouping on ``cluster_name``.

    Returns ``None`` if no valid conformers could be parsed.
    """
    structure_dir = cluster_dir / "structure"
    embedding_dir = cluster_dir / "embedding"

    # List PDB files
    pdb_files = sorted(structure_dir.glob("*.pdb"))

    # Process all conformers: PDB → features → geometry
    conformer_geoms: Dict[str, Dict[str, torch.Tensor]] = {}
    conformer_names: List[str] = []
    seq_lens: List[int] = []

    for pdb_path in pdb_files:
        complex_feats = pdb_to_complex_feats(str(pdb_path))
        if complex_feats is None:
            continue
        processed = raw_to_processed(complex_feats)
        geom = processed_to_geometry(processed)
        seq_len = int(geom["trans_1"].shape[0])
        conformer_geoms[pdb_path.stem] = geom
        conformer_names.append(pdb_path.stem)
        seq_lens.append(seq_len)

    if not conformer_names:
        return None

    # Crop all conformers to a unified seq_len across this cluster.
    seq_len = min(seq_lens)
    for name in conformer_names:
        conformer_geoms[name] = crop_geometry(conformer_geoms[name], seq_len)

    # Compute RMSD matrix (optional, typically for train clusters)
    rmsd_matrix = None
    rmsd_names = None
    if compute_rmsd and len(conformer_names) >= 2:
        from rna_backbone_design.analysis.ensemble_metrics import (
            _pairwise_rmsd_matrix as _global_pairwise_rmsd,
        )

        coords = torch.stack([conformer_geoms[n]["c4_coords"] for n in conformer_names])
        masks = torch.stack([conformer_geoms[n]["res_mask"].float() for n in conformer_names])
        mask = masks.prod(dim=0).float()
        with torch.no_grad():
            rmsd_matrix = _global_pairwise_rmsd(coords, mask)
        rmsd_names = list(conformer_names)

    # Build relative embedding path (Dataset reconstructs full path from data_dir)
    embedding_rel = str(cluster_dir.relative_to(data_dir) / "embedding")

    # ---- Build one record per conformer ----
    cluster_name = cluster_dir.name
    cluster_size = len(conformer_names)

    records: List[Dict[str, Any]] = []
    for cname in conformer_names:
        geom = conformer_geoms[cname]
        record: Dict[str, Any] = {
            # ---- identity ----
            "conformer_name": cname,
            "cluster_name": cluster_name,
            "seq_len": seq_len,
            "embedding_dir": embedding_rel,
            # ---- geometry (flat, top-level) ----
            "aatype": geom["aatype"],
            "trans_1": geom["trans_1"],
            "rotmats_1": geom["rotmats_1"],
            "torsion_angles_sin_cos": geom["torsion_angles_sin_cos"],
            "torsion_angles_mask": geom["torsion_angles_mask"],
            "res_mask": geom["res_mask"],
            "is_na_residue_mask": geom["is_na_residue_mask"],
            "c4_coords": geom["c4_coords"],
            "bb_coords": geom["bb_coords"],
            # ---- cluster metadata (for ensemble assembly) ----
            "cluster_conformer_names": list(conformer_names),
            "cluster_size": cluster_size,
        }
        if rmsd_matrix is not None:
            record["rmsd_matrix"] = rmsd_matrix
            record["rmsd_names"] = rmsd_names
        records.append(record)

    return records

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir = pathlib.Path(args.data_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")

    # ---- Determine splits ----
    if args.cluster_manifest:
        payload = json.loads(pathlib.Path(args.cluster_manifest).read_text())
        splits = {
            "train": _split_names_from_entry(payload.get("train", [])),
            "val_ensemble": _split_names_from_entry(
                payload.get("val_ensemble", payload.get("test_ensemble", []))
            ),
            "test_ensemble": _split_names_from_entry(payload.get("test_ensemble", [])),
            "val_single": _split_names_from_entry(
                payload.get("val_single", payload.get("test_single", []))
            ),
            "test_single": _split_names_from_entry(payload.get("test_single", [])),
        }
        print(f"Using cluster manifest: {args.cluster_manifest}")
    elif args.cluster_split:
        payload = json.loads(pathlib.Path(args.cluster_split).read_text())
        splits = {
            "train": _split_names_from_entry(payload.get("train", [])),
            "val_ensemble": _split_names_from_entry(payload.get("val_ensemble", [])),
            "test_ensemble": _split_names_from_entry(payload.get("test_ensemble", [])),
            "val_single": _split_names_from_entry(payload.get("val_single", [])),
            "test_single": _split_names_from_entry(payload.get("test_single", [])),
        }
        print(f"Using split file: {args.cluster_split}")
    else:
        split_cfg = CDHitSplitConfig(
            identity_threshold=args.cdhit_identity_threshold,
            word_length=args.cdhit_word_length,
            val_fraction=args.split_val_fraction,
            seed=args.split_seed,
            cdhit_bin=args.cdhit_bin,
            threads=args.cdhit_threads,
            cache_filename=args.split_cache_filename,
        )
        manifest = load_or_build_split_manifest(data_dir, split_cfg)
        splits = {
            "train": _split_names_from_entry(manifest.get("train", [])),
            "val_ensemble": _split_names_from_entry(manifest.get("val_ensemble", [])),
            "test_ensemble": _split_names_from_entry(manifest.get("test_ensemble", [])),
            "val_single": _split_names_from_entry(manifest.get("val_single", [])),
            "test_single": _split_names_from_entry(manifest.get("test_single", [])),
        }
        print(f"Using CD-HIT split cache: {args.split_cache_filename}")

    # Use PDB-based validation (does not require features/*.pkl)
    valid_dirs = collect_valid_cluster_dirs_pdb(data_dir)
    name_to_dir = {d.name: d for d in valid_dirs}

    print(f"Found {len(valid_dirs)} valid cluster directories (PDB-based)")

    for split_name in (
        "train",
        "val_ensemble",
        "val_single",
        "test_ensemble",
        "test_single",
    ):
        cluster_names = splits[split_name]
        print(f"\n{'=' * 60}")
        print(f"Processing split: {split_name} ({len(cluster_names)} clusters)")
        print(f"{'=' * 60}")

        do_rmsd = args.compute_rmsd and split_name == "train"

        conformer_list: List[Dict[str, Any]] = []
        skipped = 0
        t0 = time.time()

        for cname in tqdm(cluster_names, desc=split_name):
            cdir = name_to_dir.get(cname)
            if cdir is None:
                skipped += 1
                if args.verbose:
                    print(f"  Cluster {cname} not found in valid dirs, skipping.")
                continue

            records = process_one_cluster(
                cluster_dir=cdir,
                data_dir=data_dir,
                compute_rmsd=do_rmsd,
            )
            if records is not None:
                conformer_list.extend(records)
            else:
                skipped += 1

        elapsed = time.time() - t0
        out_path = output_dir / f"{split_name}_conformers.pt"
        print(
            f"Saving {len(conformer_list)} conformers to {out_path} "
            f"(skipped {skipped} clusters) [{elapsed:.1f}s]"
        )
        torch.save(conformer_list, out_path)
        print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")

    print(f"\nAll splits saved to {output_dir}")


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
