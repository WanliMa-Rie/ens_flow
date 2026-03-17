#!/usr/bin/env python
"""
Offline dataset builder that processes raw PDB files into preprocessed conformers.

Returns:
    train_conformers.pt: List[Dict[str, Any]],
    val_ensemble_conformers.pt: List[Dict[str, Any]],
    val_single_conformers.pt: List[Dict[str, Any]],
    test_ensemble_conformers.pt: List[Dict[str, Any]],
    test_single_conformers.pt: List[Dict[str, Any]],
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
from rna_backbone_design.analysis.ensemble_metrics import _pairwise_rmsd_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build preprocessed datasets directly from PDB files.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root directory containing cluster folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/preprocessed_data",
        help="Directory to save the processed .pt files.",
    )
    parser.add_argument(
        "--cluster_split",
        type=str,
        default="./data/cdhit_80_split_indices.json",
        help="Path to split_cdhit80.json. Defaults to <data_dir>/split_cdhit80.json.",
    )
    parser.add_argument("--split_cache_filename", type=str, default="split_cdhit80.json")
    parser.add_argument("--cdhit_identity_threshold", type=float, default=0.8)
    parser.add_argument("--cdhit_word_length", type=int, default=5)
    parser.add_argument("--split_val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-cluster debug info.",
    )
    return parser.parse_args()


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
    struct_chains = list(structure.get_chains())

    struct_feats: List[Dict[str, Any]] = []
    na_natype = None

    for chain in struct_chains:
        chain_id = str(chain.id)
        chain_index = du.chain_str_to_int(chain_id.upper())
        chain_mol = parsers.process_chain_pdb(
            chain, chain_index, chain_id,
        )
        if chain_mol is None:
            continue
        if chain_mol[-1]["molecule_type"] != "na":
            continue

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
        struct_feats.append(chain_dict)

    if not struct_feats:
        return None

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
    """
    From compact 23-atom -> atom27 -> SE(3) frames.
    """
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

    is_na_residue_mask_np = processed_feats.get("is_na_residue_mask", None)
    if is_na_residue_mask_np is None:
        is_na_residue_mask_np = np.ones((num_res,), dtype=np.bool_)
    is_na_residue_mask = torch.as_tensor(is_na_residue_mask_np).bool()

    # atom_positions is in compact 23-atom format from parsing.py.
    # Convert to atom27 (atom_order) format before extracting named atoms.
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

    # Extract named-atom coords from atom27 positions (correct atom_order indexing).
    atom27_pos = na_feats["all_atom_positions"].float()
    atom27_mask = na_feats["all_atom_mask"].float()
    res_mask = (atom27_mask[:, c4_idx] > 0.5).int()
    c4_coords = atom27_pos[:, c4_idx]
    bb_coords = atom27_pos[:, bb_atom_idx]

    return {
        "aatype": aatype,
        "trans_1": trans_1,
        "rotmats_1": rotmats_1,
        "torsion_angles_sin_cos": na_feats["torsion_angles_sin_cos"][:, :8].float(),
        "torsion_angles_mask": na_feats["torsion_angles_mask"][:, :8].float(),
        "res_mask": res_mask,
        "is_na_residue_mask": is_na_residue_mask,
        "c4_coords": c4_coords,
        "bb_coords": bb_coords,
    }





# ---------------------------------------------------------------------------
#  Per-cluster processing  (returns a flat list of conformer records)
# ---------------------------------------------------------------------------

def process_conformers(
    cluster_dir: pathlib.Path,
    split_name: str,
    do_rmsd: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    pdb_files = sorted((cluster_dir / "structure").glob("*.pdb"))
    if not pdb_files:
        return None

    cluster_name = cluster_dir.name
    embedding_dir = pathlib.Path(cluster_name) / "embedding"
    conformers: List[Dict[str, Any]] = []

    for pdb_path in pdb_files:
        complex_feats = pdb_to_complex_feats(str(pdb_path))
        if complex_feats is None:
            continue
        processed = raw_to_processed(complex_feats)
        geom = processed_to_geometry(processed)

        conformers.append(
            {
                "conformer_name": pdb_path.stem,
                "cluster_name": cluster_name,
                "embedding_dir": str(embedding_dir),
                "seq_len": int(geom["aatype"].shape[0]),
                "aatype": geom["aatype"],
                "trans_1": geom["trans_1"],
                "rotmats_1": geom["rotmats_1"],
                "torsion_angles_sin_cos": geom["torsion_angles_sin_cos"],
                "torsion_angles_mask": geom["torsion_angles_mask"],
                "res_mask": geom["res_mask"],
                "is_na_residue_mask": geom["is_na_residue_mask"],
                "c4_coords": geom["c4_coords"],
                "bb_coords": geom["bb_coords"],
            }
        )

    if not conformers:
        return None

    conformer_names = [c["conformer_name"] for c in conformers]
    cluster_size = len(conformers)
    for conformer in conformers:
        conformer["conformer_names"] = conformer_names
        conformer["cluster_size"] = cluster_size

    if split_name == "train" and do_rmsd:
        if len(conformers) >= 2:
            coords = torch.stack([conformer["c4_coords"] for conformer in conformers], dim=0)
            masks = torch.stack([conformer["res_mask"] for conformer in conformers], dim=0)
            rmsd_matrix = _pairwise_rmsd_matrix(coords, masks.prod(dim=0).float())
        else:
            rmsd_matrix = torch.zeros(1, 1)
        for conformer in conformers:
            conformer["rmsd_matrix"] = rmsd_matrix

    return conformers



def main():
    args = parse_args()
    data_dir = pathlib.Path(args.data_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_split_path = (
        pathlib.Path(args.cluster_split)
        if args.cluster_split is not None
        else data_dir / args.split_cache_filename
    )

    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Using split file: {cluster_split_path}")

    payload = json.loads(cluster_split_path.read_text())
    splits = {
        "train": list(payload.get("train", [])),
        "val_ensemble": list(payload.get("val_ensemble", [])),
        "val_single": list(payload.get("val_single", [])),
        "test_ensemble": list(payload.get("test_ensemble", [])),
        "test_single": list(payload.get("test_single", [])),
    }

    for split_name in (
        "train",
        "val_ensemble",
        "val_single",
        "test_ensemble",
        "test_single",
    ):
        split_cluster_names = splits[split_name]
        print(f"\n{'=' * 60}")
        print(f"Processing split: {split_name} ({len(split_cluster_names)} clusters)")
        print(f"{'=' * 60}")

        do_rmsd = split_name == "train"
        conformer_list: List[Dict[str, Any]] = []
        skipped = 0
        t0 = time.time()

        for cluster_name in tqdm(split_cluster_names, desc=split_name):
            records = process_conformers(
                cluster_dir=data_dir / cluster_name,
                split_name=split_name,
                do_rmsd=do_rmsd,
            )
            if records is None:
                skipped += 1
            else:
                conformer_list.extend(records)

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
