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
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from Bio import SeqIO
from Bio.PDB import MMCIFParser
from tqdm import tqdm

from rna_backbone_design.analysis.metrics import pairwise_rmsd_matrix
from rna_backbone_design.data import data_transforms
from rna_backbone_design.data import nucleotide_constants as nc
from rna_backbone_design.data import vocabulary
from rna_backbone_design.data.rigid_utils import Rigid


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
        "--full_sequence_fasta",
        type=str,
        required=True,
        help="Path to FASTA file with full sequences. Record IDs must equal cluster_name.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-cluster debug info.",
    )
    return parser.parse_args()


def load_full_sequences_from_fasta(fasta_path: str) -> Dict[str, str]:
    """Load full sequences from a FASTA file, keyed by record ID (= cluster_name)."""
    seqs: Dict[str, str] = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id in seqs:
            raise ValueError(f"Duplicate FASTA record ID: {record.id}")
        seqs[record.id] = str(record.seq).upper()
    return seqs


# FASTA single-letter (upper) to vocabulary aatype index


def align_structure_to_full_sequence(
    cif_path: str,
    full_sequence: str,
) -> Dict[str, Any]:
    """Parse a mmCIF file and align its residues to a full-length FASTA sequence.

    Uses label_seq_id (via auth_residues=False) as the canonical 1-based index
    into the full sequence. Unsolved positions get zero atom coords and mask.

    Example
    -------
    Suppose the FASTA has 25 nucleotides but only positions 5-20 are resolved in
    the CIF (label_seq_id 5..20, auth_seq_id could be anything):

        full_sequence = "AAAACGUCGAGCUAGCUACGAUACC"  # len=25
        feats = align_structure_to_full_sequence("rna.cif", full_sequence)

        feats["aatype"]          # shape (25,)  int64
                                 #   e.g. [21,21,21,21, 22,23,..., 21,21,21,21,22]
                                 #   positions 0-3 and 20-24 map from FASTA letters
        feats["atom_positions"]  # shape (25, 23, 3)  float64, centered on solved C4'
                                 #   rows 0-3 and 20-24 are all zeros (unsolved)
        feats["atom_mask"]       # shape (25, 23)  float64
                                 #   rows 0-3 and 20-24 are all zeros
        feats["atom_deoxy"]      # shape (25,)  bool, True only for DNA residues
        feats["is_na_residue_mask"]  # shape (25,)  bool, all True

    Returns full-length processed_feats dict.
    """
    seq_len = len(full_sequence)
    pdb_name = pathlib.Path(cif_path).stem
    bio_parser = MMCIFParser(QUIET=True, auth_residues=False)
    structure = bio_parser.get_structure(pdb_name, cif_path)
    model = list(structure.get_models())[0]
    chain = list(model.get_chains())[0]

    # Encode full sequence aatype from FASTA (unknown letters → 0)
    aatype = np.zeros(seq_len, dtype=np.int64)
    for i, letter in enumerate(full_sequence):
        vocab_letter = letter.lower()
        if vocab_letter in vocabulary.restype_order:
            aatype[i] = vocabulary.restype_order[vocab_letter]

    # Initialize full-length arrays (unsolved = zeros)
    num_atoms = nc.compact_atom_type_num  # 23
    atom_positions = np.zeros((seq_len, num_atoms, 3), dtype=np.float64)
    atom_mask = np.zeros((seq_len, num_atoms), dtype=np.float64)
    atom_deoxy = np.zeros(seq_len, dtype=bool)

    # Place each resolved residue at its label_seq_id position
    occupied = set()
    for res in chain:
        resname = res.resname
        label_seq_id = res.id[1]  # 1-based (auth_residues=False → label_seq_id)
        idx = label_seq_id - 1    # 0-based position in full sequence
        occupied.add(idx)

        # Fill atom coordinates
        compact_atom_order = nc.restype_name_to_compact_atom_order[resname]
        pos = np.zeros((num_atoms, 3), dtype=np.float64)
        mask = np.zeros(num_atoms, dtype=np.float64)
        for atom in res:
            if atom.name not in compact_atom_order:
                continue
            atom_idx = compact_atom_order[atom.name]
            pos[atom_idx] = atom.coord
            mask[atom_idx] = 1.0

        atom_positions[idx] = pos
        atom_mask[idx] = mask

        if hasattr(nc, "deoxy_restypes") and resname in nc.deoxy_restypes:
            atom_deoxy[idx] = True

    # Center around backbone (C4') using only solved residues
    c4_idx = nc.atom_order["C4'"]
    bb_mask = atom_mask[:, c4_idx]
    bb_pos = atom_positions[:, c4_idx]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(bb_mask) + 1e-5)
    atom_positions = atom_positions - bb_center[None, None, :]
    atom_positions = atom_positions * atom_mask[..., None]

    return {
        "aatype": aatype,
        "atom_positions": atom_positions,
        "atom_mask": atom_mask,
        "atom_deoxy": atom_deoxy,
        "is_na_residue_mask": np.ones(seq_len, dtype=bool),
    }


# ---------------------------------------------------------------------------
#  Feature dict → SE(3) geometry for offline conformer records
# ---------------------------------------------------------------------------


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

    torsion_angles_sin_cos = na_feats["torsion_angles_sin_cos"][:, :8].float()
    torsion_angles_mask = na_feats["torsion_angles_mask"][:, :8].float()

    # Normalize unsolved positions: zero geometry, identity rotation
    unsolved = (res_mask == 0)
    trans_1[unsolved] = 0.0
    rotmats_1[unsolved] = torch.eye(3, dtype=rotmats_1.dtype)
    c4_coords[unsolved] = 0.0
    bb_coords[unsolved] = 0.0
    torsion_angles_sin_cos[unsolved] = 0.0
    torsion_angles_mask[unsolved] = 0.0

    return {
        "aatype": aatype,
        "trans_1": trans_1,
        "rotmats_1": rotmats_1,
        "torsion_angles_sin_cos": torsion_angles_sin_cos,
        "torsion_angles_mask": torsion_angles_mask,
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
    full_sequence: str,
    do_rmsd: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    pdb_files = sorted((cluster_dir / "structure").glob("*.cif"))
    if not pdb_files:
        return None

    cluster_name = cluster_dir.name
    embedding_dir = pathlib.Path(cluster_name) / "embedding"
    conformers: List[Dict[str, Any]] = []

    for pdb_path in pdb_files:
        processed = align_structure_to_full_sequence(str(pdb_path), full_sequence)
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
            rmsd_matrix = pairwise_rmsd_matrix(coords, masks.prod(dim=0).float())
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

    full_sequences = load_full_sequences_from_fasta(args.full_sequence_fasta)

    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Using split file: {cluster_split_path}")
    print(f"FASTA file: {args.full_sequence_fasta} ({len(full_sequences)} sequences)")

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
            if cluster_name not in full_sequences:
                raise ValueError(
                    f"Cluster '{cluster_name}' not found in FASTA file"
                )
            records = process_conformers(
                cluster_dir=data_dir / cluster_name,
                split_name=split_name,
                full_sequence=full_sequences[cluster_name],
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
