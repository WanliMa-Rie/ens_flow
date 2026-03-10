import argparse
import collections
import functools as fn
import multiprocessing as mp
import os
import time
from tqdm import tqdm
from typing import Any, Dict, Optional

import numpy as np
import torch
from Bio import PDB

from rna_backbone_design.data import utils
from rna_backbone_design.data import parsers
from rna_backbone_design.data import nucleotide_constants as nc


def get_pdb_features(pdb_path: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Extracts relevant features from a PDB file.

    Args:
        pdb_path: Path to the PDB file.
        verbose: Whether to print verbose output.

    Returns:
        A dictionary containing the extracted features, or None if extraction fails.
    """
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("structure", pdb_path)

        # Extract all chains
        struct_chains = {chain.id.upper(): chain for chain in structure.get_chains()}

        struct_feats = []
        chain_idx = 0

        for chain_id, chain in struct_chains.items():
            # Convert chain id into int
            chain_index = utils.chain_str_to_int(chain_id)
            chain_mol = parsers.process_chain_pdb(
                chain, chain_index, chain_id, verbose=verbose
            )

            if chain_mol is None:
                continue

            # We are only interested in Nucleic Acids for this task
            if chain_mol[-1]["molecule_type"] != "na":
                continue

            chain_mol_constants = chain_mol[-1]["molecule_constants"]
            chain_mol_backbone_atom_name = chain_mol[-1]["molecule_backbone_atom_name"]

            chain_dict = parsers.macromolecule_outputs_to_dict(chain_mol)

            # Center the chain based on backbone
            chain_dict = utils.parse_chain_feats_pdb(
                chain_feats=chain_dict,
                molecule_constants=chain_mol_constants,
                molecule_backbone_atom_name=chain_mol_backbone_atom_name,
            )

            # Add entity_id, sym_id, asym_id for compatibility with concat_np_features
            seq_length = len(chain_dict["aatype"])
            chain_dict["asym_id"] = (chain_idx + 1) * np.ones(seq_length)
            chain_dict["sym_id"] = (chain_idx + 1) * np.ones(
                seq_length
            )  # Simplified assume 1:1
            chain_dict["entity_id"] = 1 * np.ones(seq_length)  # Simplified

            struct_feats.append(chain_dict)
            chain_idx += 1

        if not struct_feats:
            if verbose:
                print(f"No nucleic acid chains found in {pdb_path}")
            return None

        # Concatenate all collected features
        complex_feats = utils.concat_np_features(struct_feats, add_batch_dim=False)

        # Essential keys to keep
        keys_to_keep = [
            "aatype",
            "atom_positions",
            "atom_mask",
            "atom_deoxy",
            "bb_mask",
            # "chain_indices" # Usually mapped to 'asym_id' or similar in utils
        ]

        # Filter and return
        final_feats = {k: complex_feats.get(k) for k in keys_to_keep}

        # Add chain_id (asym_id in complex_feats)
        if "asym_id" in complex_feats:
            final_feats["chain_id"] = complex_feats["asym_id"]

        # Add modeled_idx equivalent (valid residues)
        # Using logic from process_rna_pdb_files.py: (complex_aatype != 20) & (complex_aatype != 26)
        # But here specific to RNA (na)
        # 26 is typically unknown NA in protein-heavy contexts, but let's check constants if needed.
        # For now, if we trust parsers.process_chain_pdb, it handles validity.

        return final_feats

    except Exception as e:
        if verbose:
            print(f"Error processing {pdb_path}: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        pdb_path = sys.argv[1]
        feats = get_pdb_features(pdb_path, verbose=True)
        if feats:
            print("Successfully extracted features:")
            for k, v in feats.items():
                print(f"  {k}: {type(v)} {v.shape if hasattr(v, 'shape') else ''}")
        else:
            print("Failed to extract features.")
