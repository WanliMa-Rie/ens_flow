from typing import Any, Dict, List, Optional
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from rna_backbone_design.data import data_transforms
from rna_backbone_design.data import nucleotide_constants as nc
from rna_backbone_design.data import utils as du
from rna_backbone_design.data.rigid_utils import Rigid
from rna_backbone_design.data.cdhit_split import CDHitSplitConfig, load_or_build_split_manifest, collect_valid_cluster_dirs


class RNAClusterDataset(Dataset):
    """
    Dataset that iterates over clusters of RNA structures.
    For each cluster, it randomly samples one PDB structure (conformer)
    and pairs it with the cluster's shared embeddings.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train", "val_ensemble", "val_single", "val"
        max_length: Optional[int] = None,
        overfit: bool = False,
        return_ensemble: bool = True,
        max_ensemble_conformers: Optional[int] = None,
        cdhit_identity_threshold: float = 0.8,
        cdhit_word_length: int = 5,
        split_val_fraction: float = 0.1,
        split_seed: int = 123,
        cdhit_bin: str = "cd-hit-est",
        cdhit_threads: int = 0,
        split_cache_filename: str = "split_cdhit80.json",
    ):
        """
        Args:
            data_dir: Path to the root directory containing cluster folders.
                      (e.g., 'data_ensemble/rna_ensemble_data')
            split: Data split to use (train, val, or test).
            max_length: Optional maximum sequence length to filter or crop.
            overfit: If True, restricts the dataset to a small subset for debugging.
        """
        self.data_dir = pathlib.Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.overfit = overfit
        self.return_ensemble = return_ensemble
        self.max_ensemble_conformers = max_ensemble_conformers

        valid_dirs = collect_valid_cluster_dirs(self.data_dir)
        split_cfg = CDHitSplitConfig(
            identity_threshold=cdhit_identity_threshold,
            word_length=cdhit_word_length,
            val_fraction=split_val_fraction,
            seed=split_seed,
            cdhit_bin=cdhit_bin,
            threads=cdhit_threads,
            cache_filename=split_cache_filename,
        )
        manifest = load_or_build_split_manifest(self.data_dir, split_cfg)

        if split == "train":
            keep = set(manifest["train"])
        elif split == "val_ensemble":
            keep = set(manifest["val_ensemble"])
        elif split == "val_single":
            keep = set(manifest["val_single"])
        elif split == "val":
            keep = set(manifest["val_ensemble"]) | set(manifest["val_single"])
        else:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of: train, val_ensemble, val_single, val."
            )

        self.clusters = [d for d in valid_dirs if d.name in keep]
        if self.overfit:
            self.clusters = self.clusters[:2]

        self.items: Optional[List[Dict[str, Any]]] = None
        if self.split == "train":
            items: List[Dict[str, Any]] = []
            for cluster_dir in self.clusters:
                feature_dir = cluster_dir / "features"
                pkl_files = sorted(list(feature_dir.glob("*.pkl")))
                for pkl_path in pkl_files:
                    items.append({"cluster_dir": cluster_dir, "pkl_path": pkl_path})
            self.items = items

        print(f"RNAClusterDataset ({self.split}): {len(self.clusters)} clusters")

    def __len__(self) -> int:
        if self.items is not None:
            return len(self.items)
        return len(self.clusters)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.items is not None:
            item = self.items[idx]
            cluster_dir = item["cluster_dir"]
            pkl_path = item["pkl_path"]
        else:
            cluster_dir = self.clusters[idx]
        
        # 1. Select a structure (conformer) from the cluster
        feature_dir = cluster_dir / "features"
        pkl_files = sorted(list(feature_dir.glob("*.pkl")))
        if self.items is None:
            pkl_path = random.choice(pkl_files)
        raw_feats = du.read_pkl(str(pkl_path), verbose=False)
        
        # 2. Load shared embeddings
        embedding_dir = cluster_dir / "embedding"
        # We use the first available embedding; usually there's only one pair of single/pair per cluster
        single_files = sorted(list(embedding_dir.glob("*_single.npy")))
        pair_files = sorted(list(embedding_dir.glob("*_pair.npy")))
        
        single_embedding = torch.from_numpy(np.load(single_files[0])).float()
        pair_embedding = torch.from_numpy(np.load(pair_files[0])).float()

        processed_feats = du.parse_complex_feats(raw_feats)

        modeled_idx = processed_feats.get("modeled_idx", None)
        if modeled_idx is not None and len(modeled_idx) > 0:
            min_idx = int(np.min(modeled_idx))
            max_idx = int(np.max(modeled_idx))
            full_len = int(np.asarray(processed_feats["aatype"]).shape[0])
            processed_feats = {
                k: (
                    v[min_idx : (max_idx + 1)]
                    if (
                        isinstance(v, np.ndarray)
                        and v.ndim >= 1
                        and int(v.shape[0]) == full_len
                    )
                    else v
                )
                for k, v in processed_feats.items()
            }

        # Re-compute Geometry (Frames & Torsions) on-the-fly to match training pipeline
        # -----------------------------------------------------------------------------
        # 1. Convert to tensor
        aatype = torch.as_tensor(processed_feats["aatype"]).long()
        atom_positions = torch.as_tensor(processed_feats["atom_positions"]).double()
        atom_mask = torch.as_tensor(processed_feats["atom_mask"]).double()
        atom_deoxy = torch.as_tensor(processed_feats["atom_deoxy"]).bool()

        num_res = int(aatype.shape[0])
        c4_idx = int(nc.atom_order["C4'"])
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
            na_feats, ["all_atom_positions", "all_atom_mask"], inplace=True
        )
        na_feats = data_transforms.atom27_to_frames(na_feats)
        na_feats = data_transforms.atom27_to_torsion_angles()(na_feats)

        rigids_1 = Rigid.from_tensor_4x4(na_feats["rigidgroups_gt_frames"])[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()

        out: Dict[str, Any] = {
            "aatype": aatype,
            "trans_1": trans_1.float(),
            "rotmats_1": rotmats_1.float(),
            "torsion_angles_sin_cos": na_feats["torsion_angles_sin_cos"][:, :8].float(),
            "torsion_angles_mask": na_feats["torsion_angles_mask"][:, :8].float(),
            "res_mask": res_mask,
            "is_na_residue_mask": is_na_residue_mask,
            "single_embedding": single_embedding,
            "pair_embedding": pair_embedding,
            "pdb_name": pkl_path.stem,
            "cluster_name": cluster_dir.name,
        }

        if self.split == "val_ensemble" and self.return_ensemble:
            gt_coords: List[torch.Tensor] = []
            k = len(pkl_files)
            if self.max_ensemble_conformers is not None:
                k = min(k, int(self.max_ensemble_conformers))
            for p in pkl_files[:k]:
                feats_k = du.parse_complex_feats(du.read_pkl(str(p), verbose=False))
                modeled_idx_k = feats_k.get("modeled_idx", None)
                if modeled_idx_k is not None and len(modeled_idx_k) > 0:
                    min_k = int(np.min(modeled_idx_k))
                    max_k = int(np.max(modeled_idx_k))
                    full_len_k = int(np.asarray(feats_k["aatype"]).shape[0])
                    feats_k = {
                        kk: (
                            vv[min_k : (max_k + 1)]
                            if (
                                isinstance(vv, np.ndarray)
                                and vv.ndim >= 1
                                and int(vv.shape[0]) == full_len_k
                            )
                            else vv
                        )
                        for kk, vv in feats_k.items()
                    }
                atom_pos_k = torch.as_tensor(feats_k["atom_positions"]).float()
                if atom_pos_k.ndim == 3 and int(atom_pos_k.shape[1]) > c4_idx:
                    gt_coords.append(atom_pos_k[:, c4_idx])
                elif atom_pos_k.ndim == 2 and int(atom_pos_k.shape[1]) == 3:
                    gt_coords.append(atom_pos_k)
                else:
                    continue

            if len(gt_coords) >= 2:
                min_len = min(int(x.shape[0]) for x in gt_coords)
                out["gt_c4_ensemble"] = torch.stack([x[:min_len] for x in gt_coords], dim=0)

        return out
