"""
Runtime datasets for preprocessed RNA conformer records.
"""

import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from rna_backbone_design import utils as eu

logger = eu.get_pylogger(__name__)


class _BaseRNAConformerDataset(Dataset):
    ensemble_as_cluster = False

    def __init__(
        self,
        preprocessed_dir: pathlib.Path,
        data_dir: pathlib.Path,
        split: str,
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.split = split

        self.records: List[Dict[str, Any]] = torch.load(preprocessed_dir, map_location="cpu")

        # cluster_name → {conformer_name: record_index}
        self._cluster_map: Dict[str, Dict[str, int]] = {}
        for idx, rec in enumerate(self.records):
            cluster = str(rec["cluster_name"])
            conformer = str(rec["conformer_name"])
            self._cluster_map.setdefault(cluster, {})[conformer] = idx

        self.cluster_names: List[str] = sorted(self._cluster_map)


    def _load_embeddings(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        embedding_dir = self.data_dir / record["embedding_dir"]
        single_path = next(embedding_dir.glob("*_single.npy"))
        pair_path = next(embedding_dir.glob("*_pair.npy"))
        single_embedding = torch.from_numpy(np.load(single_path).astype(np.float32))
        pair_embedding = torch.from_numpy(np.load(pair_path).astype(np.float32))
        seq_len = int(record["seq_len"])
        assert single_embedding.shape[0] == seq_len, (
            f"single_embedding length {single_embedding.shape[0]} != seq_len {seq_len} "
            f"for {record['cluster_name']}/{record['conformer_name']}"
        )
        assert pair_embedding.shape[0] == seq_len and pair_embedding.shape[1] == seq_len, (
            f"pair_embedding shape {pair_embedding.shape[:2]} != ({seq_len}, {seq_len}) "
            f"for {record['cluster_name']}/{record['conformer_name']}"
        )
        return {
            "single_embedding": single_embedding,
            "pair_embedding": pair_embedding,
        }

    @staticmethod
    def _record_conformer_names(record: Dict[str, Any]) -> Optional[List[str]]:
        names = record.get("conformer_names")
        if names is None:
            return None
        return [str(name) for name in names]

    @staticmethod
    def _anchor_name_from_cluster_name(cluster_name: str) -> str:
        parts = cluster_name.split("_", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError(f"Invalid cluster_name format for anchor lookup: {cluster_name}")
        return parts[1]



class RNAConformerDataset(_BaseRNAConformerDataset):
    """
    Sample-level dataset where each index corresponds to one conformer.
    Containing "train", "val_single", and "test_single" splits.
    """

    ensemble_as_cluster = False

    def __init__(
        self,
        preprocessed_dir: pathlib.Path,
        data_dir: pathlib.Path,
        split: str = "train",
        posterior_nearest_p: int = 3,
    ):
        super().__init__(
            preprocessed_dir=preprocessed_dir,
            data_dir=data_dir,
            split=split,
        )
        self.posterior_nearest_p = int(posterior_nearest_p)

    def _find_nearest_records(self, conf: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return up to p nearest conformer records by RMSD, or [] if unavailable."""
        rmsd_matrix = conf.get("rmsd_matrix")
        conformer_names = self._record_conformer_names(conf)
        if rmsd_matrix is None or conformer_names is None:
            return []

        anchor_name = str(conf["conformer_name"])

        anchor_idx = conformer_names.index(anchor_name)
        row = rmsd_matrix[anchor_idx].clone()
        row[anchor_idx] = float("inf")

        finite_idx = torch.where(torch.isfinite(row))[0]
        if finite_idx.numel() == 0:
            return []

        nearest = finite_idx[torch.argsort(row[finite_idx])][: self.posterior_nearest_p]
        conformer_lookup = self._cluster_map[str(conf["cluster_name"])]
        return [
            self.records[conformer_lookup[conformer_names[i]]]
            for i in nearest.tolist()
            if conformer_names[i] in conformer_lookup
        ]

    def _load_neighbors(self, conf: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Load nearest-neighbor geometry; falls back to self-copy.

        Same-cluster conformers share identical sequence length,
        so no per-residue slicing or masking is needed.
        """
        records = self._find_nearest_records(conf) or [conf]
        return {
            "nbr_trans": torch.stack([r["trans_1"] for r in records]),
            "nbr_rotmats": torch.stack([r["rotmats_1"] for r in records]),
            "nbr_c4": torch.stack([r["c4_coords"] for r in records]),
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, record_idx: int) -> Dict[str, Any]:
        conformer = dict(self.records[record_idx])
        conformer.update(self._load_embeddings(conformer))
        if self.split == "train" and self.posterior_nearest_p > 0:
            conformer.update(self._load_neighbors(conformer))
        return conformer


class RNAClusterDataset(_BaseRNAConformerDataset):
    """Cluster-level dataset where each index corresponds to one cluster."""

    ensemble_as_cluster = True

    def __len__(self) -> int:
        return len(self.cluster_names)

    def __getitem__(self, cluster_idx: int) -> Dict[str, Any]:
        cluster_name = self.cluster_names[cluster_idx]
        cluster_conformers = self._cluster_map[cluster_name]

        anchor_name = self._anchor_name_from_cluster_name(cluster_name)
        anchor_idx = cluster_conformers.get(anchor_name)
        if anchor_idx is None:
            anchor_idx = next(iter(cluster_conformers.values()))

        anchor = dict(self.records[anchor_idx])
        embeddings = self._load_embeddings(anchor)
        anchor.update(embeddings)

        c4_ensemble = [
            self.records[idx]["c4_coords"]
            for idx in cluster_conformers.values()
        ]
        anchor["gt_c4_ensemble"] = torch.stack(c4_ensemble, dim=0)
        return anchor
