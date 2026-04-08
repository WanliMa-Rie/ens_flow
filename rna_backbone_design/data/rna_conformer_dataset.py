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
        max_seq_len: Optional[int] = None,
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.split = split

        all_records: List[Dict[str, Any]] = torch.load(preprocessed_dir, map_location="cpu")
        self.records = [
            r for r in all_records
            if r.get("res_mask") is not None and r["res_mask"].sum() >= 16
        ]
        n_dropped = len(all_records) - len(self.records)
        if n_dropped > 0:
            logger.info(f"[{split}] Dropped {n_dropped} records with fewer than 16 resolved residues")
        if max_seq_len is not None:
            before = len(self.records)
            self.records = [r for r in self.records if int(r["seq_len"]) <= max_seq_len]
            logger.info(
                f"[{split}] Filtered {before} → {len(self.records)} records "
                f"with seq_len <= {max_seq_len}"
            )

        # cluster_name → {conformer_name: record_index}
        self._cluster_map: Dict[str, Dict[str, int]] = {}
        for idx, rec in enumerate(self.records):
            cluster = str(rec["cluster_name"])
            conformer = str(rec["conformer_name"])
            self._cluster_map.setdefault(cluster, {})[conformer] = idx

        self.cluster_names: List[str] = sorted(self._cluster_map)


    def _load_embeddings(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        embedding_dir = self.data_dir / record["embedding_dir"]
        npz_path = next(embedding_dir.glob("*_best_embeddings.npz"))
        data = np.load(npz_path)
        single_embedding = torch.from_numpy(data["single_embeddings"].astype(np.float32))
        pair_embedding = torch.from_numpy(data["pair_embeddings"].astype(np.float32))
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
        max_seq_len: Optional[int] = None,
    ):
        super().__init__(
            preprocessed_dir=preprocessed_dir,
            data_dir=data_dir,
            split=split,
            max_seq_len=max_seq_len,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, record_idx: int) -> Dict[str, Any]:
        conformer = dict(self.records[record_idx])
        conformer.update(self._load_embeddings(conformer))
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
