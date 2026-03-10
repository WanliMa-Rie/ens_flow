"""
Lightweight dataset backed by precomputed .pt files.
"""

from collections import OrderedDict
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from rna_backbone_design import utils as eu

logger = eu.get_pylogger(__name__)


_REQUIRED_RECORD_KEYS = {
    "conformer_name",
    "cluster_name",
    "seq_len",
    "embedding_dir",
    "aatype",
    "trans_1",
    "rotmats_1",
    "torsion_angles_sin_cos",
    "torsion_angles_mask",
    "res_mask",
    "is_na_residue_mask",
    "c4_coords",
    "bb_coords",
}


def _validate_record_schema(record: Dict[str, Any], index: int) -> None:
    missing = sorted(_REQUIRED_RECORD_KEYS.difference(record.keys()))
    if not missing:
        return
    conf_name = record.get("conformer_name", "<unknown>")
    cluster_name = record.get("cluster_name", "<unknown>")
    raise ValueError(
        "Invalid conformer record at index "
        f"{index} (cluster={cluster_name}, conformer={conf_name}). "
        f"Missing keys: {missing}"
    )


class RNAConformerDataset(Dataset):
    """Dataset class for RNA clusters with preprocessed .pt files."""

    def __init__(
        self,
        preprocessed_dir: str,
        data_dir: str,
        split: str = "train",
        posterior_nearest_p: int = 3,
        val_ensemble_as_cluster: bool = False,
        test_ensemble_as_cluster: bool = False,
        embedding_cache_size: int = 4,
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.split = split
        self.posterior_nearest_p = posterior_nearest_p
        self.embedding_cache_size = max(0, int(embedding_cache_size))
        self._embedding_cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        self.ensemble_as_cluster = bool(
            (val_ensemble_as_cluster and split == "val_ensemble")
            or (test_ensemble_as_cluster and split == "test_ensemble")
        )

        self.records: List[Dict[str, Any]] = torch.load(preprocessed_dir, map_location="cpu")
        num_loaded = len(self.records)

        for idx, rec in enumerate(self.records):
            _validate_record_schema(rec, idx)

        if self.split in {"val_ensemble", "val_single", "test_ensemble", "test_single"}:
            num_records_before_filter = len(self.records)
            self.records = [
                rec
                for rec in self.records
                if not self._is_msa_conformer_name(str(rec["conformer_name"]))
            ]
            num_filtered = num_records_before_filter - len(self.records)
            self._print_dataset_summary(
                f"{self.split}: {num_records_before_filter} total conformers, "
                f"kept {len(self.records)} real conformers."
            )
        else:
            self._print_dataset_summary(f"train: {num_loaded} total conformers.")

        self.cluster_to_name_to_idx: Dict[str, Dict[str, int]] = {}
        for i, rec in enumerate(self.records):
            cname = rec["cluster_name"]
            self.cluster_to_name_to_idx.setdefault(cname, {})
            self.cluster_to_name_to_idx[cname][rec["conformer_name"]] = i

        self.cluster_to_indices: Dict[str, List[int]] = {}
        for i, rec in enumerate(self.records):
            self.cluster_to_indices.setdefault(rec["cluster_name"], []).append(i)
        self.cluster_to_eval_indices: Dict[str, List[int]] = self.cluster_to_indices
        self.cluster_names: List[str] = sorted(self.cluster_to_indices.keys())

        if self.ensemble_as_cluster:
            filtered: Dict[str, List[int]] = {}
            dropped_clusters = 0
            for cluster_name, indices in self.cluster_to_indices.items():
                keep = [
                    i
                    for i in indices
                    if not self._is_msa_conformer_name(str(self.records[i]["conformer_name"]))
                ]
                if keep:
                    filtered[cluster_name] = keep
                else:
                    dropped_clusters += 1

            self.cluster_to_eval_indices = filtered
            self.cluster_names = sorted(self.cluster_to_eval_indices.keys())
            if dropped_clusters > 0:
                logger.warning(
                    "Dropped %d val_ensemble clusters with zero non-MSA conformers.",
                    dropped_clusters,
                )

    @staticmethod
    def _first_embedding_file_or_raise(
        embedding_dir: pathlib.Path,
        glob_pattern: str,
        source_name: str,
    ) -> pathlib.Path:
        files = sorted(embedding_dir.glob(glob_pattern))
        if files:
            return files[0]

        available = sorted(p.name for p in embedding_dir.glob("*.npy"))
        raise FileNotFoundError(
            f"Missing {source_name} embedding matching '{glob_pattern}' under '{embedding_dir}'. "
            f"Available .npy files: {available}"
        )

    def _get_cached_cluster_embeddings(
        self,
        embedding_dir: pathlib.Path,
    ) -> Dict[str, torch.Tensor]:
        cache_key = str(embedding_dir)
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            self._embedding_cache.move_to_end(cache_key)
            return cached

        single_path = self._first_embedding_file_or_raise(
            embedding_dir=embedding_dir,
            glob_pattern="*_single.npy",
            source_name="single",
        )
        pair_path = self._first_embedding_file_or_raise(
            embedding_dir=embedding_dir,
            glob_pattern="*_pair.npy",
            source_name="pair",
        )
        single_embedding = torch.from_numpy(np.load(single_path).astype(np.float32))
        pair_embedding = torch.from_numpy(np.load(pair_path).astype(np.float32))

        loaded = {
            "single_embedding": single_embedding,
            "pair_embedding": pair_embedding,
        }
        if self.embedding_cache_size > 0:
            self._embedding_cache[cache_key] = loaded
            self._embedding_cache.move_to_end(cache_key)
            while len(self._embedding_cache) > self.embedding_cache_size:
                self._embedding_cache.popitem(last=False)
        return loaded

    @staticmethod
    def _crop_record(record: Dict[str, Any], seq_len: int) -> Dict[str, Any]:
        out = dict(record)
        tensor_keys = [
            "aatype",
            "trans_1",
            "rotmats_1",
            "torsion_angles_sin_cos",
            "torsion_angles_mask",
            "res_mask",
            "is_na_residue_mask",
            "c4_coords",
            "bb_coords",
        ]
        for key in tensor_keys:
            value = out.get(key, None)
            if isinstance(value, torch.Tensor):
                out[key] = value[:seq_len]
        out["seq_len"] = int(seq_len)
        return out

    @staticmethod
    def _is_msa_conformer_name(stem: str) -> bool:
        s = stem.lower()
        return s.startswith("msa_group_") or s.startswith("msa_") or "msa_group_" in s

    @staticmethod
    @rank_zero_only
    def _print_dataset_summary(msg: str) -> None:
        print(msg)

    def __len__(self) -> int:
        if self.ensemble_as_cluster:
            return len(self.cluster_names)
        return len(self.records)

    def _select_neighbors(
        self,
        conf: Dict[str, Any],
        seq_len: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.posterior_nearest_p <= 0:
            return None
        rmsd_matrix = conf.get("rmsd_matrix")
        rmsd_names = conf.get("rmsd_names")
        if rmsd_matrix is None or rmsd_names is None:
            return None

        anchor = conf["conformer_name"]
        anchor_index = rmsd_names.index(anchor)
        row = rmsd_matrix[anchor_index].clone()
        row[anchor_index] = float("inf")

        finite_idx = torch.where(torch.isfinite(row))[0]
        if finite_idx.numel() == 0:
            return None

        ordered = finite_idx[torch.argsort(row[finite_idx])]
        nbr_names = [rmsd_names[i] for i in ordered[: self.posterior_nearest_p].tolist()]

        cname = conf["cluster_name"]
        name_to_idx = self.cluster_to_name_to_idx.get(cname, {})

        nbr_records = []
        for name in nbr_names:
            j = name_to_idx.get(name)
            if j is not None:
                nbr_records.append(self.records[j])

        if not nbr_records:
            return None

        return {
            "nbr_trans": torch.stack([r["trans_1"][:seq_len] for r in nbr_records], dim=0),
            "nbr_rotmats": torch.stack([r["rotmats_1"][:seq_len] for r in nbr_records], dim=0),
            "nbr_mask": torch.stack(
                [torch.ones(seq_len, dtype=torch.float32) for _ in nbr_records], dim=0
            ),
            "nbr_c4": torch.stack([r["c4_coords"][:seq_len] for r in nbr_records], dim=0),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.ensemble_as_cluster:
            return self._getitem_val_ensemble_cluster(idx)

        conf = dict(self.records[idx])
        seq_len = int(conf["seq_len"])

        embedding_dir = self.data_dir / conf["embedding_dir"]
        embeddings = self._get_cached_cluster_embeddings(embedding_dir)
        single_embedding = embeddings["single_embedding"]
        pair_embedding = embeddings["pair_embedding"]
        seq_len = min(
            seq_len,
            int(single_embedding.shape[0]),
            int(pair_embedding.shape[0]),
            int(pair_embedding.shape[1]),
        )

        conf = self._crop_record(conf, seq_len)
        conf["single_embedding"] = single_embedding[:seq_len]
        conf["pair_embedding"] = pair_embedding[:seq_len, :seq_len]

        if self.split == "train" and self.posterior_nearest_p > 0:
            nbr = self._select_neighbors(conf, seq_len)
            if nbr is None:
                nbr = {
                    "nbr_trans": conf["trans_1"][:seq_len].unsqueeze(0),
                    "nbr_rotmats": conf["rotmats_1"][:seq_len].unsqueeze(0),
                    "nbr_mask": torch.ones(1, seq_len, dtype=torch.float32),
                    "nbr_c4": conf["c4_coords"][:seq_len].unsqueeze(0),
                }
            conf.update(nbr)

        return conf

    def _getitem_val_ensemble_cluster(self, idx: int) -> Dict[str, Any]:
        cluster_name = self.cluster_names[idx]
        indices = self.cluster_to_eval_indices[cluster_name]
        anchor = dict(self.records[indices[0]])
        seq_len = int(anchor["seq_len"])

        embedding_dir = self.data_dir / anchor["embedding_dir"]
        embeddings = self._get_cached_cluster_embeddings(embedding_dir)
        single_embedding = embeddings["single_embedding"]
        pair_embedding = embeddings["pair_embedding"]
        seq_len = min(
            seq_len,
            int(single_embedding.shape[0]),
            int(pair_embedding.shape[0]),
            int(pair_embedding.shape[1]),
        )

        anchor = self._crop_record(anchor, seq_len)
        anchor["single_embedding"] = single_embedding[:seq_len]
        anchor["pair_embedding"] = pair_embedding[:seq_len, :seq_len]

        conformer_records: List[Dict[str, Any]] = []
        conformer_names: List[str] = []
        min_len = seq_len
        for j in indices:
            rec = self.records[j]
            bb_len = int(rec["bb_coords"].shape[0])
            c4_len = int(rec["c4_coords"].shape[0])
            conf_len = min(bb_len, c4_len)
            if conf_len <= 0:
                continue
            conformer_records.append(rec)
            conformer_names.append(str(rec["conformer_name"]))
            min_len = min(min_len, conf_len)

        if not conformer_records:
            raise ValueError(
                f"Cluster '{cluster_name}' has no valid conformers for ensemble assembly."
            )

        bb_ensemble = [
            rec["bb_coords"][:min_len].reshape(min_len * 3, 3) for rec in conformer_records
        ]
        c4_ensemble = [rec["c4_coords"][:min_len] for rec in conformer_records]

        anchor["cluster_name"] = cluster_name
        anchor["cluster_conformer_names"] = conformer_names
        anchor["cluster_size"] = len(conformer_records)
        anchor["gt_bb_ensemble"] = torch.stack(bb_ensemble, dim=0)
        anchor["gt_c4_ensemble"] = torch.stack(c4_ensemble, dim=0)
        return anchor
