from __future__ import annotations

import functools
import json
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from rna_backbone_design import utils as eu
from rna_backbone_design.data.rna_conformer_dataset import RNAConformerDataset

logger = eu.get_pylogger(__name__)


def infer_embedding_dim(data_dir: str) -> int:
    """Probe one single-embedding file under *data_dir* to determine the feature dimension."""
    candidates = sorted(pathlib.Path(data_dir).glob("cluster_*/embedding/*_single.npy"))
    if not candidates:
        raise FileNotFoundError(
            f"No single embedding files matching 'cluster_*/embedding/*_single.npy' "
            f"under '{data_dir}'. Check data_dir."
        )
    arr = np.load(candidates[0], mmap_mode="r")
    return int(arr.shape[-1])


def _pad_tensor(t: torch.Tensor, length: int) -> torch.Tensor:
    l = t.shape[0]
    if l == length:
        return t
    if l > length:
        return t[:length]
    pad_shape = (length - l,) + t.shape[1:]
    return torch.cat([t, t.new_zeros(pad_shape)], dim=0)


def _pad_pair_tensor(t: torch.Tensor, length: int) -> torch.Tensor:
    l0 = int(t.shape[0])
    if l0 >= length:
        return t[:length, :length]
    out = t.new_zeros((length, length, *t.shape[2:]))
    out[:l0, :l0] = t
    return out


_TENSOR_KEYS = {
    "aatype",
    "res_mask",
    "is_na_residue_mask",
    "trans_1",
    "rotmats_1",
    "torsion_angles_sin_cos",
    "torsion_angles_mask",
    "c4_coords",
    "bb_coords",
    "single_embedding",
}

_PAIR_TENSOR_KEYS = {
    "pair_embedding",
}

_LIST_KEYS = {
    "nbr_trans",
    "nbr_rotmats",
    "nbr_mask",
    "nbr_c4",
    "gt_bb_ensemble",
    "gt_c4_ensemble",
    "cluster_conformer_names",
}


def conformer_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}

    padding_length = max(int(item["seq_len"]) for item in batch)
    out: Dict[str, Any] = {"padding_length": padding_length}

    for key in _TENSOR_KEYS:
        values = [item.get(key) for item in batch]
        if any(v is None for v in values):
            continue
        out[key] = torch.stack([_pad_tensor(v, padding_length) for v in values])

    for key in _PAIR_TENSOR_KEYS:
        values = [item.get(key) for item in batch]
        if any(v is None for v in values):
            continue
        out[key] = torch.stack([_pad_pair_tensor(v, padding_length) for v in values])

    for key in _LIST_KEYS:
        values = [item.get(key) for item in batch]
        if any(v is None for v in values):
            continue
        out[key] = values

    out["seq_len"] = torch.tensor([int(item["seq_len"]) for item in batch], dtype=torch.long)
    out["conformer_name"] = [item["conformer_name"] for item in batch]
    out["cluster_name"] = [item["cluster_name"] for item in batch]
    if all("cluster_size" in item for item in batch):
        out["cluster_size"] = torch.tensor(
            [int(item["cluster_size"]) for item in batch], dtype=torch.long
        )

    out["cluster_id"] = out["cluster_name"]
    out["pdb_name"] = out["conformer_name"]
    return out


class RNAConformerDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_cfg = data_cfg
        self.data_dir = str(getattr(data_cfg, "data_dir", "data_ensemble/rna_ensemble_data"))
        self.batch_size = int(getattr(data_cfg, "batch_size", 1))
        self.num_workers = int(getattr(data_cfg, "num_workers", 0))
        self.pin_memory = bool(getattr(data_cfg, "pin_memory", True))
        self.persistent_workers = (
            bool(getattr(data_cfg, "persistent_workers", self.num_workers > 0))
            and self.num_workers > 0
        )
        self.drop_last = bool(getattr(data_cfg, "drop_last", False))
        self.posterior_nearest_p = int(getattr(data_cfg, "posterior_nearest_p", 0))
        self.embedding_cache_size = int(getattr(data_cfg, "embedding_cache_size", 4))
        self.val_ensemble_as_cluster = bool(getattr(data_cfg, "val_ensemble_as_cluster", True))
        self.test_ensemble_as_cluster = bool(getattr(data_cfg, "test_ensemble_as_cluster", True))

        self.train_dataset: Optional[RNAConformerDataset] = None
        self.val_ensemble_dataset: Optional[RNAConformerDataset] = None
        self.val_single_dataset: Optional[RNAConformerDataset] = None
        self.test_ensemble_dataset: Optional[RNAConformerDataset] = None
        self.test_single_dataset: Optional[RNAConformerDataset] = None

    @staticmethod
    def _cluster_names_from_dataset(dataset: Optional[RNAConformerDataset]) -> List[str]:
        if dataset is None:
            return []
        if hasattr(dataset, "cluster_to_indices"):
            return sorted(str(name) for name in dataset.cluster_to_indices.keys())
        return sorted({str(rec["cluster_name"]) for rec in dataset.records})

    @functools.cached_property
    def embedding_dim(self) -> int:
        return infer_embedding_dim(self.data_dir)

    @staticmethod
    def _resolve_path(path_like: str | pathlib.Path) -> pathlib.Path:
        p = pathlib.Path(path_like)
        if p.is_absolute():
            return p
        try:
            from hydra.utils import get_original_cwd  # type: ignore
            return pathlib.Path(get_original_cwd()) / p
        except Exception:
            return pathlib.Path.cwd() / p

    def _preprocessed_root(self) -> pathlib.Path:
        pre = getattr(self.data_cfg, "preprocessed_dir", None)
        if pre is not None:
            return self._resolve_path(pre)
        return self._resolve_path(pathlib.Path(self.data_dir) / "preprocessed_data")

    def _preprocessed_path(self, cfg_key: str, default_name: str) -> pathlib.Path:
        override = getattr(self.data_cfg, cfg_key, None)
        if override:
            return self._resolve_path(override)
        return self._preprocessed_root() / default_name

    def _make_dataset(self, path: pathlib.Path, split: str) -> RNAConformerDataset:
        if not path.exists():
            raise FileNotFoundError(f"Preprocessed file not found: {path}")
        return RNAConformerDataset(
            preprocessed_dir=str(path),
            data_dir=self.data_dir,
            split=split,
            posterior_nearest_p=self.posterior_nearest_p if split == "train" else 0,
            embedding_cache_size=self.embedding_cache_size,
            val_ensemble_as_cluster=self.val_ensemble_as_cluster,
            test_ensemble_as_cluster=self.test_ensemble_as_cluster,
        )

    def _make_loader(self, dataset: RNAConformerDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=conformer_collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last and shuffle,
        )


    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit", "validate"):
            if self.train_dataset is None:
                self.train_dataset = self._make_dataset(
                    self._preprocessed_path("train_preprocessed_path", "train_conformers.pt"),
                    split="train",
                )
            if self.val_ensemble_dataset is None:
                self.val_ensemble_dataset = self._make_dataset(
                    self._preprocessed_path(
                        "val_ensemble_preprocessed_path", "val_ensemble_conformers.pt"
                    ),
                    split="val_ensemble",
                )
            if self.val_single_dataset is None:
                self.val_single_dataset = self._make_dataset(
                    self._preprocessed_path(
                        "val_single_preprocessed_path", "val_single_conformers.pt"
                    ),
                    split="val_single",
                )
        if stage in (None, "test"):
            if self.test_ensemble_dataset is None:
                self.test_ensemble_dataset = self._make_dataset(
                    self._preprocessed_path(
                        "test_ensemble_preprocessed_path", "test_ensemble_conformers.pt"
                    ),
                    split="test_ensemble",
                )
            if self.test_single_dataset is None:
                self.test_single_dataset = self._make_dataset(
                    self._preprocessed_path(
                        "test_single_preprocessed_path", "test_single_conformers.pt"
                    ),
                    split="test_single",
                )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> List[DataLoader]:
        if self.val_ensemble_dataset is None or self.val_single_dataset is None:
            self.setup("validate")
        assert self.val_ensemble_dataset is not None
        assert self.val_single_dataset is not None
        return [
            self._make_loader(self.val_ensemble_dataset, shuffle=False),
            self._make_loader(self.val_single_dataset, shuffle=False),
        ]

    def test_dataloader(self) -> List[DataLoader]:
        if self.test_ensemble_dataset is None or self.test_single_dataset is None:
            self.setup("test")
        assert self.test_ensemble_dataset is not None
        assert self.test_single_dataset is not None
        return [
            self._make_loader(self.test_ensemble_dataset, shuffle=False),
            self._make_loader(self.test_single_dataset, shuffle=False),
        ]
