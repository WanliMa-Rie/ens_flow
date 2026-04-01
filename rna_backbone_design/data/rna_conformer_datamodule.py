from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from rna_backbone_design import utils as eu
from rna_backbone_design.data.rna_conformer_dataset import (
    RNAClusterDataset,
    RNAConformerDataset,
)

logger = eu.get_pylogger(__name__)


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


# Keys whose first two dims are both L — need symmetric padding.
_PAIR_KEYS = frozenset({"pair_embedding"})

# Tensors with non-sequence-length leading dims, or variable-shape lists.
# Collected as plain python lists, not pad-stacked.
_PASSTHROUGH_KEYS = frozenset({
    "gt_bb_ensemble", "gt_c4_ensemble",
    "conformer_names", "cluster_conformer_names",
})


def conformer_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    L = max(int(item["seq_len"]) for item in batch)
    out: Dict[str, Any] = {"padding_length": L}

    for key in batch[0]:
        values = [item[key] for item in batch]
        v0 = values[0]
        if key in _PASSTHROUGH_KEYS:
            out[key] = values
        elif isinstance(v0, torch.Tensor):
            pad_fn = _pad_pair_tensor if key in _PAIR_KEYS else _pad_tensor
            out[key] = torch.stack([pad_fn(v, L) for v in values])
        elif isinstance(v0, (int, np.integer)):
            out[key] = torch.tensor(values, dtype=torch.long)
        else:
            out[key] = values  # strings, lists, etc.

    if "cluster_name" in out:
        out["cluster_id"] = out["cluster_name"]
    if "conformer_name" in out:
        out["pdb_name"] = out["conformer_name"]
    return out


class RNAConformerDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_cfg = data_cfg
        self.data_dir = pathlib.Path(data_cfg.data_dir)
        self.preprocessed_dir = pathlib.Path(data_cfg.preprocessed_dir)
        self.batch_size = int(data_cfg.batch_size)
        self.num_workers = int(data_cfg.num_workers)
        self.pin_memory = bool(data_cfg.pin_memory)
        self.persistent_workers = bool(data_cfg.persistent_workers) and self.num_workers > 0
        self.drop_last = bool(data_cfg.drop_last)
        self.val_ensemble_as_cluster = bool(data_cfg.val_ensemble_as_cluster)
        self.test_ensemble_as_cluster = bool(data_cfg.test_ensemble_as_cluster)

        self.train_dataset: Optional[RNAConformerDataset] = None
        self.val_ensemble_dataset: Optional[Dataset] = None
        self.val_single_dataset: Optional[RNAConformerDataset] = None
        self.test_ensemble_dataset: Optional[Dataset] = None
        self.test_single_dataset: Optional[RNAConformerDataset] = None

    def _make_dataset(self, path: pathlib.Path, split: str, as_cluster: bool = False) -> Dataset:
        if as_cluster:
            return RNAClusterDataset(path, self.data_dir, split)
        return RNAConformerDataset(path, self.data_dir, split)

    def _make_loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
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
        root = self.preprocessed_dir
        if stage in (None, "fit", "validate"):
            self.train_dataset = self._make_dataset(root / "train_conformers.pt", "train")
            self.val_ensemble_dataset = self._make_dataset(root / "val_ensemble_conformers.pt", "val_ensemble", as_cluster=self.val_ensemble_as_cluster)
            self.val_single_dataset = self._make_dataset(root / "val_single_conformers.pt", "val_single")
        if stage in (None, "test"):
            self.test_ensemble_dataset = self._make_dataset(root / "test_ensemble_conformers.pt", "test_ensemble", as_cluster=self.test_ensemble_as_cluster)
            self.test_single_dataset = self._make_dataset(root / "test_single_conformers.pt", "test_single")

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> List[DataLoader]:
        return [
            self._make_loader(self.val_ensemble_dataset, shuffle=False),
            self._make_loader(self.val_single_dataset, shuffle=False),
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            self._make_loader(self.test_ensemble_dataset, shuffle=False),
            self._make_loader(self.test_single_dataset, shuffle=False),
        ]

