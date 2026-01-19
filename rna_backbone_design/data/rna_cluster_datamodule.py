import math
import pathlib

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from rna_backbone_design.data.rna_cluster_dataset import RNAClusterDataset
from rna_backbone_design.data import utils as du


def length_batching_collate(batch):
    """
    Self-defined 'collate_fn' function.
    In one batch, different sequence may have different length. 'length_batching_collate' is defined
    to deal with this situation by padding.
    """

    def _seq_len(item, key: str) -> int:
        x = item[key]
        if key == "pair_embedding":
            return int(min(x.shape[0], x.shape[1]))
        return int(x.shape[0])

    keys_for_len = [
        "res_mask",
        "aatype",
        "is_na_residue_mask",
        "trans_1",
        "rotmats_1",
        "single_embedding",
        "torsion_angles_mask",
        "torsion_angles_sin_cos",
        "pair_embedding",
    ]

    lengths = [min(_seq_len(b, k) for k in keys_for_len) for b in batch]
    max_len = max(lengths)
    item0 = batch[0]
    B = len(batch)
    padded_batch = {}

    out = item0["res_mask"].new_zeros((B, max_len))
    for i, b in enumerate(batch):
        l = lengths[i]
        out[i, :l] = b["res_mask"][:l]
    padded_batch["res_mask"] = out

    out = item0["aatype"].new_zeros((B, max_len))
    for i, b in enumerate(batch):
        l = lengths[i]
        out[i, :l] = b["aatype"][:l]
    padded_batch["aatype"] = out

    out = item0["is_na_residue_mask"].new_zeros((B, max_len))
    for i, b in enumerate(batch):
        l = lengths[i]
        out[i, :l] = b["is_na_residue_mask"][:l]
    padded_batch["is_na_residue_mask"] = out

    dims = item0["trans_1"].shape[1:]
    out = item0["trans_1"].new_zeros((B, max_len, *dims))
    for i, b in enumerate(batch):
        l = lengths[i]
        out[i, :l] = b["trans_1"][:l]
    padded_batch["trans_1"] = out

    dims = item0["rotmats_1"].shape[1:]
    out = item0["rotmats_1"].new_zeros((B, max_len, *dims))
    for i, b in enumerate(batch):
        l = lengths[i]
        out[i, :l] = b["rotmats_1"][:l]
    padded_batch["rotmats_1"] = out

    dims = item0["single_embedding"].shape[1:]
    out = item0["single_embedding"].new_zeros((B, max_len, *dims))
    for i, b in enumerate(batch):
        l = lengths[i]
        out[i, :l] = b["single_embedding"][:l]
    padded_batch["single_embedding"] = out

    dims = item0["torsion_angles_mask"].shape[1:]
    out = item0["torsion_angles_mask"].new_zeros((B, max_len, *dims))
    for i, b in enumerate(batch):
        l = lengths[i]
        out[i, :l] = b["torsion_angles_mask"][:l]
    padded_batch["torsion_angles_mask"] = out

    dims = item0["torsion_angles_sin_cos"].shape[1:]
    out = item0["torsion_angles_sin_cos"].new_zeros((B, max_len, *dims))
    for i, b in enumerate(batch):
        l = lengths[i]
        out[i, :l] = b["torsion_angles_sin_cos"][:l]
    padded_batch["torsion_angles_sin_cos"] = out

    dim = int(item0["pair_embedding"].shape[-1])
    out = item0["pair_embedding"].new_zeros((B, max_len, max_len, dim))
    for i, b in enumerate(batch):
        l = lengths[i]
        out[i, :l, :l] = b["pair_embedding"][:l, :l]
    padded_batch["pair_embedding"] = out

    if "pdb_name" in item0:
        padded_batch["pdb_name"] = [b["pdb_name"] for b in batch]
    if "cluster_name" in item0:
        padded_batch["cluster_name"] = [b["cluster_name"] for b in batch]
    if "gt_c4_ensemble" in item0:
        padded_batch["gt_c4_ensemble"] = [b.get("gt_c4_ensemble") for b in batch]

    return padded_batch


class RNAClusterDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_cfg = data_cfg

        # Parse data directory from config or default
        self.data_dir = getattr(data_cfg, "data_dir", "data_ensemble/rna_ensemble_data")
        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.num_workers

        self.train_dataset = None
        self.val_ensemble_dataset = None
        self.val_single_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        common_kwargs = {
            "data_dir": self.data_dir,
            "max_length": self.data_cfg.get("max_len", None),
            "return_ensemble": bool(getattr(self.data_cfg, "return_ensemble", True)),
            "max_ensemble_conformers": getattr(self.data_cfg, "max_ensemble_conformers", None),
            "cdhit_identity_threshold": float(getattr(self.data_cfg, "cdhit_identity_threshold", 0.8)),
            "cdhit_word_length": int(getattr(self.data_cfg, "cdhit_word_length", 5)),
            "split_val_fraction": float(getattr(self.data_cfg, "split_val_fraction", 0.1)),
            "split_seed": int(getattr(self.data_cfg, "split_seed", 123)),
            "cdhit_bin": str(getattr(self.data_cfg, "cdhit_bin", "cd-hit-est")),
            "cdhit_threads": int(getattr(self.data_cfg, "cdhit_threads", 0)),
            "split_cache_filename": str(getattr(self.data_cfg, "split_cache_filename", "split_cdhit80.json")),
        }
        if stage in (None, "fit", "validate"):
            self.train_dataset = RNAClusterDataset(
                split="train",
                **common_kwargs,
            )
            self.val_ensemble_dataset = RNAClusterDataset(
                split="val_ensemble",
                **common_kwargs,
            )
            self.val_single_dataset = RNAClusterDataset(
                split="val_single",
                **common_kwargs,
            )
        if stage == "test" or stage is None:
            self.test_dataset = None

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=length_batching_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_ensemble_dataset is None or self.val_single_dataset is None:
            self.setup("fit")
        assert self.val_ensemble_dataset is not None
        assert self.val_single_dataset is not None
        return [
            DataLoader(
                self.val_ensemble_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=length_batching_collate,
                pin_memory=True,
            ),
            DataLoader(
                self.val_single_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=length_batching_collate,
                pin_memory=True,
            ),
        ]

    def test_dataloader(self):
        loaders = self.val_dataloader()
        return loaders[1]
