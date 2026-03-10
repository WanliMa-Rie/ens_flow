"""
Inference script for Conditional RNA-FrameFlow.
Modified to work with test_clusters.json and save PDBs directly to cluster directories.
"""

import os
import time
import json
import numpy as np
import hydra
import torch
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import rna_backbone_design.utils as eu
from rna_backbone_design.models.flow_module import FlowModule
from rna_backbone_design.data.rna_conformer_datamodule import (
    RNAConformerDataModule,
    conformer_collate,
)
from rna_backbone_design.data.rna_conformer_dataset import RNAConformerDataset
from rna_backbone_design.data.interpolant import Interpolant
from rna_backbone_design.data import utils as du
from rna_backbone_design.data import all_atom as rna_all_atom
from rna_backbone_design.data import nucleotide_constants
from rna_backbone_design.analysis import utils as au

torch.set_float32_matmul_precision("high")
log = eu.get_pylogger(__name__)


class TestClusterDataset(torch.utils.data.Dataset):
    """Dataset wrapper that filters RNAConformerDataset instances by cluster."""

    def __init__(self, datasets: list[RNAConformerDataset], test_cluster_names: list[str]):
        """
        Args:
            datasets: List of RNAConformerDataset instances to search from.
            test_cluster_names: Cluster names to include.
        """
        self.test_cluster_names = set(test_cluster_names)

        self.items = []  # List of (dataset, idx, cluster_name)
        for dataset in datasets:
            if dataset is None:
                continue
            if dataset.ensemble_as_cluster:
                for idx, cluster_name in enumerate(dataset.cluster_names):
                    if cluster_name in self.test_cluster_names:
                        self.items.append((dataset, idx, cluster_name))
            else:
                for idx, record in enumerate(dataset.records):
                    cluster_name = str(record["cluster_name"])
                    if cluster_name in self.test_cluster_names:
                        self.items.append((dataset, idx, cluster_name))

        log.info(
            "TestClusterDataset: found %d items for %d requested clusters",
            len(self.items),
            len(self.test_cluster_names),
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        dataset, base_idx, cluster_name = self.items[idx]
        item = dict(dataset[base_idx])
        short_name = "_".join(cluster_name.split("_")[:2])
        item["cluster_id"] = short_name
        item["full_cluster_name"] = cluster_name
        return item



class Sampler:
    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        if os.path.isdir(ckpt_path):
            ckpts = [
                os.path.join(ckpt_path, f)
                for f in os.listdir(ckpt_path)
                if f.endswith(".ckpt")
            ]
            if len(ckpts) != 1:
                raise ValueError(
                    f"inference.ckpt_path is a directory but contains {len(ckpts)} .ckpt files: {ckpt_path}"
                )
            ckpt_path = ckpts[0]
        ckpt_dir = os.path.dirname(ckpt_path)
        # Attempt to load config from checkpoint dir
        config_path = os.path.join(ckpt_dir, "config.yaml")
        if not os.path.exists(config_path):
            # Fallback to flashipa named one if exists
            config_path = os.path.join(ckpt_dir, "config_flashipa.yaml")
        
        if os.path.exists(config_path):
            ckpt_cfg = OmegaConf.load(config_path)
            # Set-up config.
            OmegaConf.set_struct(cfg, False)
            OmegaConf.set_struct(ckpt_cfg, False)
            # Merge checkpoint config with inference config (inference config overrides checkpoint)
            cfg = OmegaConf.merge(ckpt_cfg, cfg)
        
        cfg.experiment.checkpointer.dirpath = "./"

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        self._output_dir = self._infer_cfg.output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f"Saving results to {self._output_dir}")
        
        # Save merged config for reproducibility
        with open(os.path.join(self._output_dir, "inference_config.yaml"), "w") as f:
            OmegaConf.save(config=self._cfg, f=f)

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path, cfg=cfg, map_location="cpu", weights_only=False
        )

        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._output_dir = self._output_dir

    def run_sampling(self):
        devices = self._infer_cfg.num_gpus
        log.info(f"Using {devices} devices (configured via num_gpus)")

        # Load test clusters from JSON
        test_clusters_path = getattr(self._infer_cfg, "test_clusters_json", 
                                     "/projects/u6bk/wanli/inference/test_clusters.json")
        with open(test_clusters_path, "r") as f:
            test_clusters_data = json.load(f)
        
        # Select which test split(s) to use
        split = getattr(self._infer_cfg, "test_cluster_split", "all")
        if split in {"all", "both"}:
            all_test_clusters = []
            if "test_single" in test_clusters_data:
                all_test_clusters.extend(test_clusters_data["test_single"])
            if "test_ensemble" in test_clusters_data:
                all_test_clusters.extend(test_clusters_data["test_ensemble"])
        elif split in {"test_single", "single"}:
            all_test_clusters = test_clusters_data.get("test_single", [])
        elif split in {"test_ensemble", "ensemble"}:
            all_test_clusters = test_clusters_data.get("test_ensemble", [])
        else:
            raise ValueError(
                f"Unsupported inference.test_cluster_split='{split}'. "
                "Use one of: all, both, test_single, single, test_ensemble, ensemble."
            )
        
        log.info(
            f"Loaded {len(all_test_clusters)} test clusters from {test_clusters_path} "
            f"(split={split})"
        )

        datamodule = RNAConformerDataModule(self._cfg.data_cfg)
        datamodule.setup(stage="test")

        if split in {"all", "both"}:
            datasets = [
                datamodule.test_ensemble_dataset,
                datamodule.test_single_dataset,
            ]
        elif split in {"test_single", "single"}:
            datasets = [datamodule.test_single_dataset]
        else:
            datasets = [datamodule.test_ensemble_dataset]

        test_dataset = TestClusterDataset(datasets, all_test_clusters)
        if len(test_dataset) == 0:
            log.warning("No matching conformers/clusters found for inference; exiting early.")
            return

        dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(getattr(self._cfg.data_cfg, "num_workers", 0)),
            collate_fn=conformer_collate,
            pin_memory=bool(getattr(self._cfg.data_cfg, "pin_memory", True)),
        )

        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy="ddp" if (isinstance(devices, int) and devices > 1) or (isinstance(devices, list) and len(devices) > 1) else "auto",
            devices=devices if torch.cuda.is_available() else "auto",
        )

        start_time = time.time()
        
        # Override predict_step to use simpler sampling
        self._flow_module._infer_cfg = self._infer_cfg
        
        predict_out = trainer.predict(self._flow_module, dataloaders=dataloader)
        elapsed_time = time.time() - start_time

        log.info(f"Finished in {elapsed_time:.2f}s")
        log.info(f"Generated samples are stored here: {self._output_dir}")
        
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@hydra.main(
    version_base=None, config_path="configs", config_name="inference"
)
def run(cfg: DictConfig) -> None:
    if cfg.inference.run_inference:
        log.info("Starting inference")
        sampler = Sampler(cfg)
        sampler.run_sampling()


if __name__ == "__main__":
    run()
