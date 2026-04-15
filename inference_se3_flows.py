"""
Inference script for Conditional RNA-FrameFlow.
Reads test clusters from split_cdhit80.json in ensemble_dataset,
runs level-dependent sampling, and saves .cif files to the output directory.
Metrics are computed separately via evaluate_metrics.py.
"""

import os
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

torch.set_float32_matmul_precision("high")
log = eu.get_pylogger(__name__)


class TestClusterDataset(torch.utils.data.Dataset):
    """Dataset wrapper that filters cluster-level datasets by cluster name.

    Both test datasets are now cluster-level (`RNAClusterDataset`), so each
    cluster is scored exactly once — no more CIF overwrite from repeated
    records per cluster.
    """

    def __init__(self, datasets: list[RNAConformerDataset], test_cluster_names: list[str]):
        """
        Args:
            datasets: List of cluster-level datasets to search from.
            test_cluster_names: Cluster names to include.
        """
        self.test_cluster_names = set(test_cluster_names)

        self.items = []  # List of (dataset, idx, cluster_name)
        for dataset in datasets:
            if dataset is None:
                continue
            if not dataset.ensemble_as_cluster:
                raise ValueError(
                    f"TestClusterDataset expects cluster-level datasets "
                    f"(ensemble_as_cluster=True); got {type(dataset).__name__}."
                )
            for idx, cluster_name in enumerate(dataset.cluster_names):
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
        item["cluster_id"] = cluster_name
        item["full_cluster_name"] = cluster_name
        return item



class Sampler:
    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: Hydra config loaded from inference.yaml (inference-only
                 settings + optional data_cfg overrides).  The training
                 config is loaded from the checkpoint directory and used
                 as the base; ``cfg`` is then merged on top so that
                 inference.yaml takes precedence.
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
                    f"inference.ckpt_path is a directory but contains "
                    f"{len(ckpts)} .ckpt files: {ckpt_path}"
                )
            ckpt_path = ckpts[0]
        ckpt_dir = os.path.dirname(ckpt_path)

        # Load training config from checkpoint directory (base).
        ckpt_config_path = os.path.join(ckpt_dir, "config.yaml")
        if not os.path.exists(ckpt_config_path):
            ckpt_config_path = os.path.join(ckpt_dir, "config_flashipa.yaml")
        if not os.path.exists(ckpt_config_path):
            raise FileNotFoundError(
                f"No config.yaml found in checkpoint directory: {ckpt_dir}"
            )

        ckpt_cfg = OmegaConf.load(ckpt_config_path)
        OmegaConf.set_struct(ckpt_cfg, False)
        OmegaConf.set_struct(cfg, False)

        # Merge: ckpt config (model, data, training) as base,
        # inference.yaml (inference settings + data_cfg overrides) on top.
        cfg = OmegaConf.merge(ckpt_cfg, cfg)

        cfg.experiment.checkpointer.dirpath = "./"

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Determine level from config (old ckpts may lack stochastic_bridge)
        bridge_cfg = OmegaConf.select(cfg, "stochastic_bridge", default=None)
        self._level = int(bridge_cfg.level) if bridge_cfg is not None and "level" in bridge_cfg else 1

        # Output directory: /projects/u6bk/wanli/inference_ours/ens_flow{level}/
        default_output = f"/projects/u6bk/wanli/inference_ours/ens_flow{self._level}"
        self._output_dir = str(
            getattr(self._infer_cfg, "output_dir", default_output) or default_output
        )
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
        self._flow_module._output_format = "cif"

    def run_sampling(self):
        devices = self._infer_cfg.num_gpus
        log.info(f"Using {devices} devices (configured via num_gpus)")

        # --- Load test cluster names from split JSON in ensemble_dataset ---
        data_dir = self._cfg.data_cfg.data_dir
        split_json_path = os.path.join(data_dir, "split_cdhit80.json")
        with open(split_json_path, "r") as f:
            split_data = json.load(f)

        # Select which split(s) to run inference on
        split = getattr(self._infer_cfg, "test_cluster_split", "all")
        if split in {"all", "both"}:
            all_test_clusters = []
            if "test_single" in split_data:
                all_test_clusters.extend(split_data["test_single"])
            if "test_ensemble" in split_data:
                all_test_clusters.extend(split_data["test_ensemble"])
        elif split in {"test_single", "single"}:
            all_test_clusters = split_data.get("test_single", [])
        elif split in {"test_ensemble", "ensemble"}:
            all_test_clusters = split_data.get("test_ensemble", [])
        else:
            raise ValueError(
                f"Unsupported inference.test_cluster_split='{split}'. "
                "Use one of: all, both, test_single, single, test_ensemble, ensemble."
            )

        log.info(
            f"Loaded {len(all_test_clusters)} test clusters from {split_json_path} "
            f"(split={split})"
        )

        # --- Build datasets via the existing DataModule ---
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

        self._flow_module._infer_cfg = self._infer_cfg

        predict_out = trainer.predict(self._flow_module, dataloaders=dataloader)

        log.info(f"Generated samples stored at: {self._output_dir}")

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
