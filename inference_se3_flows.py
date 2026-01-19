"""
Inference script for Conditional RNA-FrameFlow.
Iterates through a chosen split of the RNAClusterDataset and generates structures.
"""

import os
import time
import json
import csv
import numpy as np
import hydra
import torch
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf

import rna_backbone_design.utils as eu
from rna_backbone_design.models.flow_module import FlowModule
from rna_backbone_design.data.rna_cluster_datamodule import RNAClusterDataModule

torch.set_float32_matmul_precision("high")
log = eu.get_pylogger(__name__)


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
        self._output_dir = os.path.join(
            self._infer_cfg.output_dir,
            self._infer_cfg.name,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f"Saving results to {self._output_dir}")
        
        # Save merged config for reproducibility
        with open(os.path.join(self._output_dir, "inference_config.yaml"), "w") as f:
            OmegaConf.save(config=self._cfg, f=f)

        # Read checkpoint and initialize module.
        # Allow loading OmegaConf DictConfig which is not in the default safe globals
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path, cfg=cfg, map_location="cpu", weights_only=False
        )

        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._output_dir = self._output_dir

    def run_sampling(self):
        # Simplify device selection to rely on PyTorch Lightning / CUDA_VISIBLE_DEVICES
        # GPUtil can cause issues with DDP if it returns global indices that conflict with
        # the process's visible devices or if orderings mismatch.
        devices = self._infer_cfg.num_gpus
        log.info(f"Using {devices} devices (configured via num_gpus)")

        split = str(getattr(self._infer_cfg, "dataset_split", "val_single"))
        datamodule = RNAClusterDataModule(self._cfg.data_cfg)
        datamodule.setup(stage="fit")
        if split == "train":
            dataloader = datamodule.train_dataloader()
        elif split == "val_ensemble":
            dataloader = datamodule.val_dataloader()[0]
        elif split == "val_single":
            dataloader = datamodule.val_dataloader()[1]
        else:
            raise ValueError(
                f"Unsupported inference.dataset_split='{split}'. Expected: train, val_ensemble, val_single."
            )

        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy="ddp" if (isinstance(devices, int) and devices > 1) or (isinstance(devices, list) and len(devices) > 1) else "auto",
            devices=devices if torch.cuda.is_available() else "auto",
        )

        start_time = time.time()
        predict_out = trainer.predict(self._flow_module, dataloaders=dataloader)
        elapsed_time = time.time() - start_time

        flat_rows = []
        for batch_out in predict_out:
            if batch_out is None:
                continue
            if isinstance(batch_out, list):
                flat_rows.extend(batch_out)
            else:
                flat_rows.append(batch_out)

        metrics_path_json = os.path.join(self._output_dir, "inference_metrics.json")
        with open(metrics_path_json, "w") as f:
            json.dump(flat_rows, f, indent=2)

        if len(flat_rows) > 0:
            metrics_path_csv = os.path.join(self._output_dir, "inference_metrics.csv")
            fieldnames = sorted({k for r in flat_rows for k in r.keys()})
            with open(metrics_path_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_rows)

            rmsds = [
                float(r["rmsd_c4"])
                for r in flat_rows
                if r.get("rmsd_c4") is not None and not np.isnan(r["rmsd_c4"])
            ]
            if len(rmsds) > 0:
                log.info(f"Mean RMSD(C4') over {len(rmsds)} samples: {np.mean(rmsds):.4f}")

        log.info(f"Finished in {elapsed_time:.2f}s")
        log.info(
            f"Generated samples are stored here: {self._output_dir}"
        )
        
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
