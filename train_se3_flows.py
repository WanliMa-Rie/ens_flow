"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/experiments/train_se3_flows.py
"""

import os
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger

from rna_backbone_design.data.rna_conformer_datamodule import RNAConformerDataModule
from rna_backbone_design.models.flow_module import FlowModule
import rna_backbone_design.utils as eu
import wandb

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision("high")
if hasattr(torch.autograd, "graph"):
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


class Experiment:
    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data_cfg
        self._exp_cfg = cfg.experiment
        self._model = FlowModule(self._cfg)
        self._datamodule = RNAConformerDataModule(data_cfg=self._data_cfg)

    def train(self):
        ckpt_dir = None

        logger = WandbLogger(**self._exp_cfg.wandb,)

        # Checkpoint directory
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Checkpoints saved to {ckpt_dir}")


        # Save config
        cfg_path = os.path.join(ckpt_dir, "config.yaml")
        with open(cfg_path, "w") as f:
            OmegaConf.save(config=self._cfg, f=f.name)
        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        flat_cfg = dict(eu.flatten_dict(cfg_dict))
        if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
            logger.experiment.config.update(flat_cfg)

        # Define trainer
        trainer = Trainer(
            **self._exp_cfg.trainer,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices="auto",
        )

        # Train
        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
        )


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    exp = Experiment(cfg=cfg)
    exp.train()


if __name__ == "__main__":
    main()
