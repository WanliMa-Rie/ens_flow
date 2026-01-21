"""
Visualize sampling trajectories for a single RNA cluster.

uv run python visualize_cluster_inference.py \
  inference.cluster_id=cluster_0409_6sv4_BS \
  inference.ckpt_path=ckpt/se3-fm/rna-frameflow_cond_ensemble/last.ckpt \
  inference.output_dir=generated_rna_bb_samples/rna-frameflow_cond_ensemble2 \
  inference.name=generated_samples \
  inference.interpolant.sampling.num_timesteps=60
"""

import os
import json
import pathlib
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import rna_backbone_design.utils as eu
from rna_backbone_design.analysis import utils as au
from rna_backbone_design.data import utils as du
from rna_backbone_design.data.interpolant import Interpolant
from rna_backbone_design.data.rna_cluster_dataset import RNAClusterDataset
from rna_backbone_design.data.rna_cluster_datamodule import length_batching_collate
from rna_backbone_design.models.flow_module import FlowModule

torch.set_float32_matmul_precision("high")
log = eu.get_pylogger(__name__)


def _select_step_indices(num_timesteps: int) -> List[int]:
    if num_timesteps <= 0:
        return [0]
    idx = np.linspace(0, num_timesteps, 6)
    idx = [int(round(x)) for x in idx]
    out = []
    for x in idx:
        x = max(0, min(num_timesteps, x))
        if len(out) == 0 or out[-1] != x:
            out.append(x)
    return out


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def run(cfg: DictConfig) -> None:
    if not hasattr(cfg.inference, "cluster_id"):
        raise ValueError("Please provide inference.cluster_id=cluster_XXXX_YYYY_Z")

    num_vis_samples = 3
    cluster_id = str(cfg.inference.cluster_id)

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
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config_flashipa.yaml")

    if os.path.exists(config_path):
        ckpt_cfg = OmegaConf.load(config_path)
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(ckpt_cfg, cfg)

    device = torch.device(
        "cuda:0" if bool(cfg.inference.use_gpu) and torch.cuda.is_available() else "cpu"
    )

    output_dir = os.path.join(cfg.inference.output_dir, cfg.inference.name, cluster_id)
    sampled_dir = os.path.join(output_dir, "sampled")
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(sampled_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    with open(os.path.join(output_dir, "inference_config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    flow_module = FlowModule.load_from_checkpoint(
        checkpoint_path=ckpt_path, cfg=cfg, map_location="cpu", weights_only=False
    )
    flow_module.eval()
    flow_module.to(device)

    flow_module._infer_cfg = cfg.inference
    flow_module._output_dir = os.path.join(cfg.inference.output_dir, cfg.inference.name)
    flow_module._save_gt_pdbs(cluster_id, gt_dir)

    dataset = RNAClusterDataset(
        data_dir=cfg.data_cfg.data_dir,
        split="val",
        max_length=cfg.data_cfg.get("max_len", None),
        return_ensemble=bool(cfg.data_cfg.get("return_ensemble", False)),
        max_ensemble_conformers=cfg.data_cfg.get("max_ensemble_conformers", None),
    )
    cluster_dir = os.path.join(str(cfg.data_cfg.data_dir), cluster_id)
    if not os.path.isdir(cluster_dir):
        raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")
    dataset.clusters = [pathlib.Path(cluster_dir)]
    item = dataset[0]
    batch = length_batching_collate([item])

    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    interpolant = Interpolant(cfg.inference.interpolant)
    interpolant.set_device(device)

    sample_length = int(batch["res_mask"][0].sum().item())
    context = {
        "single_embedding": batch["single_embedding"][:, :sample_length],
        "pair_embedding": batch["pair_embedding"][:, :sample_length, :sample_length],
    }

    num_timesteps = int(cfg.inference.interpolant.sampling.num_timesteps)
    step_indices = _select_step_indices(num_timesteps)
    meta = {
        "cluster_id": cluster_id,
        "pdb_name": batch["pdb_name"][0],
        "sample_length": sample_length,
        "num_timesteps": num_timesteps,
        "selected_step_indices": step_indices,
        "num_vis_samples": num_vis_samples,
    }
    with open(os.path.join(output_dir, "visualization_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for s in range(num_vis_samples):
        sample_name = f"{batch['pdb_name'][0]}_vis{s}"
        sample_out_dir = os.path.join(sampled_dir, sample_name)
        os.makedirs(sample_out_dir, exist_ok=True)

        atom37_traj, _, _ = interpolant.sample(
            1, sample_length, flow_module.model, context=context
        )
        traj = torch.concat(atom37_traj, dim=0)

        for t_idx in step_indices:
            if traj.ndim == 4:
                pos_t = traj[t_idx]
            elif traj.ndim == 5:
                pos_t = traj[t_idx, 0]
            else:
                raise ValueError(f"Unexpected trajectory shape: {tuple(traj.shape)}")
            pos = du.to_numpy(pos_t)
            au.write_complex_to_pdbs(
                pos,
                os.path.join(sample_out_dir, f"step_{t_idx:03d}"),
                is_na_residue_mask=np.ones(sample_length, dtype=np.int64),
            )

    log.info(f"Saved visualization outputs to {output_dir}")


if __name__ == "__main__":
    run()
