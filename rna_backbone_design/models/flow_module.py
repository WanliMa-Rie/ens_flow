"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/flow_module.py
"""

import torch
import time
import os
import glob
import shutil
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule

from rna_backbone_design.analysis import metrics
from rna_backbone_design.analysis.ensemble_metrics import compute_ensemble_metrics
from rna_backbone_design.analysis import utils as au
from rna_backbone_design.models.flow_model import FlowModel
from rna_backbone_design.models import utils as mu
from rna_backbone_design.data.interpolant import Interpolant
from rna_backbone_design.data import utils as du
from rna_backbone_design.data import all_atom as rna_all_atom
from rna_backbone_design.data import so3_utils
from rna_backbone_design.data import nucleotide_constants
from rna_backbone_design.analysis import utils as au
from pytorch_lightning.loggers.wandb import WandbLogger


class FlowModule(LightningModule):
    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics_by_loader = {"ensemble": [], "single": []}
        self.validation_epoch_samples = []
        self.save_hyperparameters()

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            "train/epoch_time_minutes",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch):
        """
        Given a batch of corrupted Frame objects, predicts the vector field, and calculate the loss between the predicted one and the ground truth.
        Params:
            noisy_batch (dict) : dictionary of tensors corresponding to corrupted Frame objects

        Remarks:
            Computes the different core and auxiliary losses between ground truth and predicted backbones

        Returns:
            Dictionary of core and auxiliary losses
        """
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch["res_mask"]
        is_na_residue_mask = noisy_batch["is_na_residue_mask"]

        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch["res_plddt"] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask

        num_batch, num_res = loss_mask.shape

        torsions_start_index = 0
        torsions_end_index = 8
        num_torsions = torsions_end_index - torsions_start_index

        if training_cfg.num_non_frame_atoms == 0:
            bb_filtered_atom_idx = [2, 3, 6]  # [C3', C4', O4']
        elif training_cfg.num_non_frame_atoms == 3:
            bb_filtered_atom_idx = [2, 3, 6] + [
                0,
                7,
                9,
            ]  # [C3', C4', O4'] + [C1', O3', P]
        elif training_cfg.num_non_frame_atoms == 7:
            bb_filtered_atom_idx = [2, 3, 6] + [
                0,
                4,
                7,
                9,
                10,
                11,
                12,
            ]  # [C3', C4', O4'] + [C1', C5', O3', P, OP1, OP2, N1]
        else:
            # NOTE: default is the original frame
            bb_filtered_atom_idx = [2, 3, 6]  # [C3', C4', O4']

        n_merged_atoms = len(bb_filtered_atom_idx)

        # Ground truth labels
        gt_trans_1 = noisy_batch["trans_1"]
        gt_rotmats_1 = noisy_batch["rotmats_1"]
        gt_torsions_1 = noisy_batch["torsion_angles_sin_cos"][
            :, :, torsions_start_index:torsions_end_index, :
        ].reshape(num_batch, num_res, num_torsions * 2)
        rotmats_t = noisy_batch["rotmats_t"]
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))

        gt_bb_atoms = rna_all_atom.to_atom37_rna(
            gt_trans_1,
            gt_rotmats_1,
            torch.ones_like(is_na_residue_mask),
            torsions=gt_torsions_1,
        )
        gt_bb_atoms = gt_bb_atoms[:, :, bb_filtered_atom_idx]

        # Timestep used for normalization.
        t = noisy_batch["t"]
        norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(training_cfg.t_normalize_clip)
        )

        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output["pred_trans"]
        pred_rotmats_1 = model_output["pred_rotmats"]
        pred_torsions_1 = model_output["pred_torsions"].reshape(
            num_batch, num_res, num_torsions * 2
        )
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        pred_bb_atoms = rna_all_atom.to_atom37_rna(
            pred_trans_1,
            pred_rotmats_1,
            torch.ones_like(is_na_residue_mask),
            torsions=pred_torsions_1,
        )
        pred_bb_atoms = pred_bb_atoms[:, :, bb_filtered_atom_idx]

        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * n_merged_atoms
        bb_atom_loss = (
            torch.sum(
                (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
                dim=(-1, -2, -3),
            )
            / loss_denom
        )

        # Translation VF loss
        trans_error = (
            (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 frame atoms
        trans_loss = (
            training_cfg.translation_loss_weight
            * torch.sum(trans_error**2 * loss_mask[..., None], dim=(-1, -2))
            / loss_denom
        )

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 frame atoms
        rots_vf_loss = (
            training_cfg.rotation_loss_weights
            * torch.sum(rots_vf_error**2 * loss_mask[..., None], dim=(-1, -2))
            / loss_denom
        )

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * n_merged_atoms, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_bb_atoms.reshape(
            [num_batch, num_res * n_merged_atoms, 3]
        )
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_merged_atoms))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * n_merged_atoms])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_merged_atoms))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * n_merged_atoms])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2)) - num_res

        # Torsion angles loss
        pred_torsions_1 = pred_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        gt_torsions_1 = gt_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        loss_denom = torch.sum(loss_mask, dim=-1) * 8  # 8 torsion angles
        tors_loss = (
            training_cfg.tors_loss_scale
            * torch.sum(
                torch.linalg.norm(pred_torsions_1 - gt_torsions_1, dim=-1) ** 2
                * loss_mask[..., None],
                dim=(-1, -2),
            )
            / loss_denom
        )

        assert bb_atom_loss.shape[0] == dist_mat_loss.shape[0] == tors_loss.shape[0], (
            f"Loss tensors shape mismatch: {bb_atom_loss.shape} vs {dist_mat_loss.shape} vs {tors_loss.shape}"
        )

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss + tors_loss) * (
            t[:, 0] > training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight

        if torch.isnan(auxiliary_loss).any():
            print("NaN loss in aux_loss")
            auxiliary_loss = torch.zeros_like(auxiliary_loss).to(se3_vf_loss.device)

        if torch.isnan(se3_vf_loss).any():
            # raise ValueError('NaN loss encountered')
            print("NaN loss in se3_vf_loss")
            se3_vf_loss = torch.zeros_like(se3_vf_loss).to(se3_vf_loss.device)

        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "torsion_loss": tors_loss,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        self.interpolant.set_device(batch["res_mask"].device)
        loader_name = "ensemble" if int(dataloader_idx) == 0 else "single"
        res_mask = batch["res_mask"]
        mask_bool = res_mask > 0.5
        num_batch, num_res = res_mask.shape

        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch["trans_sc"] = model_sc["pred_trans"]

        batch_losses = self.model_step(noisy_batch)
        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}
        val_loss = total_losses[self._exp_cfg.training.loss] + total_losses["auxiliary_loss"]

        self._log_scalar(
            f"valid/{loader_name}_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=num_batch,
        )

        with torch.no_grad():
            model_out = self.model(noisy_batch)

        torsions_start_index = 0
        torsions_end_index = 8
        num_torsions = torsions_end_index - torsions_start_index

        gt_trans_1 = batch["trans_1"]
        gt_rotmats_1 = batch["rotmats_1"]
        gt_torsions_1 = batch["torsion_angles_sin_cos"][
            :, :, torsions_start_index:torsions_end_index, :
        ].reshape(num_batch, num_res, num_torsions * 2)

        pred_trans_1 = model_out["pred_trans"]
        pred_rotmats_1 = model_out["pred_rotmats"]
        pred_torsions_1 = model_out["pred_torsions"].reshape(
            num_batch, num_res, num_torsions * 2
        )

        is_na_residue_mask = batch["is_na_residue_mask"].bool()
        gt_atoms = rna_all_atom.to_atom37_rna(
            gt_trans_1, gt_rotmats_1, is_na_residue_mask, torsions=gt_torsions_1
        )
        pred_atoms = rna_all_atom.to_atom37_rna(
            pred_trans_1,
            pred_rotmats_1,
            is_na_residue_mask,
            torsions=pred_torsions_1,
        )

        c4_idx = nucleotide_constants.atom_order["C4'"]
        gt_c4 = gt_atoms[:, :, c4_idx]
        pred_c4 = pred_atoms[:, :, c4_idx]

        valid_align = mask_bool.sum(dim=1) >= 3
        aligned_pred_c4 = pred_c4
        if valid_align.any():
            idx = torch.nonzero(valid_align, as_tuple=False).squeeze(-1)
            try:
                aligned_subset = metrics.superimpose(
                    gt_c4[idx], pred_c4[idx], mask=mask_bool[idx]
                )
                aligned_pred_c4 = pred_c4.clone()
                aligned_pred_c4[idx] = aligned_subset
            except RuntimeError:
                aligned_pred_c4 = pred_c4

        batch_metrics = []
        pred_atoms_np = pred_atoms.detach().cpu().numpy()
        gt_atoms_np = gt_atoms.detach().cpu().numpy()
        mask_np = mask_bool.detach().cpu().numpy()

        ensemble_cfg = getattr(self._exp_cfg, "ensemble_metrics", None)
        compute_ensemble = False
        max_batches = 0
        num_generated = 0
        num_gt = 0
        if ensemble_cfg is not None:
            compute_ensemble = bool(getattr(ensemble_cfg, "enabled", False))
            max_batches = int(getattr(ensemble_cfg, "max_batches", 0))
            num_generated = int(getattr(ensemble_cfg, "num_generated", 0))
            num_gt = int(getattr(ensemble_cfg, "num_gt", 0))

        rmsd_c4 = None
        if loader_name == "single":
            rmsd_c4 = metrics.rmsd(gt_c4, aligned_pred_c4, mask=mask_bool.float())
            rmsd_c4 = torch.where(
                valid_align, rmsd_c4, torch.full_like(rmsd_c4, torch.nan)
            )

        for i in range(num_batch):
            valid_len = int(mask_np[i].sum())

            c4_metrics = {}
            if loader_name == "single":
                assert rmsd_c4 is not None
                c4_metrics["rmsd_c4"] = float(rmsd_c4.detach().cpu().numpy()[i])

            gt_ens_list = batch.get("gt_c4_ensemble", None)
            gt_ens = None
            if isinstance(gt_ens_list, list):
                gt_ens = gt_ens_list[i]

            if (
                loader_name == "ensemble"
                and compute_ensemble
                and batch_idx < max_batches
                and gt_ens is not None
                and isinstance(gt_ens, torch.Tensor)
            ):
                eff_len = min(valid_len, int(gt_ens.shape[1]))
                k_gt = min(num_gt, int(gt_ens.shape[0]), num_generated)
                if eff_len >= 3 and k_gt >= 2 and num_generated >= 2:
                    context = {
                        "single_embedding": batch["single_embedding"][
                            i : i + 1, :eff_len
                        ].repeat(num_generated, 1, 1),
                        "pair_embedding": batch["pair_embedding"][
                            i : i + 1, :eff_len, :eff_len
                        ].repeat(num_generated, 1, 1, 1),
                    }
                    atom37_traj, _, _ = self.interpolant.sample(
                        num_generated, eff_len, self.model, context=context
                    )
                    pred_ens = atom37_traj[-1][:, :eff_len, c4_idx]
                    gt_ens_i = gt_ens[:k_gt, :eff_len].to(pred_ens.device)
                    pred_ens_i = pred_ens[:k_gt]
                    m = mask_bool[i, :eff_len].to(pred_ens.device)
                    ens = compute_ensemble_metrics(gt_ens_i, pred_ens_i, m)
                    c4_metrics.update(
                        {
                            "ens_pairwise_rmsd": ens.pairwise_rmsd,
                            "ens_w2": ens.w2_distance,
                            "ens_pairwise_rmsd_r": ens.pairwise_rmsd_r,
                        }
                    )
            batch_metrics.append(c4_metrics)


            if loader_name == "single" and batch_idx == 0 and i == 0:
                # Perform actual generative sampling (from noise) for visualization
                # This uses the embeddings from the validation set but generates structure from scratch
                context = {
                    "single_embedding": batch["single_embedding"][i : i + 1, :valid_len],
                    "pair_embedding": batch["pair_embedding"][i : i + 1, :valid_len, :valid_len],
                }
                # Sample returns list of [B, N, 37, 3] tensors
                atom37_traj, _, _ = self.interpolant.sample(
                    1, valid_len, self.model, context=context
                )
                # Take the last step (final generated structure) and the first batch item
                generated_sample = atom37_traj[-1][0].detach().cpu().numpy()

                restype = None
                if "aatype" in batch:
                    restype = batch["aatype"][i, :valid_len].detach().cpu().numpy()
                
                # Save generated sample with unique name including epoch/step
                filename = f"generated_epoch_{self.current_epoch}_step_{self.global_step}_len_{valid_len}.pdb"
                saved_rna_path = au.write_complex_to_pdbs(
                    generated_sample,
                    os.path.join(
                        self._sample_write_dir,
                        filename,
                    ),
                    is_na_residue_mask=mask_np[i, :valid_len],
                    restype=restype,
                )
                if isinstance(self.logger, WandbLogger):
                    self.validation_epoch_samples.append(
                        [saved_rna_path, self.global_step, wandb.Molecule(saved_rna_path)]
                    )

        self.validation_epoch_metrics_by_loader[loader_name].append(
            pd.DataFrame(batch_metrics)
        )

    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key="valid/samples",
                columns=["sample_path", "global_step", "RNA"],
                data=self.validation_epoch_samples,
            )
            self.validation_epoch_samples.clear()

        for loader_name, dfs in self.validation_epoch_metrics_by_loader.items():
            if len(dfs) == 0:
                continue
            val_epoch_metrics = pd.concat(dfs)
            for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
                self._log_scalar(
                    f"valid/{metric_name}",
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
            self.validation_epoch_metrics_by_loader[loader_name].clear()

    def _log_scalar(
        self,
        key,
        value,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        batch_size=None,
        sync_dist=False,
        rank_zero_only=True,
    ):
        if sync_dist and rank_zero_only:
            raise ValueError("Unable to sync dist when rank_zero_only=True")
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
        )

    def training_step(self, batch, batch_idx):
        """
        Performs one iteration of SE(3) flow matching and returns total training loss
        using the core and auxiliary losses computed in `model_step`.
        """
        step_start_time = time.time()
        self.interpolant.set_device(batch["res_mask"].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)

        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch["trans_sc"] = model_sc["pred_trans"]

        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses["bb_atom_loss"].shape[0]
        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}

        for k, v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch["t"])
        self._log_scalar(
            "train/t", np.mean(du.to_numpy(t)), prog_bar=False, batch_size=num_batch
        )
        for loss_name, loss_dict in batch_losses.items():
            stratified_losses = mu.t_stratified_loss(t, loss_dict, loss_name=loss_name)
            for k, v in stratified_losses.items():
                self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/length",
            batch["res_mask"].shape[1],
            prog_bar=False,
            batch_size=num_batch,
        )
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)

        step_time = time.time() - step_start_time
        self._log_scalar("train/eps", num_batch / step_time)

        train_loss = (
            total_losses[self._exp_cfg.training.loss] + total_losses["auxiliary_loss"]
        )
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)

        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(), **self._exp_cfg.optimizer
        )

    def _save_gt_pdbs(self, cluster_id, gt_dir):
        cfg = None
        if hasattr(self, "hparams"):
            try:
                if "cfg" in self.hparams:
                    cfg = self.hparams["cfg"]
            except Exception:
                cfg = None
            if cfg is None:
                cfg = getattr(self.hparams, "cfg", None)

        data_dir = None
        if cfg is not None:
            try:
                data_dir = cfg.data_cfg.data_dir
            except Exception:
                try:
                    data_dir = cfg["data_cfg"]["data_dir"]
                except Exception:
                    data_dir = None

        if not data_dir:
            self._print_logger.warning(f"Missing data_cfg.data_dir; skipping GT for {cluster_id}")
            return

        cluster_dir = os.path.join(str(data_dir), cluster_id)
        structure_dir = os.path.join(cluster_dir, "structure")

        pdb_paths = []
        if os.path.isdir(structure_dir):
            pdb_paths = sorted(glob.glob(os.path.join(structure_dir, "*.pdb")))

        if pdb_paths:
            for src in pdb_paths:
                dst = os.path.join(gt_dir, os.path.basename(src))
                if os.path.exists(dst):
                    continue
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    self._print_logger.warning(f"Failed to copy GT PDB {src}: {e}")
            return

        feature_dir = os.path.join(cluster_dir, "features")
        if not os.path.isdir(feature_dir):
            self._print_logger.warning(f"Missing structure/features for {cluster_id}; skipping GT")
            return

        pkl_paths = sorted(glob.glob(os.path.join(feature_dir, "*.pkl")))
        for pkl_path in pkl_paths:
            name = os.path.splitext(os.path.basename(pkl_path))[0]
            expected = os.path.join(gt_dir, f"na_{name}.pdb")
            if os.path.exists(expected):
                continue
            try:
                feats = du.read_pkl(pkl_path)
                feats = du.parse_complex_feats(feats)
                pos = feats["atom_positions"]
                if pos.ndim == 3:
                    n_res, n_atoms, _ = pos.shape
                    if n_atoms < 37:
                        pos = np.pad(pos, ((0, 0), (0, 37 - n_atoms), (0, 0)))
                au.write_complex_to_pdbs(
                    pos,
                    os.path.join(gt_dir, name),
                    is_na_residue_mask=np.ones(pos.shape[0], dtype=np.int64),
                )
            except Exception as e:
                self._print_logger.warning(f"Failed to write GT from {pkl_path}: {e}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        device = batch["res_mask"].device
        interpolant = Interpolant(self._infer_cfg.interpolant)
        interpolant.set_device(device)

        res_mask = batch["res_mask"]
        num_batch = res_mask.shape[0]
        mask_bool = res_mask > 0.5

        gt_trans_1 = batch["trans_1"]
        gt_rotmats_1 = batch["rotmats_1"]
        gt_torsions_1 = batch["torsion_angles_sin_cos"][:, :, :8, :].reshape(
            num_batch, -1, 16
        )
        is_na_residue_mask = batch["is_na_residue_mask"].bool()
        gt_atoms = rna_all_atom.to_atom37_rna(
            gt_trans_1, gt_rotmats_1, is_na_residue_mask, torsions=gt_torsions_1
        )
        c4_idx = nucleotide_constants.atom_order["C4'"]
        gt_c4 = gt_atoms[:, :, c4_idx]

        pdb_names = batch.get("pdb_name", None)
        if pdb_names is None:
            pdb_names = [None] * num_batch
        cluster_ids = batch.get("cluster_id", None)
        if cluster_ids is None:
            cluster_ids = [None] * num_batch

        outputs = []

        ensemble_cfg = getattr(self._infer_cfg, "ensemble", None)
        ensemble_enabled = False
        ensemble_num_generated = 0
        ensemble_write_pdbs = False
        if ensemble_cfg is not None:
            ensemble_enabled = bool(getattr(ensemble_cfg, "enabled", False))
            ensemble_num_generated = int(getattr(ensemble_cfg, "num_generated", 0))
            ensemble_write_pdbs = bool(getattr(ensemble_cfg, "write_ensemble_pdbs", False))

        for i in range(num_batch):
            sample_length = int(res_mask[i].sum().item())
            if sample_length < 1:
                continue

            cluster_id = (
                cluster_ids[i]
                if isinstance(cluster_ids, list) and cluster_ids[i] is not None
                else "unknown_cluster"
            )
            
            if ensemble_enabled:
                base_dir = os.path.join(self._output_dir, cluster_id)
                sample_dir = os.path.join(base_dir, "sampled")
                gt_dir = os.path.join(base_dir, "gt")
                os.makedirs(sample_dir, exist_ok=True)
                os.makedirs(gt_dir, exist_ok=True)
                self._save_gt_pdbs(cluster_id, gt_dir)
            else:
                sample_dir = os.path.join(self._output_dir, cluster_id, f"length_{sample_length}")
                os.makedirs(sample_dir, exist_ok=True)

            context = {
                "single_embedding": batch["single_embedding"][i : i + 1, :sample_length],
                "pair_embedding": batch["pair_embedding"][
                    i : i + 1, :sample_length, :sample_length
                ],
            }

            atom37_traj, _, _ = interpolant.sample(
                1, sample_length, self.model, context=context
            )
            bb_traj = du.to_numpy(torch.concat(atom37_traj, dim=0))
            sample = bb_traj[-1]

            pdb_name = pdb_names[i] if isinstance(pdb_names, list) else None
            sample_name = pdb_name if pdb_name else f"sample_{batch_idx}_{i}"
            is_na_residue_mask = np.ones(sample_length, dtype=np.int64)

            au.write_complex_to_pdbs(
                sample,
                os.path.join(sample_dir, f"sample_{sample_name}"),
                is_na_residue_mask=is_na_residue_mask,
            )

            au.write_complex_to_pdbs(
                bb_traj,
                os.path.join(sample_dir, f"sample_{sample_name}"),
                is_na_residue_mask=is_na_residue_mask,
            )

            pred_c4 = atom37_traj[-1][0, :sample_length, c4_idx]
            gt_c4_i = gt_c4[i : i + 1, :sample_length]
            mask_i = mask_bool[i : i + 1, :sample_length]
            valid_align = bool(mask_i.sum().item() >= 3)
            rmsd_c4 = float("nan")
            if valid_align:
                try:
                    aligned_pred_c4 = metrics.superimpose(
                        gt_c4_i, pred_c4[None, ...], mask=mask_i
                    )
                    rmsd_c4 = float(
                        metrics.rmsd(gt_c4_i, aligned_pred_c4, mask=mask_i.float())
                        .detach()
                        .cpu()
                        .numpy()[0]
                    )
                except RuntimeError:
                    rmsd_c4 = float("nan")

            out_row = {
                "cluster_id": cluster_id,
                "pdb_name": pdb_name,
                "sample_name": sample_name,
                "length": sample_length,
                "rmsd_c4": rmsd_c4,
                "sample_dir": sample_dir,
            }

            gt_ens_list = batch.get("gt_c4_ensemble", None)
            gt_ens = None
            if isinstance(gt_ens_list, list):
                gt_ens = gt_ens_list[i]

            if (
                ensemble_enabled
                and gt_ens is not None
                and isinstance(gt_ens, torch.Tensor)
                and ensemble_num_generated >= 2
                and int(gt_ens.shape[0]) >= 2
            ):
                eff_len = min(sample_length, int(gt_ens.shape[1]))
                k = min(ensemble_num_generated, int(gt_ens.shape[0]))
                if eff_len >= 3 and k >= 2:
                    context_ens = {
                        "single_embedding": batch["single_embedding"][
                            i : i + 1, :eff_len
                        ].repeat(ensemble_num_generated, 1, 1),
                        "pair_embedding": batch["pair_embedding"][
                            i : i + 1, :eff_len, :eff_len
                        ].repeat(ensemble_num_generated, 1, 1, 1),
                    }
                    atom37_traj_ens, _, _ = interpolant.sample(
                        ensemble_num_generated, eff_len, self.model, context=context_ens
                    )
                    pred_ens = atom37_traj_ens[-1][:, :eff_len, c4_idx]
                    gt_ens_i = gt_ens[:k, :eff_len].to(pred_ens.device)
                    pred_ens_i = pred_ens[:k]
                    m = mask_bool[i, :eff_len].to(pred_ens.device)
                    ens = compute_ensemble_metrics(gt_ens_i, pred_ens_i, m)
                    out_row.update(
                        {
                            "ens_pairwise_rmsd": ens.pairwise_rmsd,
                            "ens_w2": ens.w2_distance,
                            "ens_pairwise_rmsd_r": ens.pairwise_rmsd_r,
                            "ens_k": k,
                            "ens_num_generated": ensemble_num_generated,
                            "ens_num_gt_available": int(gt_ens.shape[0]),
                        }
                    )

                    if ensemble_write_pdbs:
                        for j in range(ensemble_num_generated):
                            sample_j = du.to_numpy(atom37_traj_ens[-1][j : j + 1])[0]
                            au.write_complex_to_pdbs(
                                sample_j,
                                os.path.join(sample_dir, f"sample_{sample_name}_ens{j}"),
                                is_na_residue_mask=np.ones(eff_len, dtype=np.int64),
                            )

            outputs.append(
                {
                    **out_row,
                }
            )

        return outputs
