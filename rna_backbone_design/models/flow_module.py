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
        self._validation = self._exp_cfg.validation
        self._model_cfg = cfg.model
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        # U-STEER: optional stochastic diffusion amplitude
        self._steer_cfg = getattr(cfg, 'steer', None)
        self._steer_enabled = self._steer_cfg is not None and self._steer_cfg.enabled
        if self._steer_enabled:
            from rna_backbone_design.models.amplitude_net import AmplitudeNet
            self.amplitude_net = AmplitudeNet(**self._steer_cfg.amplitude_net)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics_by_loader = {"ensemble": [], "single": []}
        self.validation_epoch_samples = []
        self.save_hyperparameters()

    def on_fit_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.define_metric("epoch")
            self.logger.experiment.define_metric("train/*", step_metric="epoch")
            self.logger.experiment.define_metric("valid/*", step_metric="epoch")

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        self._log_scalar(
            "epoch",
            float(self.current_epoch),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        # epoch_time = (time.time() - self._epoch_start_time) / 60.0
        # self.log(
        #     "train/epoch_time_minutes",
        #     epoch_time,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        # )
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
        """
        res_mask: [B, N]. residue mask, 0/1
        is_na_residue_mask: [B, N]. nucletide/non-nucleotide residue mask, 0/1
        num_res: int. number of nucletides in the sample (without padding)
        """


        self.interpolant.set_device(batch["res_mask"].device)
        loader_name = "ensemble" if int(dataloader_idx) == 0 else "single"

        num_batch = batch["is_na_residue_mask"].shape[0]
        num_generated = int(getattr(self._validation, "num_generated", 10))
        c4_idx = nucleotide_constants.atom_order["C4'"]

        batch_metrics = []

        for i in range(num_batch):
            sample = batch[i]
            mask = sample["is_na_residue_mask"]
            num_res = int(mask.sum().item())

            context = {
                "single_embedding": sample["single_embedding"].repeat(num_generated, 1, 1),
                "pair_embedding": sample["pair_embedding"].repeat(num_generated, 1, 1, 1),
            }

            if self._steer_enabled:
                b_norm = du.normalize_b_factors(
                    sample["b_factors"].unsqueeze(0).expand(num_generated, -1),
                    mask.unsqueeze(0).expand(num_generated, -1),
                )
                atom37_traj, _, _, _qv = self.interpolant.sample_stochastic(
                    num_generated, num_res, self.model, self.amplitude_net,
                    b_norm, context=context,
                )
            else:
                atom37_traj, _, _ = self.interpolant.sample(
                    num_generated, num_res, self.model, context=context
                )
            pred_c4 = atom37_traj[-1][:, :, c4_idx]

            sample_metrics = {}

            if loader_name == "single":
                gt_c4 = sample["c4_coords"]
                single_result = metrics.compute_single_metrics(
                    pred_c4, gt_c4, mask
                )
                sample_metrics["single_rmsd"] = float(single_result["rmsd"])
                sample_metrics["single_tm_score"] = float(single_result["tm_score"])

            elif loader_name == "ensemble":
                gt_c4_ens = sample["gt_c4_ensemble"]
                ensemble_result = metrics.compute_ensemble_metrics(pred_c4, gt_c4_ens, mask)
                sample_metrics.update(ensemble_result)

            batch_metrics.append(sample_metrics)

        self.validation_epoch_metrics_by_loader[loader_name].append(
            pd.DataFrame(batch_metrics)
        )

    def on_validation_epoch_end(self):
        for loader_name, dfs in self.validation_epoch_metrics_by_loader.items():
            if len(dfs) == 0:
                continue
            val_epoch_metrics = pd.concat(dfs)
            for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
                self._log_scalar(
                    f"valid/{loader_name}_{metric_name}",
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
        if on_epoch and not on_step:
            sync_dist = True
            rank_zero_only = False
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

    def training_step(self, batch):
        """
        Performs one iteration of SE(3) flow matching and returns total training loss
        using the core and auxiliary losses computed in `model_step`.
        """
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
            self._log_scalar(
                f"train/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=num_batch,
            )

        # # Losses to track. Stratified across t.
        # t = torch.squeeze(noisy_batch["t"])
        # self._log_scalar(
        #     "train/t", np.mean(du.to_numpy(t)), prog_bar=False, batch_size=num_batch
        # )
        # for loss_name, loss_dict in batch_losses.items():
        #     stratified_losses = mu.t_stratified_loss(t, loss_dict, loss_name=loss_name)
        #     for k, v in stratified_losses.items():
        #         self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # # Training throughput
        # self._log_scalar(
        #     "train/length",
        #     batch["res_mask"].shape[1],
        #     prog_bar=False,
        #     batch_size=num_batch,
        # )
        # self._log_scalar("train/batch_size", num_batch, prog_bar=False)
        #
        # step_time = time.time() - step_start_time
        # self._log_scalar("train/eps", num_batch / step_time)

        train_loss = (
            total_losses[self._exp_cfg.training.loss] + total_losses["auxiliary_loss"]
        )

        # U-STEER diffusion losses (SE3L_diffusion_losses_derivation.md)
        if self._steer_enabled:
            b_norm = du.normalize_b_factors(batch["b_factors"], batch["res_mask"])
            mask = batch["res_mask"]

            # L_qv: quadratic-variation calibration loss (Section 3)
            # Q_i = E[∫ Tr(D_{x,i}) dt], single-step MC at uniformly sampled t
            a_trans, _a_rots = self.amplitude_net(b_norm, noisy_batch["t"])
            b_hat = du.normalize_b_factors(a_trans ** 2, mask)
            qv_loss = ((b_hat - b_norm) ** 2 * mask).sum(dim=-1) / mask.sum(dim=-1)
            qv_loss = qv_loss.mean()
            train_loss = train_loss + self._steer_cfg.loss.qv_weight * qv_loss
            self._log_scalar(
                "train/qv_loss", qv_loss,
                on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch,
            )

            # L_cov: terminal empirical covariance calibration loss (Section 6)
            # M rollouts → empirical Tr(Ĉ_i) → normalize → compare to B̃
            cov_weight = self._steer_cfg.loss.cov_weight
            if cov_weight > 0:
                M = self._steer_cfg.loss.cov_num_rollouts
                # Use first sample in the batch, expand for M rollouts
                idx = 0
                num_res = int(mask[idx].sum().item())
                sample_b_norm = b_norm[idx:idx+1].expand(M, -1)
                sample_mask = mask[idx:idx+1].expand(M, -1)
                context_cov = {
                    "single_embedding": batch["single_embedding"][idx:idx+1].expand(M, -1, -1),
                    "pair_embedding": batch["pair_embedding"][idx:idx+1].expand(M, -1, -1, -1),
                }
                terminal_trans = self.interpolant.rollout_terminal_trans(
                    M, num_res, self.model, self.amplitude_net,
                    sample_b_norm, context=context_cov,
                )  # [M, num_res, 3]
                # ŝ_i² = Tr(Ĉ_i) = Σ_d Var_m(x^{(m,i)}_d)
                mean_trans = terminal_trans.mean(dim=0, keepdim=True)
                s_sq = ((terminal_trans - mean_trans) ** 2).mean(dim=0).sum(dim=-1)  # [num_res]
                c_hat = du.normalize_b_factors(s_sq.unsqueeze(0), sample_mask[0:1])
                target_b = b_norm[idx:idx+1]
                cov_loss = ((c_hat - target_b) ** 2 * sample_mask[0:1]).sum() / sample_mask[0:1].sum()
                train_loss = train_loss + cov_weight * cov_loss
                self._log_scalar(
                    "train/cov_loss", cov_loss,
                    on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch,
                )

        self.log(
            "pbar/loss",
            train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
            batch_size=num_batch,
            rank_zero_only=True,
        )
        self._log_scalar(
            "train/loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=num_batch,
        )

        return train_loss

    def configure_optimizers(self):
        if not self._steer_enabled:
            return torch.optim.AdamW(
                params=self.model.parameters(), **self._exp_cfg.optimizer
            )
        drift_lr = self._exp_cfg.optimizer.lr * self._steer_cfg.drift_lr_scale
        param_groups = [
            {"params": self.model.parameters(), "lr": drift_lr},
            {"params": self.amplitude_net.parameters(), "lr": self._steer_cfg.optimizer.lr},
        ]
        return torch.optim.AdamW(param_groups)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        interpolant = Interpolant(self._infer_cfg.interpolant)
        interpolant.set_device(batch["res_mask"].device)

        num_batch = batch["is_na_residue_mask"].shape[0]
        c4_idx = nucleotide_constants.atom_order["C4'"]

        ensemble_cfg = getattr(self._infer_cfg, "ensemble", None)
        ensemble_enabled = bool(getattr(ensemble_cfg, "enabled", False)) if ensemble_cfg else False
        num_generated = int(getattr(ensemble_cfg, "num_generated", 1)) if ensemble_enabled else 1
        num_generated = max(1, num_generated)

        outputs = []

        for i in range(num_batch):
            sample = batch[i]
            mask = sample["is_na_residue_mask"]
            num_res = int(mask.sum().item())
            cluster_id = str(sample.get("cluster_id", sample.get("cluster_name", "unknown")))
            pdb_name = sample.get("pdb_name", None)

            # Output directory
            sample_dir = os.path.join(self._output_dir, cluster_id)
            if not ensemble_enabled:
                sample_dir = os.path.join(sample_dir, f"length_{num_res}")
            os.makedirs(sample_dir, exist_ok=True)

            # Generate samples
            context = {
                "single_embedding": sample["single_embedding"].repeat(num_generated, 1, 1),
                "pair_embedding": sample["pair_embedding"].repeat(num_generated, 1, 1, 1),
            }
            if self._steer_enabled:
                b_norm = du.normalize_b_factors(
                    sample["b_factors"].unsqueeze(0).expand(num_generated, -1),
                    mask.unsqueeze(0).expand(num_generated, -1),
                )
                atom37_traj, _, _, _qv = interpolant.sample_stochastic(
                    num_generated, num_res, self.model, self.amplitude_net,
                    b_norm, context=context,
                )
            else:
                atom37_traj, _, _ = interpolant.sample(
                    num_generated, num_res, self.model, context=context
                )
            pred_c4 = atom37_traj[-1][:, :, c4_idx]  # [num_generated, num_res, 3]

            # Write PDB files
            is_na = np.ones(num_res, dtype=np.int64)
            for j in range(num_generated):
                au.write_complex_to_pdbs(
                    du.to_numpy(atom37_traj[-1][j:j+1])[0],
                    os.path.join(sample_dir, f"{cluster_id}_conf{j+1}"),
                    is_na_residue_mask=is_na,
                )

            # GT C4' coords
            gt_torsions = sample["torsion_angles_sin_cos"][:, :8, :].reshape(1, -1, 16)
            gt_atoms = rna_all_atom.to_atom37_rna(
                sample["trans_1"].unsqueeze(0),
                sample["rotmats_1"].unsqueeze(0),
                mask.unsqueeze(0).bool(),
                torsions=gt_torsions,
            )
            gt_c4 = gt_atoms[:, :, c4_idx]  # [1, num_res, 3]

            # Single metrics (best RMSD/TM across all generated)
            single_result = metrics.compute_single_metrics(
                pred_c4, gt_c4, mask.unsqueeze(0)
            )

            out_row = {
                "cluster_id": cluster_id,
                "pdb_name": pdb_name,
                "length": num_res,
                "rmsd": float(single_result["rmsd"]),
                "tm_score": float(single_result["tm_score"]),
                "sample_dir": sample_dir,
            }

            # Ensemble metrics
            if ensemble_enabled and num_generated >= 2:
                gt_c4_ens = sample.get("gt_c4_ensemble", None)
                if gt_c4_ens is not None:
                    ens_result = metrics.compute_ensemble_metrics(pred_c4, gt_c4_ens, mask)
                    out_row.update(ens_result)

            outputs.append(out_row)

        return outputs
